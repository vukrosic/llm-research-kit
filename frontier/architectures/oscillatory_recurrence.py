"""
Oscillatory Recurrence Network — Rotating Memory with Content-Dependent Frequency
====================================================================================
A gated linear recurrence where the forget gate is a complex exponential
(damped rotation), not just a scalar decay. This means memory ROTATES
as it decays, naturally creating interference patterns across time.

Key difference from Mamba/GLA:
- Mamba: h[t] = decay * h[t-1] + B*x  (scalar decay, memory fades monotonically)
- Ours:  h[t] = R(freq,decay) * h[t-1] + B*x  (rotating decay, memory oscillates)

In real arithmetic:
  h_r[t] = decay * (cos(freq) * h_r[t-1] - sin(freq) * h_i[t-1]) + input_r[t]
  h_i[t] = decay * (sin(freq) * h_r[t-1] + cos(freq) * h_i[t-1]) + input_i[t]

The frequency and decay are INPUT-DEPENDENT (selective), so:
- Important tokens increase decay → longer memory
- Tokens at structural boundaries modulate frequency → change resonance
- The model learns which temporal frequencies to attend to per-token

This is like having a bank of input-dependent band-pass filters, where
each "oscillator" in the state resonates with different temporal patterns.

Complexity: O(n) sequential, parallelizable via chunked scan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any

from frontier.architectures.base import (
    FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
)
from frontier.architectures.registry import register_arch


# ──────────────────────────────────────────────────────
# Shared FFN
# ──────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff, bias=True):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=bias)
        self.up = nn.Linear(d, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ═══════════════════════════════════════════════════════
# OSCILLATORY RECURRENCE MIXER
# ═══════════════════════════════════════════════════════

class OscillatoryRecurrenceMixer(nn.Module):
    """
    Gated linear recurrence with complex (rotating) state.

    Each head maintains d_state oscillators. Each oscillator has:
    - A frequency (input-dependent): controls rotation speed
    - A decay (input-dependent): controls memory length
    - An input (projected from x): drives the oscillator

    The readout combines real and imaginary parts of the state
    to produce a content-dependent, temporally-structured output.
    """

    def __init__(self, d_model, n_heads=8, d_state=64, chunk_size=64):
        super().__init__()
        self.n_heads = n_heads
        self.d_state = d_state
        self.d_out = d_model
        self.chunk_size = chunk_size

        # Input projections: real + imaginary input to oscillators
        self.input_proj = nn.Linear(d_model, n_heads * d_state * 2, bias=False)

        # Input-dependent frequency and decay
        self.freq_proj = nn.Linear(d_model, n_heads * d_state, bias=True)
        self.decay_proj = nn.Linear(d_model, n_heads * d_state, bias=True)

        # Readout: combine real and imaginary parts
        self.readout = nn.Linear(n_heads * d_state * 2, d_model, bias=False)

        # Output gating
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)

        # Normalization on state readout
        self.state_norm = nn.RMSNorm(n_heads * d_state * 2)

        # Initialize frequency bias to cover a range of base frequencies
        with torch.no_grad():
            # Different heads get different base frequencies
            base_freqs = torch.linspace(0.01, 1.0, n_heads).unsqueeze(1).expand(n_heads, d_state)
            # Within each head, slight variation
            freq_noise = torch.randn(n_heads, d_state) * 0.1
            self.freq_proj.bias.copy_((base_freqs + freq_noise).reshape(-1))

            # Initialize decay bias so decay starts near 0.9 (long memory)
            # sigmoid(2.2) ≈ 0.9
            self.decay_proj.bias.fill_(2.2)

    def forward(self, x):
        B, L, D = x.shape
        H, S = self.n_heads, self.d_state

        # Project inputs
        inp = self.input_proj(x).view(B, L, H, S, 2)
        inp_real = inp[..., 0]  # (B, L, H, S)
        inp_imag = inp[..., 1]

        # Input-dependent frequency and decay
        freq = self.freq_proj(x).view(B, L, H, S)
        decay = torch.sigmoid(self.decay_proj(x)).view(B, L, H, S)

        # Precompute cos/sin
        cos_f = torch.cos(freq)
        sin_f = torch.sin(freq)

        # ── Chunked sequential scan ──
        # Process in chunks for better GPU utilization
        CHUNK = self.chunk_size
        h_real = torch.zeros(B, H, S, device=x.device, dtype=x.dtype)
        h_imag = torch.zeros(B, H, S, device=x.device, dtype=x.dtype)

        all_outputs_real = []
        all_outputs_imag = []

        for c_start in range(0, L, CHUNK):
            c_end = min(c_start + CHUNK, L)

            for t in range(c_start, c_end):
                d = decay[:, t]    # (B, H, S)
                c = cos_f[:, t]
                s = sin_f[:, t]

                # Rotating state update
                new_real = d * (c * h_real - s * h_imag) + inp_real[:, t]
                new_imag = d * (s * h_real + c * h_imag) + inp_imag[:, t]

                h_real = new_real
                h_imag = new_imag

                all_outputs_real.append(h_real)
                all_outputs_imag.append(h_imag)

        # Stack outputs: (B, L, H, S)
        out_real = torch.stack(all_outputs_real, dim=1)
        out_imag = torch.stack(all_outputs_imag, dim=1)

        # Combine real and imaginary for readout
        out = torch.cat([
            out_real.reshape(B, L, H * S),
            out_imag.reshape(B, L, H * S)
        ], dim=-1)  # (B, L, 2*H*S)

        out = self.state_norm(out)
        out = self.readout(out)  # (B, L, D)

        # Gated output
        gate = torch.sigmoid(self.gate_proj(x))
        return out * gate


# ═══════════════════════════════════════════════════════
# BLOCK AND MODEL
# ═══════════════════════════════════════════════════════

class OscillatoryRecurrenceBlock(nn.Module):
    def __init__(self, d, d_ff, n_heads=8, d_state=64, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = OscillatoryRecurrenceMixer(d, n_heads, d_state)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs

    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch(
    "OscillatoryRecurrenceLM",
    "novel",
    "Complex-valued gated recurrence with input-dependent rotating memory (oscillatory SSM)"
)
class OscillatoryRecurrenceLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        n_heads = ac.get("n_heads", 8)
        d_state = ac.get("d_state", 64)
        rs = ac.get("residual_scale", 1.0)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            OscillatoryRecurrenceBlock(d, config.d_ff, n_heads, d_state, rs, ac.get("use_bias", True))
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                # Don't overwrite carefully initialized biases
                pass
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))

    @classmethod
    def arch_family(cls):
        return "novel"

    def describe(self):
        ac = self.config.arch_config
        n_heads = ac.get("n_heads", 8)
        d_state = ac.get("d_state", 64)
        return (
            f"OscillatoryRecurrence: {self.config.n_layers}L, "
            f"{n_heads}H x {d_state}S complex rotating state, "
            f"input-dependent freq/decay, O(n)"
        )

    def supports_recurrent_inference(self):
        return True

    def sequence_mixing_complexity(self):
        return "O(n)"
