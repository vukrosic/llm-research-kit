"""
Wave Interference Network v2 — Physics-Inspired Sequence Mixing
================================================================
Two-stage wave interference with multiplicative gating for content-content
interaction, plus short-range causal depthwise convolution.

Stage 1: FFT convolution with damped oscillatory kernels (long-range)
Gate:    Multiplicative interaction with input-dependent values
Stage 2: FFT convolution with second set of kernels (creates implicit
         content-content interaction through the gate)

This addresses v1's weakness: single convolution lacks the multiplicative
content-content interaction that makes attention powerful. The two-stage
design creates O(n log n) implicit pairwise interaction through
convolve → gate → convolve.

Also includes a short-range depthwise causal conv (like Mamba's d_conv)
for capturing local n-gram patterns cheaply.
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
# FFT CAUSAL CONVOLUTION HELPER
# ═══════════════════════════════════════════════════════

def _fft_causal_conv(emission, kernel, seq_len):
    """
    Causal convolution via FFT. Handles bf16→f32 casting.
    emission: (B, n_bands, L, d_band)
    kernel:   (n_bands, L, d_band)
    Returns:  (B, n_bands, L, d_band)
    """
    fft_n = 2 * seq_len  # linear (non-circular) convolution
    orig_dtype = emission.dtype

    E_fft = torch.fft.rfft(emission.float(), n=fft_n, dim=2)
    K_fft = torch.fft.rfft(kernel.float(), n=fft_n, dim=1)
    Y_fft = E_fft * K_fft.unsqueeze(0)
    y = torch.fft.irfft(Y_fft, n=fft_n, dim=2)[:, :, :seq_len, :]

    return y.to(orig_dtype)


def _build_oscillatory_kernel(log_freq, freq_offset, log_decay, band_scale, seq_len, device, dtype):
    """
    Build causal damped oscillatory kernel.
    k(t) = scale * exp(-decay * t) * cos(freq * t)
    Returns: (n_bands, seq_len, d_band)
    """
    t = torch.arange(seq_len, device=device, dtype=torch.float32)

    freq = torch.exp(log_freq.float()) + freq_offset.float()  # (n_bands, d_band)
    decay = F.softplus(log_decay.float())  # (n_bands, d_band) positive

    # (n_bands, L, d_band)
    exp_decay = torch.exp(-decay.unsqueeze(1) * t[None, :, None])
    oscillation = torch.cos(freq.unsqueeze(1) * t[None, :, None])

    kernel = band_scale.float().unsqueeze(1) * exp_decay * oscillation
    return kernel.to(dtype)


# ═══════════════════════════════════════════════════════
# WAVE INTERFERENCE MIXER V2 — TWO-STAGE WITH GATING
# ═══════════════════════════════════════════════════════

class WaveInterferenceMixerV2(nn.Module):
    """
    Two-stage wave interference with multiplicative content gating.

    1. Project input to emissions and values
    2. Stage 1: convolve emissions with oscillatory kernel (long-range context)
    3. Gate: multiply convolved signal with SiLU(values) — content-content interaction
    4. Short conv: causal depthwise conv for local patterns
    5. Stage 2: convolve again with second kernel (creates pairwise interaction)
    6. Output projection
    """

    def __init__(self, d_model, n_bands=8, d_conv=4):
        super().__init__()
        self.n_bands = n_bands
        self.d_band = d_model // n_bands
        assert d_model % n_bands == 0

        # Input projections (emission + value for gating)
        self.emit_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)

        # Short-range causal depthwise conv (like Mamba's d_conv)
        self.short_conv = nn.Conv1d(
            d_model, d_model, d_conv,
            padding=d_conv - 1, groups=d_model, bias=True
        )

        # Stage 1 kernel parameters
        freq_init_1 = torch.linspace(math.log(0.01), math.log(2.0), n_bands)
        self.log_freq_1 = nn.Parameter(
            freq_init_1.unsqueeze(1).expand(n_bands, self.d_band).clone()
        )
        self.freq_offset_1 = nn.Parameter(torch.randn(n_bands, self.d_band) * 0.05)
        decay_init_1 = torch.linspace(math.log(0.01), math.log(0.3), n_bands)
        self.log_decay_1 = nn.Parameter(
            decay_init_1.unsqueeze(1).expand(n_bands, self.d_band).clone()
        )
        self.band_scale_1 = nn.Parameter(torch.ones(n_bands, 1) * 0.5)

        # Stage 2 kernel parameters (different frequencies for diversity)
        freq_init_2 = torch.linspace(math.log(0.05), math.log(3.0), n_bands)
        self.log_freq_2 = nn.Parameter(
            freq_init_2.unsqueeze(1).expand(n_bands, self.d_band).clone()
        )
        self.freq_offset_2 = nn.Parameter(torch.randn(n_bands, self.d_band) * 0.05)
        decay_init_2 = torch.linspace(math.log(0.02), math.log(0.5), n_bands)
        self.log_decay_2 = nn.Parameter(
            decay_init_2.unsqueeze(1).expand(n_bands, self.d_band).clone()
        )
        self.band_scale_2 = nn.Parameter(torch.ones(n_bands, 1) * 0.5)

        # Output
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape

        # Input projections
        emission = self.emit_proj(x)  # (B, L, D)
        value = self.value_proj(x)    # (B, L, D)

        # Reshape emission to bands: (B, n_bands, L, d_band)
        h = emission.view(B, L, self.n_bands, self.d_band).permute(0, 2, 1, 3)

        # ── Stage 1: long-range oscillatory convolution ──
        kernel_1 = _build_oscillatory_kernel(
            self.log_freq_1, self.freq_offset_1, self.log_decay_1,
            self.band_scale_1, L, x.device, x.dtype
        )
        h = _fft_causal_conv(h, kernel_1, L)

        # Reshape back: (B, L, D)
        h = h.permute(0, 2, 1, 3).reshape(B, L, D)

        # ── Gate: multiplicative content-content interaction ──
        h = h * F.silu(value)

        # ── Short-range causal conv for local patterns ──
        h = self.short_conv(h.transpose(1, 2))[:, :, :L].transpose(1, 2)

        # ── Stage 2: second oscillatory convolution ──
        h = h.view(B, L, self.n_bands, self.d_band).permute(0, 2, 1, 3)
        kernel_2 = _build_oscillatory_kernel(
            self.log_freq_2, self.freq_offset_2, self.log_decay_2,
            self.band_scale_2, L, x.device, x.dtype
        )
        h = _fft_causal_conv(h, kernel_2, L)
        h = h.permute(0, 2, 1, 3).reshape(B, L, D)

        return self.out_proj(h)


# ═══════════════════════════════════════════════════════
# BLOCK AND MODEL
# ═══════════════════════════════════════════════════════

class WaveInterferenceBlockV2(nn.Module):
    def __init__(self, d, d_ff, n_bands=8, d_conv=4, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = WaveInterferenceMixerV2(d, n_bands, d_conv)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs

    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch(
    "WaveInterferenceV2LM",
    "novel",
    "Two-stage wave interference: convolve→gate→convolve with oscillatory kernels + local conv"
)
class WaveInterferenceV2LM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        n_bands = ac.get("n_bands", 8)
        d_conv = ac.get("d_conv", 4)
        rs = ac.get("residual_scale", 1.0)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            WaveInterferenceBlockV2(d, config.d_ff, n_bands, d_conv, rs, ac.get("use_bias", True))
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
                nn.init.zeros_(m.bias)
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
        n_bands = ac.get("n_bands", 8)
        return (
            f"WaveInterferenceV2: {self.config.n_layers}L, "
            f"two-stage {n_bands}-band oscillatory FFT conv + gating + local conv, "
            f"O(n log n)"
        )

    def supports_recurrent_inference(self):
        return False

    def sequence_mixing_complexity(self):
        return "O(n log n)"
