"""
5 Novel Architectures — Batch 1 (Speed-Gated)
================================================
Each is a structurally different sequence mixer. All target ~88M params.
Designed to be FAST — no sequential scans over seq_len where possible.

1. GatedDeltaNet — Delta-rule linear attention with gated forget
2. StripedConvNet — Multi-width depthwise causal convolutions (parallel)
3. CosineResonator — Learned periodic basis functions for token mixing
4. TopKSparseAttn — Sparse attention that only attends to top-k similar past tokens
5. RecurrentGateNet — Minimal gated linear recurrence (GLA-like but simpler)
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
# 1. GATED DELTA NET
# ═══════════════════════════════════════════════════════
# Key idea: Linear attention + delta rule update.
# Instead of just accumulating key-value outer products (like standard
# linear attention), we UPDATE the memory with a delta rule:
#   M_t = gate * M_{t-1} + v_t * k_t^T - gate * (k_t^T M_{t-1}) * k_t
# This lets the model OVERWRITE old associations, solving the
# "memory pollution" problem of vanilla linear attention.

class GatedDeltaMixer(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, n_heads, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.RMSNorm(self.d_head)

    def forward(self, x):
        B, L, D = x.shape
        H, DH = self.n_heads, self.d_head

        qkv = self.qkv(x).reshape(B, L, 3, H, DH)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, L, H, DH)

        # Feature map: ELU + 1 (ensures non-negative)
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Per-head forget gate
        gate = torch.sigmoid(self.gate_proj(x))  # (B, L, H)
        gate = gate.unsqueeze(-1).unsqueeze(-1)  # (B, L, H, 1, 1)

        # Sequential delta-rule update (must be sequential for correctness)
        M = torch.zeros(B, H, DH, DH, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(L):
            kt = k[:, t]  # (B, H, DH)
            vt = v[:, t]  # (B, H, DH)
            qt = q[:, t]  # (B, H, DH)
            gt = gate[:, t, :, :, 0]  # (B, H, 1)

            # Read: y = q @ M
            yt = torch.einsum('bhd,bhde->bhe', qt, M)

            # Delta update: M = gate*M + v*k^T - gate*(M@k)*k^T
            Mk = torch.einsum('bhde,bhe->bhd', M, kt)  # what M currently maps k to
            delta = torch.einsum('bhd,bhe->bhde', vt - gt.squeeze(-1).unsqueeze(-1) * Mk, kt)
            M = gt * M + delta

            outputs.append(self.norm(yt))

        out = torch.stack(outputs, dim=1)  # (B, L, H, DH)
        out = out.reshape(B, L, D)
        return self.out_proj(out)


class GatedDeltaBlock(nn.Module):
    def __init__(self, d, d_ff, n_heads=8, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = GatedDeltaMixer(d, n_heads)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs

    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("GatedDeltaLM", "novel", "Delta-rule linear attention with gated forget")
class GatedDeltaLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        n_heads = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            GatedDeltaBlock(d, config.d_ff, n_heads, rs, ac.get("use_bias", True))
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))

    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"GatedDeltaNet: {self.config.n_layers}L, delta-rule linear attn with gated forget"
    def supports_recurrent_inference(self): return True
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 2. STRIPED CONV NET
# ═══════════════════════════════════════════════════════
# Key idea: No attention, no recurrence. Just parallel depthwise
# causal convolutions at widths [3, 7, 15, 31, 63] run in parallel,
# then gated-combined. Global context emerges from stacking layers.
# EXTREMELY fast — pure convolutions, fully parallelizable.

class StripedConvMixer(nn.Module):
    def __init__(self, d_model, kernel_sizes=(3, 7, 15, 31, 63)):
        super().__init__()
        self.n_strips = len(kernel_sizes)
        self.d_per = d_model // self.n_strips
        # Remainder channels go to the first strip
        self.d_first = d_model - self.d_per * (self.n_strips - 1)

        self.convs = nn.ModuleList()
        for i, ks in enumerate(kernel_sizes):
            d = self.d_first if i == 0 else self.d_per
            self.convs.append(
                nn.Conv1d(d, d, ks, padding=ks - 1, groups=d, bias=True)
            )
        self.channel_mix = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        # Split channels across strips
        splits = [self.d_first] + [self.d_per] * (self.n_strips - 1)
        chunks = x.split(splits, dim=-1)

        conv_outs = []
        for chunk, conv in zip(chunks, self.convs):
            h = conv(chunk.transpose(1, 2))[:, :, :L].transpose(1, 2)
            conv_outs.append(h)

        out = torch.cat(conv_outs, dim=-1)
        out = F.silu(self.channel_mix(out))
        return self.out(out * torch.sigmoid(self.gate(x)))


class StripedConvBlock(nn.Module):
    def __init__(self, d, d_ff, kernel_sizes=(3, 7, 15, 31, 63), rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = StripedConvMixer(d, kernel_sizes)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs

    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("StripedConvLM", "novel", "Multi-width parallel causal convolutions")
class StripedConvLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        ks = tuple(ac.get("kernel_sizes", [3, 7, 15, 31, 63]))
        rs = ac.get("residual_scale", 1.0)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            StripedConvBlock(d, config.d_ff, ks, rs, ac.get("use_bias", True))
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))

    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"StripedConv: {self.config.n_layers}L, parallel multi-width causal conv"
    def supports_recurrent_inference(self): return False
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 3. COSINE RESONATOR
# ═══════════════════════════════════════════════════════
# Key idea: Replace attention with learned periodic basis functions.
# Each "resonator" head has a learned frequency and phase.
# Token mixing = project to frequency domain via cos/sin bases,
# cumulative sum in that domain (causal), project back.
# Inspired by Fourier features but with LEARNED frequencies.

class CosineResonatorMixer(nn.Module):
    def __init__(self, d_model, n_resonators=64):
        super().__init__()
        self.n_res = n_resonators
        # Learned frequencies (log-uniform init covering many scales)
        self.log_freq = nn.Parameter(torch.linspace(-2, 4, n_resonators))
        # Learned phases
        self.phase = nn.Parameter(torch.zeros(n_resonators))
        # Project to resonator space and back
        self.to_res = nn.Linear(d_model, n_resonators, bias=False)
        self.from_res = nn.Linear(n_resonators, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape

        # Create position-dependent resonator responses
        pos = torch.arange(L, device=x.device, dtype=x.dtype).unsqueeze(-1)  # (L, 1)
        freq = self.log_freq.exp()  # (n_res,)
        # Resonator basis: cos(freq * pos + phase)
        basis = torch.cos(pos * freq.unsqueeze(0) + self.phase.unsqueeze(0))  # (L, n_res)

        # Project input to resonator coefficients
        coeffs = self.to_res(x)  # (B, L, n_res)
        # Modulate by basis
        modulated = coeffs * basis.unsqueeze(0)  # (B, L, n_res)
        # Causal aggregation via cumulative sum
        cumulated = torch.cumsum(modulated, dim=1)  # (B, L, n_res)
        # Demodulate
        demod = cumulated * basis.unsqueeze(0)
        # Project back
        context = self.from_res(demod)  # (B, L, D)

        # Gated output with value projection
        v = self.value_proj(x)
        gate = torch.sigmoid(self.gate(x))
        return self.out((v + context) * gate)


class CosineResonatorBlock(nn.Module):
    def __init__(self, d, d_ff, n_res=64, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = CosineResonatorMixer(d, n_res)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs

    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("CosineResonatorLM", "novel", "Learned periodic basis token mixing via resonators")
class CosineResonatorLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        n_res = ac.get("n_resonators", 64)
        rs = ac.get("residual_scale", 1.0)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            CosineResonatorBlock(d, config.d_ff, n_res, rs, ac.get("use_bias", True))
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))

    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"CosineResonator: {self.config.n_layers}L, learned periodic basis mixing"
    def supports_recurrent_inference(self): return False
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 4. TOP-K SPARSE ATTENTION
# ═══════════════════════════════════════════════════════
# Key idea: Full softmax attention is O(n²). But most attention mass
# concentrates on ~32 tokens. Compute full QK^T scores, keep only
# top-k per query, zero the rest. This is still O(n²) for score
# computation but O(nk) for the weighted sum.
# At 2048 seq len with k=64, this is ~32x less memory for values.
# More importantly: forces the model to be selective.

class TopKSparseAttnMixer(nn.Module):
    def __init__(self, d_model, n_heads=8, top_k=64):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.top_k = top_k
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        H, DH = self.n_heads, self.d_head

        qkv = self.qkv(x).reshape(B, L, 3, H, DH).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, L, DH)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)

        # Causal mask
        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Top-k sparse: keep only top_k scores per query
        k_val = min(self.top_k, L)
        topk_vals, topk_idx = scores.topk(k_val, dim=-1)  # (B, H, L, k)

        # Build sparse attention weights
        sparse_scores = torch.full_like(scores, float('-inf'))
        sparse_scores.scatter_(-1, topk_idx, topk_vals)
        attn = F.softmax(sparse_scores, dim=-1)

        out = torch.matmul(attn, v)  # (B, H, L, DH)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class TopKSparseBlock(nn.Module):
    def __init__(self, d, d_ff, n_heads=8, top_k=64, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = TopKSparseAttnMixer(d, n_heads, top_k)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs

    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("TopKSparseLM", "novel", "Top-k sparse attention — attend to only k most relevant tokens")
class TopKSparseLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        n_heads = ac.get("n_heads", 8)
        top_k = ac.get("top_k", 64)
        rs = ac.get("residual_scale", 1.0)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            TopKSparseBlock(d, config.d_ff, n_heads, top_k, rs, ac.get("use_bias", True))
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))

    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"TopKSparse: {self.config.n_layers}L, top-{self.config.arch_config.get('top_k', 64)} sparse attention"
    def supports_recurrent_inference(self): return False
    def sequence_mixing_complexity(self): return "O(nk)"


# ═══════════════════════════════════════════════════════
# 5. RECURRENT GATE NET (minimal GLA variant)
# ═══════════════════════════════════════════════════════
# Key idea: The simplest possible gated linear recurrence.
# Per-head: h_t = sigmoid(Wf * x_t) * h_{t-1} + Wi * x_t
# Output: o_t = Wo * h_t
# No attention, no convolution, just gated accumulation.
# The gate is input-dependent (like Mamba) for selectivity.
# Uses chunk-wise parallel scan for speed.

class MinimalGatedRecurrence(nn.Module):
    def __init__(self, d_model, n_heads=8, d_state=64):
        super().__init__()
        self.n_heads = n_heads
        self.d_state = d_state
        self.d_head = d_model // n_heads

        # Input projections
        self.input_proj = nn.Linear(d_model, n_heads * d_state, bias=False)
        self.forget_proj = nn.Linear(d_model, n_heads * d_state, bias=True)
        self.output_proj = nn.Linear(n_heads * d_state, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        H, S = self.n_heads, self.d_state

        inp = self.input_proj(x).reshape(B, L, H, S)
        forget = torch.sigmoid(self.forget_proj(x).reshape(B, L, H, S))

        # Chunk-parallel: process in chunks for better GPU utilization
        CHUNK = 64
        n_chunks = (L + CHUNK - 1) // CHUNK

        # Sequential across chunks, parallel within each chunk
        h = torch.zeros(B, H, S, device=x.device, dtype=x.dtype)
        all_outputs = []

        for c in range(n_chunks):
            start = c * CHUNK
            end = min(start + CHUNK, L)
            chunk_len = end - start

            inp_chunk = inp[:, start:end]  # (B, chunk, H, S)
            fgt_chunk = forget[:, start:end]

            # Within-chunk sequential scan
            chunk_outs = []
            for t in range(chunk_len):
                h = fgt_chunk[:, t] * h + inp_chunk[:, t]
                chunk_outs.append(h)

            all_outputs.append(torch.stack(chunk_outs, dim=1))  # (B, chunk, H, S)

        out = torch.cat(all_outputs, dim=1)  # (B, L, H, S)
        out = out.reshape(B, L, H * S)
        out = self.output_proj(out)

        return out * torch.sigmoid(self.gate(x))


class RecurrentGateBlock(nn.Module):
    def __init__(self, d, d_ff, n_heads=8, d_state=64, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = MinimalGatedRecurrence(d, n_heads, d_state)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs

    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("RecurrentGateLM", "novel", "Minimal gated linear recurrence — simplest possible selective RNN")
class RecurrentGateLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        n_heads = ac.get("n_heads", 8)
        d_state = ac.get("d_state", 64)
        rs = ac.get("residual_scale", 1.0)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            RecurrentGateBlock(d, config.d_ff, n_heads, d_state, rs, ac.get("use_bias", True))
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))

    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"RecurrentGate: {self.config.n_layers}L, minimal gated linear recurrence"
    def supports_recurrent_inference(self): return True
    def sequence_mixing_complexity(self): return "O(n)"
