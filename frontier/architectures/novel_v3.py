"""
Novel Architectures — Batch 3 (Exploiting StripedConv Win)
============================================================
StripedConv beat the transformer. Now exploit that finding:
1. StripedConvDeep: More layers (20L) with narrower FFN to stay at 88M
2. StripedConvGated: Add per-layer learned gating between conv widths
3. ConvRecurrent: StripedConv layers + a few gated recurrence layers for global memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from frontier.architectures.base import (
    FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
)
from frontier.architectures.registry import register_arch


class SwiGLU(nn.Module):
    def __init__(self, d, d_ff, bias=True):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=bias)
        self.up = nn.Linear(d, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d, bias=bias)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ═══════════════════════════════════════════════════════
# 1. STRIPED CONV DEEP (20L, narrower FFN)
# ═══════════════════════════════════════════════════════

class StripedConvMixerV3(nn.Module):
    """Multi-width depthwise causal convolutions with channel mixing."""
    def __init__(self, d_model, kernel_sizes=(3, 7, 15, 31)):
        super().__init__()
        self.n_strips = len(kernel_sizes)
        self.d_per = d_model // self.n_strips
        self.d_first = d_model - self.d_per * (self.n_strips - 1)

        self.convs = nn.ModuleList()
        for i, ks in enumerate(kernel_sizes):
            d = self.d_first if i == 0 else self.d_per
            self.convs.append(nn.Conv1d(d, d, ks, padding=ks - 1, groups=d, bias=True))
        self.channel_mix = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        splits = [self.d_first] + [self.d_per] * (self.n_strips - 1)
        chunks = x.split(splits, dim=-1)
        conv_outs = []
        for chunk, conv in zip(chunks, self.convs):
            h = conv(chunk.transpose(1, 2))[:, :, :L].transpose(1, 2)
            conv_outs.append(h)
        out = torch.cat(conv_outs, dim=-1)
        out = F.silu(self.channel_mix(out))
        return self.out(out * torch.sigmoid(self.gate(x)))


class StripedConvBlockV3(nn.Module):
    def __init__(self, d, d_ff, kernel_sizes=(3, 7, 15, 31), rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = StripedConvMixerV3(d, kernel_sizes)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("StripedConvDeepLM", "novel", "Deep StripedConv: 20L with narrower FFN for 88M params")
class StripedConvDeepLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        ks = tuple(ac.get("kernel_sizes", [3, 7, 15, 31]))
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            StripedConvBlockV3(d, config.d_ff, ks, rs, ac.get("use_bias", True))
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
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"StripedConvDeep: {self.config.n_layers}L, deeper narrower variant"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 2. STRIPED CONV GATED (per-width learned gating)
# ═══════════════════════════════════════════════════════

class GatedStripedConvMixer(nn.Module):
    """Each conv width has its own learned importance gate."""
    def __init__(self, d_model, kernel_sizes=(3, 7, 15, 31, 63)):
        super().__init__()
        self.n_strips = len(kernel_sizes)
        # Each strip processes full d_model but gated
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, ks, padding=ks - 1, groups=d_model, bias=True)
            for ks in kernel_sizes
        ])
        # Learned importance per strip
        self.strip_gate = nn.Linear(d_model, self.n_strips)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        # Run all convolutions in parallel
        conv_outs = []
        for conv in self.convs:
            h = conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
            conv_outs.append(F.silu(h))
        stacked = torch.stack(conv_outs, dim=-1)  # (B, L, D, n_strips)

        # Per-token, per-strip gating
        gates = F.softmax(self.strip_gate(x), dim=-1)  # (B, L, n_strips)
        mixed = (stacked * gates.unsqueeze(2)).sum(-1)  # (B, L, D)
        return self.out(mixed)


class GatedStripedConvBlock(nn.Module):
    def __init__(self, d, d_ff, kernel_sizes=(3, 7, 15, 31, 63), rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = GatedStripedConvMixer(d, kernel_sizes)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("StripedConvGatedLM", "novel", "StripedConv with per-strip learned gating")
class StripedConvGatedLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        ks = tuple(ac.get("kernel_sizes", [3, 7, 15, 31, 63]))
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            GatedStripedConvBlock(d, config.d_ff, ks, rs, ac.get("use_bias", True))
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
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"StripedConvGated: {self.config.n_layers}L, per-strip gating"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 3. CONV-RECURRENT HYBRID
# ═══════════════════════════════════════════════════════
# 80% StripedConv layers + 20% lightweight gated recurrence layers.
# Conv handles local patterns, recurrence handles global memory.
# The recurrence uses a SMALL state (d_state=16) to avoid OOM.

class LightRecurrence(nn.Module):
    """Lightweight gated recurrence with small state, using cumsum trick."""
    def __init__(self, d_model, expand=2):
        super().__init__()
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, 4, padding=3, groups=self.d_inner)

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # Local conv
        x_in = self.conv(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_in = F.silu(x_in)

        gate = torch.sigmoid(z)

        # Exponential moving average via cumsum
        # h_t = gate * h_{t-1} + (1-gate) * x_t
        log_g = torch.log(gate.clamp(min=1e-6))
        cum_log_g = torch.cumsum(log_g, dim=1)

        inp_weighted = (1 - gate) * x_in
        # Apply decay: multiply each input by exp(-sum of future log_gates)
        weighted = inp_weighted * torch.exp(-cum_log_g)
        cum = torch.cumsum(weighted, dim=1)
        out = cum * torch.exp(cum_log_g)

        return self.out_proj(out)


class ConvRecurrentBlock(nn.Module):
    """Uses StripedConv or LightRecurrence based on layer index."""
    def __init__(self, d, d_ff, layer_idx, total_layers, kernel_sizes=(3, 7, 15, 31, 63), rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs

        # Every 5th layer is recurrence, rest is conv
        if layer_idx % 5 == 4:
            self.mix = LightRecurrence(d, expand=2)
        else:
            self.mix = StripedConvMixerV3(d, kernel_sizes)

    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("ConvRecurrentLM", "novel", "StripedConv + lightweight recurrence every 5th layer")
class ConvRecurrentLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        ks = tuple(ac.get("kernel_sizes", [3, 7, 15, 31, 63]))
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            ConvRecurrentBlock(d, config.d_ff, i, n, ks, rs, ac.get("use_bias", True))
            for i in range(n)
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
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"ConvRecurrent: {self.config.n_layers}L, conv + recurrence hybrid"
    def supports_recurrent_inference(self): return True
    def sequence_mixing_complexity(self): return "O(n)"
