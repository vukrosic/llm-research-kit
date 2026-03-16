"""
Novel Architectures — Batch 9 (Break Plateau)
===============================================
All conv-based architectures plateau at ~4.04 with 6M tokens.
Hypothesis: the plateau is a training budget issue, not architecture.

New strategies:
1. MHConvPoolBestLM: The B6 winner retrained at 12M tokens (2x budget)
2. GatedConvResidualLM: Dense residual connections between conv layers (DenseNet-style)
3. ConvSandwichLM: Interleave fine (k=3) and coarse (k=31) conv layers
4. AdaptiveConvLM: Input-dependent kernel selection per token
5. ConvPyramidLM: Progressive channel expansion then contraction (hourglass)
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


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)


class MultiHeadConvMixerB9(nn.Module):
    def __init__(self, d_model, n_heads=8, max_kernel=65):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        kernel_sizes = [min(2**(i+1) + 1, max_kernel) for i in range(n_heads)]
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.head_convs = nn.ModuleList([
            nn.Conv1d(self.d_head, self.d_head, ks, padding=ks - 1, groups=self.d_head)
            for ks in kernel_sizes
        ])
        self.gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head)
        head_outs = []
        for h in range(self.n_heads):
            vh = v[:, :, h]
            out = self.head_convs[h](vh.transpose(1, 2))[:, :, :L].transpose(1, 2)
            head_outs.append(F.silu(out))
        combined = torch.cat(head_outs, dim=-1)
        return self.out(combined * torch.sigmoid(self.gate(x)))


class CausalPoolMixerB9(nn.Module):
    def __init__(self, d_model, scales=(2, 4, 8, 16)):
        super().__init__()
        self.scales = scales
        self.projs = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in scales])
        self.scale_gate = nn.Linear(d_model, len(scales))
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        cumsum = torch.cumsum(x, dim=1)
        pooled_outs = []
        for scale, proj in zip(self.scales, self.projs):
            shifted = F.pad(cumsum[:, :-scale], (0, 0, scale, 0))
            pooled = cumsum - shifted
            denom = torch.arange(1, L+1, device=x.device, dtype=x.dtype).clamp(max=scale).unsqueeze(0).unsqueeze(-1)
            pooled_outs.append(F.silu(proj(pooled / denom)))
        gates = F.softmax(self.scale_gate(x), dim=-1)
        stacked = torch.stack(pooled_outs, dim=-1)
        result = (stacked * gates.unsqueeze(2)).sum(-1)
        return self.out(result * torch.sigmoid(self.gate(x)))


class MHConvPoolBlockB9(nn.Module):
    def __init__(self, d, d_ff, layer_idx, n_heads=8, pool_every=3, scales=(2,4,8,16), rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        if layer_idx % pool_every == pool_every - 1:
            self.mix = CausalPoolMixerB9(d, scales)
        else:
            self.mix = MultiHeadConvMixerB9(d, n_heads)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


# ═══════════════════════════════════════════════════════
# 1. MHCONV POOL BEST — 12M tokens training (2x budget)
# ═══════════════════════════════════════════════════════

@register_arch("MHConvPoolBestLM", "novel", "Best MHConvPool config retrained with 12M tokens")
class MHConvPoolBestLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        ps = tuple(ac.get("pool_scales", [2,4,8,16]))
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            MHConvPoolBlockB9(d, config.d_ff, i, nh, 3, ps, rs, ac.get("use_bias", True))
            for i in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_init_weights)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"MHConvPoolBest: {self.config.n_layers}L, {self.config.train_tokens/1e6:.0f}M tok"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 2. GATED CONV RESIDUAL — Dense skip connections
# ═══════════════════════════════════════════════════════

class GatedDenseConvBlock(nn.Module):
    """Each layer gets a gated skip from the previous 2 layers (DenseNet-style)."""
    def __init__(self, d, d_ff, layer_idx, n_heads=8, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = MultiHeadConvMixerB9(d, n_heads)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
        self.layer_idx = layer_idx
        # Dense connection gate (combines current + up to 2 previous layers)
        if layer_idx >= 1:
            self.dense_gate = nn.Linear(d, d)
            self.dense_proj = nn.Linear(d, d, bias=False)

    def forward(self, x, prev=None):
        h = x + self.rs * self.mix(self.n1(x))
        h = h + self.rs * self.ffn(self.n2(h))
        # Dense skip connection
        if prev is not None and self.layer_idx >= 1:
            gate = torch.sigmoid(self.dense_gate(h))
            h = h + gate * self.dense_proj(prev)
        return h


@register_arch("GatedConvResidualLM", "novel", "MHConv with dense gated residual connections")
class GatedConvResidualLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            GatedDenseConvBlock(d, config.d_ff, i, nh, rs, ac.get("use_bias", True))
            for i in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_init_weights)
    def forward(self, x):
        h = self.embed(x)
        prev = None
        for b in self.blocks:
            new_h = b(h, prev)
            prev = h
            h = new_h
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"GatedConvResidual: {self.config.n_layers}L, dense skips"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 3. CONV SANDWICH — Alternating fine and coarse conv
# ═══════════════════════════════════════════════════════

class FineConvMixer(nn.Module):
    """Short-range convolution: all heads use small kernels (3-7)."""
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        kernel_sizes = [3, 3, 5, 5, 7, 7, 3, 5][:n_heads]
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.head_convs = nn.ModuleList([
            nn.Conv1d(self.d_head, self.d_head, ks, padding=ks-1, groups=self.d_head)
            for ks in kernel_sizes
        ])
        self.gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head)
        outs = []
        for h in range(self.n_heads):
            o = self.head_convs[h](v[:,:,h].transpose(1,2))[:,:,:L].transpose(1,2)
            outs.append(F.silu(o))
        return self.out(torch.cat(outs, -1) * torch.sigmoid(self.gate(x)))


class CoarseConvMixer(nn.Module):
    """Long-range convolution: all heads use large kernels (15-63)."""
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        kernel_sizes = [15, 21, 31, 31, 45, 45, 63, 63][:n_heads]
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.head_convs = nn.ModuleList([
            nn.Conv1d(self.d_head, self.d_head, ks, padding=ks-1, groups=self.d_head)
            for ks in kernel_sizes
        ])
        self.gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head)
        outs = []
        for h in range(self.n_heads):
            o = self.head_convs[h](v[:,:,h].transpose(1,2))[:,:,:L].transpose(1,2)
            outs.append(F.silu(o))
        return self.out(torch.cat(outs, -1) * torch.sigmoid(self.gate(x)))


class ConvSandwichBlock(nn.Module):
    def __init__(self, d, d_ff, layer_idx, n_heads=8, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        # Alternate: fine → coarse → fine → coarse
        if layer_idx % 2 == 0:
            self.mix = FineConvMixer(d, n_heads)
        else:
            self.mix = CoarseConvMixer(d, n_heads)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("ConvSandwichLM", "novel", "Alternating fine (k=3-7) and coarse (k=15-63) conv layers")
class ConvSandwichLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            ConvSandwichBlock(d, config.d_ff, i, nh, rs, ac.get("use_bias", True))
            for i in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_init_weights)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"ConvSandwich: {self.config.n_layers}L, fine/coarse alternating"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 4. ADAPTIVE CONV — Input-dependent kernel selection
# ═══════════════════════════════════════════════════════

class AdaptiveConvMixer(nn.Module):
    """
    Multiple conv heads with different kernels, but the model learns
    per-token weights over which kernels to use (soft routing).
    """
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        kernel_sizes = [3, 5, 9, 17, 33, 33, 65, 65][:n_heads]

        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.head_convs = nn.ModuleList([
            nn.Conv1d(self.d_head, self.d_head, ks, padding=ks-1, groups=self.d_head)
            for ks in kernel_sizes
        ])
        # Per-token routing over heads
        self.router = nn.Linear(d_model, n_heads)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head)

        # Get all head outputs
        head_outs = []
        for h in range(self.n_heads):
            vh = v[:, :, h]
            out = self.head_convs[h](vh.transpose(1, 2))[:, :, :L].transpose(1, 2)
            head_outs.append(F.silu(out))

        stacked = torch.stack(head_outs, dim=-1)  # (B, L, d_head, n_heads)

        # Soft routing weights per-token
        weights = F.softmax(self.router(x), dim=-1)  # (B, L, n_heads)

        # Weighted sum across heads for each d_head group
        # Reshape for broadcasting
        weighted = (stacked * weights.unsqueeze(2)).sum(-1)  # (B, L, d_head)
        # But we have n_heads groups of d_head each... need to route per group
        # Actually, let each head output its d_head, then concat
        # Route = weighted selection among all heads for the full d_model
        result = torch.cat(head_outs, dim=-1)  # (B, L, D)
        # Apply routing as a residual modulation
        route_scale = weights.repeat_interleave(self.d_head, dim=-1)  # (B, L, D)
        return self.out(result * route_scale * self.n_heads)


class AdaptiveConvBlock(nn.Module):
    def __init__(self, d, d_ff, n_heads=8, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = AdaptiveConvMixer(d, n_heads)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("AdaptiveConvLM", "novel", "Input-dependent soft routing over multi-kernel conv heads")
class AdaptiveConvLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            AdaptiveConvBlock(d, config.d_ff, nh, rs, ac.get("use_bias", True))
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_init_weights)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"AdaptiveConv: {self.config.n_layers}L, routed kernel selection"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 5. CONV PYRAMID — Hourglass shape (expand then contract)
# ═══════════════════════════════════════════════════════

class ConvPyramidBlock(nn.Module):
    """Conv block with variable d_ff based on position in the hourglass."""
    def __init__(self, d, d_ff, n_heads=8, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = MultiHeadConvMixerB9(d, n_heads)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("ConvPyramidLM", "novel", "MHConv with hourglass FFN width (wide middle, narrow ends)")
class ConvPyramidLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers

        # Hourglass: FFN width expands to middle then contracts
        base_ff = config.d_ff
        ff_widths = []
        for i in range(n):
            # Triangle: peaks at middle
            progress = abs(i - n/2) / (n/2)  # 1 at edges, 0 at middle
            ff = int(base_ff * (1.0 + 1.0 * (1.0 - progress)))  # 1x at edges, 2x at middle
            ff_widths.append(ff)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            ConvPyramidBlock(d, ff_widths[i], nh, rs, ac.get("use_bias", True))
            for i in range(n)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_init_weights)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"ConvPyramid: {self.config.n_layers}L, hourglass FFN"
    def sequence_mixing_complexity(self): return "O(n)"
