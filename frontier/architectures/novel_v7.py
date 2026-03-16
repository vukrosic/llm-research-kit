"""
Novel Architectures — Batch 7 (Push Past 4.0)
===============================================
Current best: MHConvPool 4.0375 (MultiHeadConv + causal pooling every 3rd layer)

Strategy: Deep exploitation of winning formula + 2 radical new ideas
1. MHConvPoolDeepLM: MHConvPool at 20L with narrower FFN (depth exploit)
2. MHConvPoolWideLM: More heads (16), wider kernels
3. StripedMHConvLM: Combine StripedConv parallel channels with MultiHead structure
4. TokenShiftConvLM: RWKV-style token shift combined with multi-head conv
5. ConvDiffPoolLM: Three-way hybrid: conv + diffusion + pool layers
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


class MultiHeadConvMixerV3(nn.Module):
    """Multi-head conv v3 — with gating, configurable heads and kernels."""
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


class CausalPoolMixerV2(nn.Module):
    """Causal multi-scale pooling with learned scale importance."""
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

        # Learned scale importance gating
        gates = F.softmax(self.scale_gate(x), dim=-1)  # (B, L, n_scales)
        stacked = torch.stack(pooled_outs, dim=-1)  # (B, L, D, n_scales)
        result = (stacked * gates.unsqueeze(2)).sum(-1)
        return self.out(result * torch.sigmoid(self.gate(x)))


# ═══════════════════════════════════════════════════════
# Shared block factories
# ═══════════════════════════════════════════════════════

class MHConvPoolBlockV2(nn.Module):
    """MultiHeadConv + CausalPool alternating (every 3rd = pool)."""
    def __init__(self, d, d_ff, layer_idx, n_heads=8, max_kernel=65, scales=(2,4,8,16), rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        if layer_idx % 3 == 2:
            self.mix = CausalPoolMixerV2(d, scales)
        else:
            self.mix = MultiHeadConvMixerV3(d, n_heads, max_kernel)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


def _make_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)


# ═══════════════════════════════════════════════════════
# 1. MH CONV POOL DEEP — 20L, narrower FFN
# ═══════════════════════════════════════════════════════

@register_arch("MHConvPoolDeepLM", "novel", "Deep MHConvPool: 20L with narrower FFN")
class MHConvPoolDeepLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            MHConvPoolBlockV2(d, config.d_ff, i,
                              ac.get("n_heads", 8), ac.get("max_kernel", 65),
                              tuple(ac.get("pool_scales", [2,4,8,16])),
                              ac.get("residual_scale", 0.5), ac.get("use_bias", True))
            for i in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_make_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"MHConvPoolDeep: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 2. MH CONV POOL WIDE — 16 heads, wider kernels
# ═══════════════════════════════════════════════════════

@register_arch("MHConvPoolWideLM", "novel", "Wide MHConvPool: 16 heads, wider kernels")
class MHConvPoolWideLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            MHConvPoolBlockV2(d, config.d_ff, i,
                              ac.get("n_heads", 16), ac.get("max_kernel", 129),
                              tuple(ac.get("pool_scales", [2,4,8,16,32])),
                              ac.get("residual_scale", 1.0), ac.get("use_bias", True))
            for i in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_make_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"MHConvPoolWide: {self.config.n_layers}L, 16 heads"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 3. STRIPED MULTI-HEAD CONV — StripedConv + MH structure
# ═══════════════════════════════════════════════════════

class StripedMHConvMixer(nn.Module):
    """
    Each head gets a group of channels (like StripedConv) but also has
    a value projection (like attention). Combines the parallelism of
    StripedConv with the head structure of attention.
    """
    def __init__(self, d_model, n_heads=8, kernel_sizes=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7, 11, 15, 21, 31, 63][:n_heads]

        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)  # query-like gating

        # Per-head convolutions
        self.head_convs = nn.ModuleList([
            nn.Conv1d(self.d_head, self.d_head, ks, padding=ks-1, groups=self.d_head)
            for ks in kernel_sizes
        ])
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head)
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head)

        head_outs = []
        for h in range(self.n_heads):
            vh = v[:, :, h]
            qh = q[:, :, h]
            conv_out = self.head_convs[h](vh.transpose(1, 2))[:, :, :L].transpose(1, 2)
            # Query-gated convolution output
            head_outs.append(F.silu(conv_out) * torch.sigmoid(qh))

        combined = torch.cat(head_outs, dim=-1)
        return self.out(combined)


class StripedMHConvBlock(nn.Module):
    def __init__(self, d, d_ff, n_heads=8, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = StripedMHConvMixer(d, n_heads)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("StripedMHConvLM", "novel", "StripedConv channels + MH attention structure + query gating")
class StripedMHConvLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            StripedMHConvBlock(d, config.d_ff, nh, rs, ac.get("use_bias", True))
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_make_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"StripedMHConv: {self.config.n_layers}L, query-gated multi-head conv"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 4. TOKEN SHIFT CONV — RWKV-style token shift + multi-head conv
# ═══════════════════════════════════════════════════════

class TokenShiftConvMixer(nn.Module):
    """
    RWKV-inspired token shifting combined with multi-head conv.
    Part of each channel sees the previous token (shift), rest sees current.
    Then multi-head conv processes the shifted representation.
    """
    def __init__(self, d_model, n_heads=8, shift_ratio=0.5):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.shift_d = int(d_model * shift_ratio)

        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        kernel_sizes = [min(2**(i+1)+1, 65) for i in range(n_heads)]
        self.head_convs = nn.ModuleList([
            nn.Conv1d(self.d_head, self.d_head, ks, padding=ks-1, groups=self.d_head)
            for ks in kernel_sizes
        ])
        self.gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        # Token shift: blend current and previous token
        x_shifted = torch.cat([
            F.pad(x[:, :-1, :self.shift_d], (0, 0, 1, 0)),  # shifted channels
            x[:, :, self.shift_d:]  # unshifted channels
        ], dim=-1)

        v = self.v_proj(x_shifted).view(B, L, self.n_heads, self.d_head)
        head_outs = []
        for h in range(self.n_heads):
            vh = v[:, :, h]
            out = self.head_convs[h](vh.transpose(1, 2))[:, :, :L].transpose(1, 2)
            head_outs.append(F.silu(out))
        combined = torch.cat(head_outs, dim=-1)
        return self.out(combined * torch.sigmoid(self.gate(x)))


class TokenShiftConvBlock(nn.Module):
    def __init__(self, d, d_ff, n_heads=8, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = TokenShiftConvMixer(d, n_heads)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("TokenShiftConvLM", "novel", "RWKV-style token shift + multi-head conv")
class TokenShiftConvLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            TokenShiftConvBlock(d, config.d_ff, nh, rs, ac.get("use_bias", True))
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_make_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"TokenShiftConv: {self.config.n_layers}L, shifted MH conv"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 5. CONV-DIFF-POOL — Three-way hybrid
# ═══════════════════════════════════════════════════════

class ReactionDiffusionMixerLite(nn.Module):
    """Lightweight reaction-diffusion (2 steps, smaller kernels)."""
    def __init__(self, d_model):
        super().__init__()
        self.d_half = d_model // 2
        self.diff_u = nn.Conv1d(self.d_half, self.d_half, 7, padding=6, groups=self.d_half)
        self.diff_v = nn.Conv1d(self.d_half, self.d_half, 13, padding=12, groups=self.d_half)
        self.react_u = nn.Linear(d_model, self.d_half)
        self.react_v = nn.Linear(d_model, self.d_half)
        self.dt = nn.Parameter(torch.full((1,), 0.1))
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        u, v = x[..., :self.d_half], x[..., self.d_half:]
        for _ in range(2):
            du = self.diff_u(u.transpose(1, 2))[:, :, :L].transpose(1, 2) - u
            dv = self.diff_v(v.transpose(1, 2))[:, :, :L].transpose(1, 2) - v
            uv = torch.cat([u, v], dim=-1)
            dt = torch.sigmoid(self.dt)
            u = u + dt * (du + torch.tanh(self.react_u(uv)))
            v = v + dt * (dv + torch.tanh(self.react_v(uv)))
        result = torch.cat([u, v], dim=-1)
        return self.out(result * torch.sigmoid(self.gate(x)))


class ConvDiffPoolBlock(nn.Module):
    """Three-way hybrid: layers 0,1 = MHConv, layer 2 = reaction-diffusion, layer 3 = pool, repeat."""
    def __init__(self, d, d_ff, layer_idx, n_heads=8, scales=(2,4,8,16), rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        mod = layer_idx % 4
        if mod <= 1:
            self.mix = MultiHeadConvMixerV3(d, n_heads)
        elif mod == 2:
            self.mix = ReactionDiffusionMixerLite(d)
        else:
            self.mix = CausalPoolMixerV2(d, scales)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("ConvDiffPoolLM", "novel", "Three-way hybrid: MHConv + reaction-diffusion + pool")
class ConvDiffPoolLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        ps = tuple(ac.get("pool_scales", [2,4,8,16]))
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            ConvDiffPoolBlock(d, config.d_ff, i, nh, ps, rs, ac.get("use_bias", True))
            for i in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_make_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"ConvDiffPool: {self.config.n_layers}L, three-way hybrid"
    def sequence_mixing_complexity(self): return "O(n)"
