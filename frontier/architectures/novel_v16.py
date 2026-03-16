"""
Novel Architectures — Batch 16 (Breaking 3.55)
================================================
WideGQA at 3.5524. Need to break below 3.55.

Novel ideas + one exploitation:
1. WideGQAAdaptiveLM: Best mixer (WideGQA) + growing windows 64→512
2. ConvGQAResidualGateLM: Conv + GQA with a learned residual gate per layer
   (scale residual contribution based on layer depth — deep layers get smaller residuals)
3. TokenMixConvLM: Before conv, do a lightweight token mixing via shift+linear
   (inspired by RWKV token shift but applied to conv architecture)
4. ConvGQANormFreeLM: Remove RMSNorm from mixer sub-blocks, keep only before FFN
   (hypothesis: normalization between conv and residual hurts)
5. WideGQALargerLM: 14L x 640d — 2 more layers than current best (risk: OOM)
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


class MultiHeadConvMixer(nn.Module):
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


class WindowedCausalGQA(nn.Module):
    def __init__(self, d_model, n_heads=8, n_kv_heads=4, window=256):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads
        self.heads_per_kv = n_heads // n_kv_heads
        self.window = window
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.scale = self.d_head ** -0.5

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.d_head).transpose(1, 2)
        k = k.repeat_interleave(self.heads_per_kv, dim=1)
        v = v.repeat_interleave(self.heads_per_kv, dim=1)
        W = self.window
        output = torch.zeros_like(q)
        for i in range(0, L, W):
            end = min(i + W, L)
            start_k = max(0, i - W)
            qi = q[:, :, i:end]
            ki = k[:, :, start_k:end]
            vi = v[:, :, start_k:end]
            attn = torch.matmul(qi, ki.transpose(-1, -2)) * self.scale
            q_pos = torch.arange(i, end, device=x.device)
            k_pos = torch.arange(start_k, end, device=x.device)
            mask = q_pos.unsqueeze(-1) < k_pos.unsqueeze(0)
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = F.softmax(attn, dim=-1)
            output[:, :, i:end] = torch.matmul(attn, vi)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(output)


class FlexBlock(nn.Module):
    def __init__(self, d, d_ff, mixer, rs=1.0):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = mixer
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias=True)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


# ═══════════════════════════════════════════════════════
# 1. WIDE GQA ADAPTIVE — Growing windows on wide model
# ═══════════════════════════════════════════════════════

@register_arch("WideGQAAdaptiveLM", "novel", "12L x 640d + GQA with growing windows 64→512")
class WideGQAAdaptiveLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        n_kv = ac.get("n_kv_heads", 4)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        split = int(n * 0.5)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                attn_idx = i - split
                window = 64 * (2 ** min(attn_idx, 3))  # 64, 128, 256, 512
                window = min(window, 512)
                mixer = WindowedCausalGQA(d, nh, n_kv, window=window)
            else:
                mixer = MultiHeadConvMixer(d, nh)
            blocks.append(FlexBlock(d, config.d_ff, mixer, rs=rs))
        self.blocks = nn.ModuleList(blocks)
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
    def describe(self): return f"WideGQAAdaptive: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 2. CONV GQA WITH RESIDUAL GATE — Depth-scaled residuals
# ═══════════════════════════════════════════════════════

class DepthGatedBlock(nn.Module):
    """Block where residual scale decreases with depth (later layers contribute less)."""
    def __init__(self, d, d_ff, mixer, layer_idx, total_layers, rs=1.0):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = mixer
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias=True)
        # Depth-scaled residual: alpha = 1/sqrt(2*layer_idx+1)
        self.alpha = rs / math.sqrt(2 * layer_idx + 1)

    def forward(self, x):
        x = x + self.alpha * self.mix(self.n1(x))
        x = x + self.alpha * self.ffn(self.n2(x))
        return x


@register_arch("ConvGQADepthGateLM", "novel", "12L x 640d with depth-scaled residuals")
class ConvGQADepthGateLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        n_kv = ac.get("n_kv_heads", 4)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        split = int(n * 0.5)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                mixer = WindowedCausalGQA(d, nh, n_kv, window=256)
            else:
                mixer = MultiHeadConvMixer(d, nh)
            blocks.append(DepthGatedBlock(d, config.d_ff, mixer, i, n, rs))
        self.blocks = nn.ModuleList(blocks)
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
    def describe(self): return f"ConvGQADepthGate: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 3. TOKEN SHIFT CONV — RWKV-style token shift before conv
# ═══════════════════════════════════════════════════════

class TokenShiftConvMixer(nn.Module):
    """
    Before the multi-head conv, mix current token with previous token
    via a learned interpolation (inspired by RWKV time-mixing).
    This gives the conv a "warm start" with adjacent token info.
    """
    def __init__(self, d_model, n_heads=8, max_kernel=65):
        super().__init__()
        self.mix_weight = nn.Parameter(torch.ones(d_model) * 0.5)
        self.conv = MultiHeadConvMixer(d_model, n_heads, max_kernel)

    def forward(self, x):
        # Causal token shift: mix with previous token
        w = torch.sigmoid(self.mix_weight)
        shifted = F.pad(x[:, :-1], (0, 0, 1, 0))  # shift right by 1 (causal)
        mixed = w * x + (1 - w) * shifted
        return self.conv(mixed)


@register_arch("TokenShiftGQALM", "novel", "12L x 640d with token-shift conv + GQA")
class TokenShiftGQALM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        n_kv = ac.get("n_kv_heads", 4)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        split = int(n * 0.5)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                mixer = WindowedCausalGQA(d, nh, n_kv, window=256)
            else:
                mixer = TokenShiftConvMixer(d, nh)
            blocks.append(FlexBlock(d, config.d_ff, mixer, rs=rs))
        self.blocks = nn.ModuleList(blocks)
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
    def describe(self): return f"TokenShiftGQA: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 4. CONV GQA NORM-FREE MIXER — No norm before conv mixer
# ═══════════════════════════════════════════════════════

class NormFreeBlock(nn.Module):
    """Skip the RMSNorm before the mixer (only norm before FFN)."""
    def __init__(self, d, d_ff, mixer, rs=1.0):
        super().__init__()
        self.mix = mixer
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias=True)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(x)  # no norm before mixer
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("ConvGQANormFreeLM", "novel", "12L x 640d, no norm before mixer")
class ConvGQANormFreeLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        n_kv = ac.get("n_kv_heads", 4)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        split = int(n * 0.5)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                mixer = WindowedCausalGQA(d, nh, n_kv, window=256)
            else:
                mixer = MultiHeadConvMixer(d, nh)
            blocks.append(NormFreeBlock(d, config.d_ff, mixer, rs=rs))
        self.blocks = nn.ModuleList(blocks)
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
    def describe(self): return f"ConvGQANormFree: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 5. WIDE GQA LARGER — 14L x 640d (push layer count)
# ═══════════════════════════════════════════════════════

@register_arch("WideGQALargerLM", "novel", "14L x 640d + GQA (more layers)")
class WideGQALargerLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        n_kv = ac.get("n_kv_heads", 4)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers  # 14
        split = int(n * 0.5)  # 7 conv, 7 GQA

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                mixer = WindowedCausalGQA(d, nh, n_kv, window=256)
            else:
                mixer = MultiHeadConvMixer(d, nh)
            blocks.append(FlexBlock(d, config.d_ff, mixer, rs=rs))
        self.blocks = nn.ModuleList(blocks)
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
    def describe(self): return f"WideGQALarger: {self.config.n_layers}L x {self.config.d_model}d"
    def sequence_mixing_complexity(self): return "O(n*w)"
