"""
Novel Architectures — Batch 11 (ProgressiveConvAttn Exploitation)
================================================================
BREAKTHROUGH: ProgressiveConvAttn (3.5921) beats transformer (3.7060) at 12M tokens.
- Conv early (75%) + windowed attention late (25%), window=128

This batch explores the design space around this breakthrough:
1. ProgressiveHalf: 50/50 conv/attn split (more attention capacity)
2. ProgressiveWide: 75/25 split but window=256 (wider attention context)
3. InterleavedConvAttn: alternating conv/attn every other layer
4. ProgressiveMoE: conv→attn with MoE FFN (combines two winners)
5. ProgressiveDeep: 20 layers (deeper model, same ~88M param budget)
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
    """Per-head depthwise causal conv with exponentially-spaced kernels."""
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


class WindowedCausalAttention(nn.Module):
    """Small-window causal attention. O(n*w) where w<<n."""
    def __init__(self, d_model, n_heads=8, window=128):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window = window
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.scale = self.d_head ** -0.5

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        W = self.window
        output = torch.zeros_like(v)

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


class SparseMoEFFN(nn.Module):
    """Top-2 sparse MoE with 4 experts."""
    def __init__(self, d, d_ff, n_experts=4, top_k=2, bias=True):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d, d_ff, bias) for _ in range(n_experts)])

    def forward(self, x):
        B, L, D = x.shape
        logits = self.router(x)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        output = torch.zeros_like(x)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                mask = (indices[:, :, k] == e)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += weights[:, :, k][mask].unsqueeze(-1) * expert_output
        return output


# ═══════════════════════════════════════════════════════
# Shared block builder
# ═══════════════════════════════════════════════════════

class FlexBlock(nn.Module):
    """Flexible block: any mixer + any FFN."""
    def __init__(self, d, d_ff, mixer, ffn=None, rs=1.0):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = mixer
        self.n2 = nn.RMSNorm(d)
        self.ffn = ffn if ffn is not None else SwiGLU(d, d_ff, bias=True)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


# ═══════════════════════════════════════════════════════
# 1. PROGRESSIVE HALF — 50% conv, 50% attention
# ═══════════════════════════════════════════════════════

@register_arch("ProgressiveHalfLM", "novel", "50/50 conv/attn split")
class ProgressiveHalfLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        split = int(n * 0.5)  # 50% conv, 50% attn

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                mixer = WindowedCausalAttention(d, nh, window=128)
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
    def describe(self): return f"ProgressiveHalf: {self.config.n_layers}L (50/50)"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 2. PROGRESSIVE WIDE — 75/25 split, window=256
# ═══════════════════════════════════════════════════════

@register_arch("ProgressiveWideLM", "novel", "75/25 conv/attn, window=256")
class ProgressiveWideLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        split = int(n * 0.75)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                mixer = WindowedCausalAttention(d, nh, window=256)  # wider window
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
    def describe(self): return f"ProgressiveWide: {self.config.n_layers}L (w=256)"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 3. INTERLEAVED CONV+ATTN — alternating layers
# ═══════════════════════════════════════════════════════

@register_arch("InterleavedConvAttnLM", "novel", "Alternating conv/attn every other layer")
class InterleavedConvAttnLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i % 2 == 1:  # odd layers = attention
                mixer = WindowedCausalAttention(d, nh, window=128)
            else:  # even layers = conv
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
    def describe(self): return f"InterleavedConvAttn: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 4. PROGRESSIVE MOE — conv→attn with MoE FFN
# ═══════════════════════════════════════════════════════

@register_arch("ProgressiveMoELM", "novel", "Conv→Attn progressive with sparse MoE FFN")
class ProgressiveMoELM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        split = int(n * 0.75)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                mixer = WindowedCausalAttention(d, nh, window=128)
            else:
                mixer = MultiHeadConvMixer(d, nh)
            # MoE FFN: 4 experts, each d_ff//2 wide
            ffn = SparseMoEFFN(d, config.d_ff // 2, n_experts=4, top_k=2)
            blocks.append(FlexBlock(d, config.d_ff, mixer, ffn=ffn, rs=rs))
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
    def describe(self): return f"ProgressiveMoE: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 5. PROGRESSIVE DEEP — 20 layers, smaller d_ff
# ═══════════════════════════════════════════════════════

@register_arch("ProgressiveDeepLM", "novel", "20-layer progressive conv→attn, narrower FFN")
class ProgressiveDeepLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers  # 20 layers
        split = int(n * 0.75)  # 15 conv, 5 attn

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                mixer = WindowedCausalAttention(d, nh, window=128)
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
    def describe(self): return f"ProgressiveDeep: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"
