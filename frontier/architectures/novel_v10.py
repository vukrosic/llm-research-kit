"""
Novel Architectures — Batch 10 (Scale-Aware Design)
=====================================================
Key finding: conv beats transformer at 6M but ties at 12M.
Hypothesis: conv has limited global context → saturates faster.

Strategy: Design architectures that combine conv's efficiency with
mechanisms that get *better* with more data (like attention does).

1. MHConvGlobalGateLM: MHConv where each head's output is gated by a global average
2. MHConvCrossHeadLM: Cross-head attention within each conv layer (cheap global mixing)
3. ProgressiveConvAttnLM: Early layers = conv, late layers = small attention window
4. ConvMoELM: Conv mixer with sparse MoE FFN (more capacity, same compute)
5. MHConvPoolBest12M_v2: Retrain MHConvPool at 12M with higher LR to see if optimization matters
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


class MultiHeadConvMixerB10(nn.Module):
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


# ═══════════════════════════════════════════════════════
# 1. MHCONV GLOBAL GATE — Gate by global average
# ═══════════════════════════════════════════════════════

class MHConvGlobalGateMixer(nn.Module):
    """
    Each head's conv output is modulated by a global (causal) average gate.
    This lets the model adjust per-head importance based on the full context so far.
    """
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
        # Global gate: causal running average → per-head gate
        self.global_gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head)

        # Causal running average for global context
        cumsum = torch.cumsum(x, dim=1)
        denom = torch.arange(1, L+1, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(-1)
        global_ctx = cumsum / denom  # (B, L, D) causal average

        head_outs = []
        for h in range(self.n_heads):
            vh = v[:, :, h]
            out = self.head_convs[h](vh.transpose(1, 2))[:, :, :L].transpose(1, 2)
            head_outs.append(F.silu(out))

        combined = torch.cat(head_outs, dim=-1)
        global_g = torch.sigmoid(self.global_gate(global_ctx))
        return self.out(combined * global_g)


class MHConvGlobalGateBlock(nn.Module):
    def __init__(self, d, d_ff, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = MHConvGlobalGateMixer(d)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("MHConvGlobalGateLM", "novel", "MHConv gated by causal running average")
class MHConvGlobalGateLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            MHConvGlobalGateBlock(d, config.d_ff, rs, ac.get("use_bias", True))
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
    def describe(self): return f"MHConvGlobalGate: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 2. MHCONV CROSS-HEAD — Cross-head mixing after conv
# ═══════════════════════════════════════════════════════

class MHConvCrossHeadMixer(nn.Module):
    """
    After per-head conv, mix information across heads via a small MLP.
    This provides per-position cross-head interaction (like a miniature attention).
    """
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
        # Cross-head mixing MLP (operates on n_heads dim)
        self.cross_head = nn.Sequential(
            nn.Linear(n_heads, n_heads * 2),
            nn.SiLU(),
            nn.Linear(n_heads * 2, n_heads),
        )
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

        # Stack heads: (B, L, d_head, n_heads)
        stacked = torch.stack(head_outs, dim=-1)
        # Cross-head mixing: MLP on last dim
        mixed = self.cross_head(stacked)  # (B, L, d_head, n_heads)
        # Reshape back to (B, L, D)
        combined = mixed.transpose(-1, -2).contiguous().view(B, L, D)
        return self.out(combined * torch.sigmoid(self.gate(x)))


class MHConvCrossHeadBlock(nn.Module):
    def __init__(self, d, d_ff, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = MHConvCrossHeadMixer(d)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("MHConvCrossHeadLM", "novel", "MHConv with cross-head MLP mixing")
class MHConvCrossHeadLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            MHConvCrossHeadBlock(d, config.d_ff, rs, ac.get("use_bias", True))
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
    def describe(self): return f"MHConvCrossHead: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 3. PROGRESSIVE CONV→ATTN — Conv early, windowed attention late
# ═══════════════════════════════════════════════════════

class WindowedCausalAttention(nn.Module):
    """Small-window causal attention (window=128). O(n*w) where w<<n."""
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
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]  # (B, L, H, d_head)
        q = q.transpose(1, 2)  # (B, H, L, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Windowed causal attention
        W = self.window
        output = torch.zeros_like(v)

        for i in range(0, L, W):
            end = min(i + W, L)
            # For this window, attend to all positions <= end (causal)
            start_k = max(0, i - W)  # look back one extra window for context
            qi = q[:, :, i:end]
            ki = k[:, :, start_k:end]
            vi = v[:, :, start_k:end]

            attn = torch.matmul(qi, ki.transpose(-1, -2)) * self.scale
            # Causal mask
            q_pos = torch.arange(i, end, device=x.device)
            k_pos = torch.arange(start_k, end, device=x.device)
            mask = q_pos.unsqueeze(-1) < k_pos.unsqueeze(0)
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = F.softmax(attn, dim=-1)
            output[:, :, i:end] = torch.matmul(attn, vi)

        output = output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(output)


class ProgressiveBlock(nn.Module):
    def __init__(self, d, d_ff, layer_idx, total_layers, n_heads=8, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        # First 75% = conv, last 25% = windowed attention
        if layer_idx >= int(total_layers * 0.75):
            self.mix = WindowedCausalAttention(d, n_heads, window=128)
        else:
            self.mix = MultiHeadConvMixerB10(d, n_heads)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("ProgressiveConvAttnLM", "novel", "Conv in early layers, windowed attention in late layers")
class ProgressiveConvAttnLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            ProgressiveBlock(d, config.d_ff, i, n, nh, rs, ac.get("use_bias", True))
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
    def describe(self): return f"ProgressiveConvAttn: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 4. CONV MOE — Conv mixer + sparse MoE FFN
# ═══════════════════════════════════════════════════════

class SparseMoEFFN(nn.Module):
    """Top-2 sparse MoE with 4 experts (each is a SwiGLU)."""
    def __init__(self, d, d_ff, n_experts=4, top_k=2, bias=True):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d, d_ff, bias) for _ in range(n_experts)])

    def forward(self, x):
        B, L, D = x.shape
        # Route
        logits = self.router(x)  # (B, L, n_experts)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)  # (B, L, top_k)
        weights = F.softmax(weights, dim=-1)

        # Compute expert outputs
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                mask = (indices[:, :, k] == e)  # (B, L)
                if mask.any():
                    expert_input = x[mask]  # (N, D)
                    expert_output = self.experts[e](expert_input)
                    output[mask] += weights[:, :, k][mask].unsqueeze(-1) * expert_output

        return output


class ConvMoEBlock(nn.Module):
    def __init__(self, d, d_ff, n_heads=8, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = MultiHeadConvMixerB10(d, n_heads)
        self.n2 = nn.RMSNorm(d)
        # MoE FFN: 4 experts, each with d_ff//2 width → same total compute as single d_ff
        self.ffn = SparseMoEFFN(d, d_ff // 2, n_experts=4, top_k=2, bias=bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("ConvMoELM", "novel", "MHConv + sparse MoE FFN (4 experts, top-2)")
class ConvMoELM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            ConvMoEBlock(d, config.d_ff, nh, rs, ac.get("use_bias", True))
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
    def describe(self): return f"ConvMoE: {self.config.n_layers}L, sparse MoE FFN"
    def sequence_mixing_complexity(self): return "O(n)"
