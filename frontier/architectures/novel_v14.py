"""
Novel Architectures — Batch 14 (GPU-Friendly Innovation)
=========================================================
Batch 13 lesson: avoid Python loops over seq positions, avoid large intermediates.
Stick to operations that are GPU-friendly: matmuls, conv1d, element-wise ops, cumsum.

Best so far: AdaptiveWindow (3.5837) — conv 50% + growing attention windows.
Target: break below 3.55 at 88M params.

Strategy — 5 architectures that combine proven pieces in new ways:
1. AdaptiveDeepLM: 20L version of AdaptiveWindow (the winner) with d_ff=1536
2. ConvAttnResidualLM: Conv and attention in parallel at each layer, with separate
   residual streams that cross-communicate every N layers
3. ProgressiveGQALM: Progressive with grouped-query attention (4 KV heads, 8 Q heads)
   — more efficient attention in late layers allows wider windows
4. ConvAttnMoEAdaptiveLM: AdaptiveWindow + MoE FFN — combining two improvements
5. TripleWideLM: 12 layers, d_model=640, d_ff=2560 — wider model instead of deeper
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


class WindowedCausalAttention(nn.Module):
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


class FlexBlock(nn.Module):
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


class SparseMoEFFN(nn.Module):
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
# 1. ADAPTIVE DEEP — 20L AdaptiveWindow
# ═══════════════════════════════════════════════════════

class AdaptiveWindowBlock(nn.Module):
    def __init__(self, d, d_ff, layer_idx, total_layers, n_heads=8, rs=1.0):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff)
        self.rs = rs
        conv_layers = int(total_layers * 0.5)
        if layer_idx < conv_layers:
            self.mix = MultiHeadConvMixer(d, n_heads)
        else:
            attn_idx = layer_idx - conv_layers
            window = 32 * (2 ** min(attn_idx, 4))
            window = min(window, 512)
            self.mix = WindowedCausalAttention(d, n_heads, window=window)

    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("AdaptiveDeepLM", "novel", "20L AdaptiveWindow with narrower FFN")
class AdaptiveDeepLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            AdaptiveWindowBlock(d, config.d_ff, i, n, nh, rs)
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
    def describe(self): return f"AdaptiveDeep: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 2. CONV-ATTN DUAL RESIDUAL — Separate residual streams
# ═══════════════════════════════════════════════════════

class DualResidualBlock(nn.Module):
    """
    Two residual streams: one processed by conv, one by attention.
    Every 4 layers, cross-communicate via linear projection.
    """
    def __init__(self, d, d_ff, layer_idx, n_heads=8, window=128, rs=1.0, cross_every=4):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.n2 = nn.RMSNorm(d)
        self.rs = rs
        self.layer_idx = layer_idx
        self.do_cross = (layer_idx % cross_every == cross_every - 1)

        # Conv for this layer
        self.conv = MultiHeadConvMixer(d, n_heads)
        # Attention for this layer
        self.attn = WindowedCausalAttention(d, n_heads, window)
        # Shared FFN
        self.ffn = SwiGLU(d, d_ff)

        if self.do_cross:
            self.cross_conv_to_attn = nn.Linear(d, d, bias=False)
            self.cross_attn_to_conv = nn.Linear(d, d, bias=False)
            self.cross_gate = nn.Linear(d * 2, d)

    def forward(self, x, attn_stream=None):
        if attn_stream is None:
            attn_stream = x.clone()

        # Process each stream
        conv_out = x + self.rs * self.conv(self.n1(x))
        attn_out = attn_stream + self.rs * self.attn(self.n1(attn_stream))

        # Cross-communicate
        if self.do_cross:
            c2a = self.cross_conv_to_attn(conv_out)
            a2c = self.cross_attn_to_conv(attn_out)
            gate = torch.sigmoid(self.cross_gate(torch.cat([conv_out, attn_out], dim=-1)))
            conv_out = conv_out + 0.1 * a2c * gate
            attn_out = attn_out + 0.1 * c2a * (1 - gate)

        # Shared FFN on combined
        combined = conv_out + attn_out
        combined = combined + self.rs * self.ffn(self.n2(combined))
        return combined, attn_out


@register_arch("ConvAttnDualResLM", "novel", "Dual residual streams (conv+attn) with cross-talk")
class ConvAttnDualResLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            DualResidualBlock(d, config.d_ff, i, nh, window=128, rs=rs)
            for i in range(n)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_init_weights)

    def forward(self, x):
        h = self.embed(x)
        attn_h = None
        for b in self.blocks:
            h, attn_h = b(h, attn_h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"ConvAttnDualRes: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 3. PROGRESSIVE GQA — Grouped-query attention for efficiency
# ═══════════════════════════════════════════════════════

class WindowedCausalGQA(nn.Module):
    """Grouped-Query Attention: fewer KV heads, more Q heads. Saves memory."""
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

        # Repeat KV heads
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


@register_arch("ProgressiveGQALM", "novel", "Conv early + GQA with wide windows late")
class ProgressiveGQALM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        split = int(n * 0.5)  # 50/50 since GQA is cheaper

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                # GQA with wide windows — cheap enough to afford w=256
                mixer = WindowedCausalGQA(d, nh, n_kv_heads=4, window=256)
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
    def describe(self): return f"ProgressiveGQA: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 4. ADAPTIVE WINDOW + MOE — Best mixer + best FFN
# ═══════════════════════════════════════════════════════

@register_arch("AdaptiveMoELM", "novel", "AdaptiveWindow mixer + MoE FFN")
class AdaptiveMoELM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        conv_layers = int(n * 0.5)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i < conv_layers:
                mixer = MultiHeadConvMixer(d, nh)
            else:
                attn_idx = i - conv_layers
                window = 32 * (2 ** min(attn_idx, 4))
                window = min(window, 512)
                mixer = WindowedCausalAttention(d, nh, window=window)
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
    def describe(self): return f"AdaptiveMoE: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 5. WIDE PROGRESSIVE — 12L x d=640
# ═══════════════════════════════════════════════════════

@register_arch("WideProgressiveLM", "novel", "12L x 640d progressive conv→attn")
class WideProgressiveLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model  # 640
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers  # 12
        split = int(n * 0.5)  # 6 conv, 6 attn

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
    def describe(self): return f"WideProg: {self.config.n_layers}L x {self.config.d_model}d"
    def sequence_mixing_complexity(self): return "O(n*w)"
