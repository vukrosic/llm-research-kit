"""
Novel Architectures — Batch 12 (Beyond Progressive Conv→Attn)
=============================================================
Batch 11 established that progressive conv→attn is robust (~3.59).
Now we need genuinely new ideas to push below 3.55.

Strategy:
1. GatedStateConvLM: Conv layers with a per-token running state (like an RNN hidden state
   that accumulates across the sequence, but updated via conv features)
2. ConvAttnFusionLM: Within each layer, run conv AND attention in parallel, then fuse
3. AdaptiveWindowLM: Attention where window size grows with layer depth (16→32→64→128→256)
4. ProgressiveRoPELM: Add rotary position embeddings to the attention layers of progressive
5. DualStreamLM: Two parallel streams (conv stream + attn stream) with cross-stream gating
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


# ═══════════════════════════════════════════════════════
# 1. GATED STATE CONV — Conv with accumulating hidden state
# ═══════════════════════════════════════════════════════

class GatedStateConvMixer(nn.Module):
    """
    Combines MHConv with a running hidden state that evolves across the sequence.
    At each position, the conv output is gated by a state that accumulates
    information via a causal running weighted average (similar to RWKV's time-mixing).
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
        # State update: learned decay rate per channel
        self.decay = nn.Parameter(torch.zeros(d_model))  # sigmoid → (0, 1)
        self.state_gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head)
        head_outs = []
        for h in range(self.n_heads):
            vh = v[:, :, h]
            out = self.head_convs[h](vh.transpose(1, 2))[:, :, :L].transpose(1, 2)
            head_outs.append(F.silu(out))
        conv_out = torch.cat(head_outs, dim=-1)  # (B, L, D)

        # Running state via cumulative weighted average (causal, vectorized)
        # decay ∈ (0,1), state_t = decay * state_{t-1} + (1-decay) * conv_out_t
        # This is equivalent to exponentially-weighted moving average
        alpha = torch.sigmoid(self.decay).unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        # Use log-space cumsum for numerical stability
        # EMA via: output_t = sum_{k=0}^{t} alpha^k * (1-alpha) * input_{t-k}
        # Approximate with cumsum trick: weight each position, then cumsum
        weights = alpha.pow(torch.arange(L, device=x.device, dtype=x.dtype).flip(0).unsqueeze(0).unsqueeze(-1))
        weighted = conv_out * (1 - alpha) * weights
        state = torch.cumsum(weighted, dim=1) / weights.clamp(min=1e-8)

        gate = torch.sigmoid(self.state_gate(x))
        return self.out(conv_out * gate + state * (1 - gate))


# ═══════════════════════════════════════════════════════
# 2. CONV+ATTN FUSION — Parallel conv and attention within each layer
# ═══════════════════════════════════════════════════════

class ConvAttnFusionMixer(nn.Module):
    """
    Run conv and windowed attention IN PARALLEL within a single layer,
    then fuse with a learned gate. This lets each position get both
    local (conv) and semi-global (attention) information simultaneously.
    """
    def __init__(self, d_model, n_heads=8, window=128):
        super().__init__()
        half_d = d_model // 2
        self.half_d = half_d
        # Conv branch: operates on first half of dimensions
        self.conv_branch = MultiHeadConvMixer(half_d, n_heads // 2)
        # Attention branch: operates on second half
        self.attn_branch = WindowedCausalAttention(half_d, n_heads // 2, window)
        # Fusion
        self.fuse_gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        x_conv = x[:, :, :self.half_d]
        x_attn = x[:, :, self.half_d:]

        conv_out = self.conv_branch(x_conv)
        attn_out = self.attn_branch(x_attn)

        combined = torch.cat([conv_out, attn_out], dim=-1)
        gate = torch.sigmoid(self.fuse_gate(x))
        return self.out(combined * gate)


# ═══════════════════════════════════════════════════════
# 3. ADAPTIVE WINDOW — Window size grows with depth
# ═══════════════════════════════════════════════════════

class AdaptiveWindowBlock(nn.Module):
    """Conv early, attention late with growing windows: 32→64→128→256."""
    def __init__(self, d, d_ff, layer_idx, total_layers, n_heads=8, rs=1.0):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff)
        self.rs = rs

        conv_layers = int(total_layers * 0.5)  # first 50% are conv
        if layer_idx < conv_layers:
            self.mix = MultiHeadConvMixer(d, n_heads)
        else:
            # Window grows: 32, 64, 128, 256, 512 across attention layers
            attn_idx = layer_idx - conv_layers
            n_attn = total_layers - conv_layers
            window = 32 * (2 ** min(attn_idx, 4))  # 32, 64, 128, 256, 512
            window = min(window, 512)
            self.mix = WindowedCausalAttention(d, n_heads, window=window)

    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


# ═══════════════════════════════════════════════════════
# 4. PROGRESSIVE + RoPE — Add rotary embeddings to attention layers
# ═══════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    def __init__(self, d_head, max_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, max_len):
        t = torch.arange(max_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x, offset=0):
        # x: (B, H, L, d_head)
        L = x.shape[2]
        cos = self.cos_cached[offset:offset+L].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset:offset+L].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class WindowedCausalAttentionRoPE(nn.Module):
    def __init__(self, d_model, n_heads=8, window=128):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window = window
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.scale = self.d_head ** -0.5
        self.rope = RotaryEmbedding(self.d_head)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)

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


# ═══════════════════════════════════════════════════════
# 5. DUAL STREAM — Parallel conv and attention streams with cross-gating
# ═══════════════════════════════════════════════════════

class DualStreamBlock(nn.Module):
    """
    Two independent processing streams that interact via cross-gating.
    Stream A: conv-based (local features)
    Stream B: attention-based (global features)
    Each stream gates the other, then both contribute to the residual.
    """
    def __init__(self, d, d_ff, n_heads=8, window=128, rs=1.0):
        super().__init__()
        self.n1a = nn.RMSNorm(d)
        self.n1b = nn.RMSNorm(d)
        self.conv_mix = MultiHeadConvMixer(d, n_heads)
        self.attn_mix = WindowedCausalAttention(d, n_heads, window)
        # Cross gates
        self.gate_a = nn.Linear(d, d)  # attn output gates conv stream
        self.gate_b = nn.Linear(d, d)  # conv output gates attn stream
        self.combine = nn.Linear(d * 2, d, bias=False)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff)
        self.rs = rs

    def forward(self, x):
        a = self.conv_mix(self.n1a(x))
        b = self.attn_mix(self.n1b(x))
        # Cross-gate: each stream modulated by the other
        a_gated = a * torch.sigmoid(self.gate_a(b))
        b_gated = b * torch.sigmoid(self.gate_b(a))
        mixed = self.combine(torch.cat([a_gated, b_gated], dim=-1))
        x = x + self.rs * mixed
        x = x + self.rs * self.ffn(self.n2(x))
        return x


# ═══════════════════════════════════════════════════════
# MODEL WRAPPERS
# ═══════════════════════════════════════════════════════

@register_arch("GatedStateConvLM", "novel", "MHConv with running EMA hidden state")
class GatedStateConvLM(FrontierModel):
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
                mixer = GatedStateConvMixer(d, nh)
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
    def describe(self): return f"GatedStateConv: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n)"


@register_arch("ConvAttnFusionLM", "novel", "Parallel conv+attn fusion within each layer")
class ConvAttnFusionLM(FrontierModel):
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
            mixer = ConvAttnFusionMixer(d, nh, window=128)
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
    def describe(self): return f"ConvAttnFusion: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


@register_arch("AdaptiveWindowLM", "novel", "Conv early, attention with growing windows late")
class AdaptiveWindowLM(FrontierModel):
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
    def describe(self): return f"AdaptiveWindow: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


@register_arch("ProgressiveRoPELM", "novel", "Progressive conv→attn with RoPE in attention")
class ProgressiveRoPELM(FrontierModel):
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
                mixer = WindowedCausalAttentionRoPE(d, nh, window=128)
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
    def describe(self): return f"ProgressiveRoPE: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


@register_arch("DualStreamLM", "novel", "Parallel conv+attn streams with cross-gating")
class DualStreamLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            DualStreamBlock(d, config.d_ff, nh, window=128, rs=rs)
            for _ in range(n)
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
    def describe(self): return f"DualStream: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"
