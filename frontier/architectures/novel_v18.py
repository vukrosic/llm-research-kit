"""
Novel Architectures — Batch 18 (Breaking the 3.54 barrier)
==========================================================
Current best: WiderStillGQA 3.5397 (124M), ConvQKNormGQA 3.5424 (107M).

Strategy — 5 genuinely novel mechanisms:
1. DifferentialGQALM: Two attention patterns subtracted to cancel noise
   (differential attention concept applied to our conv→attn framework)
2. DecayMaskGQALM: Exponential decay mask instead of hard window cutoff
   (RetNet-inspired soft attention boundary)
3. ConvTSQKNormLM: Token shift + QK-norm combined (both helped, never tested together)
4. SoftRouterLM: Per-token learned soft routing between conv and attn in EVERY layer
   (not MoE — continuous weighting, genuinely different topology)
5. ValueResidualLM: Pass raw embedding directly into value projections across all
   attention layers (DeepSeek V3 insight: helps gradient flow in deep networks)
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


class TokenShiftConvMixer(nn.Module):
    def __init__(self, d_model, n_heads=8, max_kernel=65):
        super().__init__()
        self.mix_weight = nn.Parameter(torch.ones(d_model) * 0.5)
        self.conv = MultiHeadConvMixer(d_model, n_heads, max_kernel)

    def forward(self, x):
        w = torch.sigmoid(self.mix_weight)
        shifted = F.pad(x[:, :-1], (0, 0, 1, 0))
        mixed = w * x + (1 - w) * shifted
        return self.conv(mixed)


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
# 1. DIFFERENTIAL GQA — Noise-canceling attention
# ═══════════════════════════════════════════════════════

class DifferentialGQA(nn.Module):
    """
    Two parallel GQA computations with shared KV but different Q projections.
    Output = attn1 - lambda * attn2, where lambda is learned per-head.
    The subtraction cancels common-mode noise, sharpening attention.
    """
    def __init__(self, d_model, n_heads=8, n_kv_heads=4, window=256):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads
        self.heads_per_kv = n_heads // n_kv_heads
        self.window = window
        self.scale = self.d_head ** -0.5

        self.q1_proj = nn.Linear(d_model, d_model, bias=False)
        self.q2_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # QK norms (proven to help)
        self.q_norm = nn.RMSNorm(self.d_head)
        self.k_norm = nn.RMSNorm(self.d_head)

        # Learned subtraction weight per head, initialized small
        self.lambda_param = nn.Parameter(torch.ones(n_heads) * 0.5)

    def forward(self, x):
        B, L, D = x.shape
        q1 = self.q1_proj(x).view(B, L, self.n_heads, self.d_head)
        q2 = self.q2_proj(x).view(B, L, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.d_head)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.d_head)

        # QK norm
        q1 = self.q_norm(q1)
        q2 = self.q_norm(q2)  # share the same norm
        k = self.k_norm(k)

        q1 = q1.transpose(1, 2)  # (B, H, L, d)
        q2 = q2.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k = k.repeat_interleave(self.heads_per_kv, dim=1)
        v = v.repeat_interleave(self.heads_per_kv, dim=1)

        lam = torch.sigmoid(self.lambda_param).view(1, self.n_heads, 1, 1)

        W = self.window
        output = torch.zeros_like(q1)
        for i in range(0, L, W):
            end = min(i + W, L)
            start_k = max(0, i - W)
            q1i = q1[:, :, i:end]
            q2i = q2[:, :, i:end]
            ki = k[:, :, start_k:end]
            vi = v[:, :, start_k:end]

            attn1 = torch.matmul(q1i, ki.transpose(-1, -2)) * self.scale
            attn2 = torch.matmul(q2i, ki.transpose(-1, -2)) * self.scale

            q_pos = torch.arange(i, end, device=x.device)
            k_pos = torch.arange(start_k, end, device=x.device)
            mask = q_pos.unsqueeze(-1) < k_pos.unsqueeze(0)
            attn1 = attn1.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn2 = attn2.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn1 = F.softmax(attn1, dim=-1)
            attn2 = F.softmax(attn2, dim=-1)

            # Differential: subtract noise pattern
            diff_attn = attn1 - lam * attn2
            output[:, :, i:end] = torch.matmul(diff_attn, vi)

        output = output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(output)


@register_arch("DifferentialGQALM", "novel", "Conv + differential GQA (noise-canceling attention)")
class DifferentialGQALM(FrontierModel):
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
                mixer = DifferentialGQA(d, nh, n_kv, window=256)
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
    def describe(self): return f"DifferentialGQA: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 2. DECAY MASK GQA — Soft exponential boundary
# ═══════════════════════════════════════════════════════

class DecayMaskGQA(nn.Module):
    """
    Instead of hard window cutoff, use learned per-head exponential decay.
    Each head has a decay rate: attn_weight *= exp(-decay * distance).
    This gives soft boundaries — some heads focus nearby, others far.
    No explicit window needed; full causal attention with decay.
    But we still chunk for memory efficiency.
    """
    def __init__(self, d_model, n_heads=8, n_kv_heads=4, window=512):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads
        self.heads_per_kv = n_heads // n_kv_heads
        self.window = window
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.q_norm = nn.RMSNorm(self.d_head)
        self.k_norm = nn.RMSNorm(self.d_head)

        # Learned decay rates per head (initialized to span multi-scale)
        # Heads see different ranges: some local (~10 tokens), some global (~500)
        init_decays = torch.linspace(-1.0, -5.0, n_heads)  # after sigmoid: ~0.27 to ~0.007
        self.decay_logit = nn.Parameter(init_decays)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.d_head)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.d_head)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        k = k.repeat_interleave(self.heads_per_kv, dim=1)
        v = v.repeat_interleave(self.heads_per_kv, dim=1)

        # Decay rates: sigmoid to (0,1), then scale
        decay = torch.sigmoid(self.decay_logit) * 0.3  # max decay ~0.3 per position
        decay = decay.view(1, self.n_heads, 1, 1)  # (1, H, 1, 1)

        W = self.window
        output = torch.zeros_like(q)
        for i in range(0, L, W):
            end = min(i + W, L)
            start_k = max(0, i - W)
            qi = q[:, :, i:end]
            ki = k[:, :, start_k:end]
            vi = v[:, :, start_k:end]

            attn = torch.matmul(qi, ki.transpose(-1, -2)) * self.scale

            # Causal mask
            q_pos = torch.arange(i, end, device=x.device)
            k_pos = torch.arange(start_k, end, device=x.device)
            causal_mask = q_pos.unsqueeze(-1) < k_pos.unsqueeze(0)

            # Distance-based decay
            distance = (q_pos.unsqueeze(-1) - k_pos.unsqueeze(0)).float().abs()
            decay_mask = torch.exp(-decay * distance.unsqueeze(0).unsqueeze(0))

            attn = attn * decay_mask
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = F.softmax(attn, dim=-1)

            output[:, :, i:end] = torch.matmul(attn, vi)

        output = output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(output)


@register_arch("DecayMaskGQALM", "novel", "Conv + decay-masked GQA (soft exponential boundary)")
class DecayMaskGQALM(FrontierModel):
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
                mixer = DecayMaskGQA(d, nh, n_kv, window=512)
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
    def describe(self): return f"DecayMaskGQA: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 3. CONV TOKEN-SHIFT + QK-NORM — Merge two individual winners
# ═══════════════════════════════════════════════════════

class WindowedCausalGQAWithQKNorm(nn.Module):
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
        self.q_norm = nn.RMSNorm(self.d_head)
        self.k_norm = nn.RMSNorm(self.d_head)
        self.scale = self.d_head ** -0.5

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.d_head)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.d_head)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
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


@register_arch("ConvTSQKNormLM", "novel", "Token-shift conv + QK-normed GQA (merging two winners)")
class ConvTSQKNormLM(FrontierModel):
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
                mixer = WindowedCausalGQAWithQKNorm(d, nh, n_kv, window=256)
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
    def describe(self): return f"ConvTSQKNorm: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 4. SOFT ROUTER — Per-token continuous conv/attn routing
# ═══════════════════════════════════════════════════════

class SoftRouterBlock(nn.Module):
    """
    Every layer has BOTH conv and attention mixers.
    A learned router produces per-token weights in [0,1] to blend them.
    This is NOT MoE (no sparse gating, no top-k). Every token uses both paths
    with learned continuous weighting. Fundamentally different topology from
    progressive: each layer can be conv-heavy OR attn-heavy per-token.
    """
    def __init__(self, d, d_ff, n_heads=8, n_kv_heads=4, window=256, rs=1.0):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.conv_mixer = MultiHeadConvMixer(d, n_heads)
        self.attn_mixer = WindowedCausalGQAWithQKNorm(d, n_heads, n_kv_heads, window)
        self.router = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.SiLU(),
            nn.Linear(d // 4, 1),
        )
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias=True)
        self.rs = rs

    def forward(self, x):
        normed = self.n1(x)
        # Router: per-token weight for attn vs conv
        w = torch.sigmoid(self.router(normed))  # (B, L, 1)
        conv_out = self.conv_mixer(normed)
        attn_out = self.attn_mixer(normed)
        mixed = (1 - w) * conv_out + w * attn_out
        x = x + self.rs * mixed
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("SoftRouterLM", "novel", "Per-token soft routing between conv and attn in every layer")
class SoftRouterLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        n_kv = ac.get("n_kv_heads", 4)
        rs = ac.get("residual_scale", 1.0)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            SoftRouterBlock(d, config.d_ff, nh, n_kv, window=256, rs=rs)
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
    def describe(self): return f"SoftRouter: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 5. VALUE RESIDUAL — Raw embedding → value projections
# ═══════════════════════════════════════════════════════

class ValueResidualGQA(nn.Module):
    """
    Standard GQA with QK-norm, but the value projection receives a weighted
    sum of the current hidden state AND the original embedding.

    V = W_v @ (alpha * hidden + (1-alpha) * embed)

    This provides a direct gradient highway from loss to embedding,
    and gives attention layers access to unprocessed token information.
    Inspired by DeepSeek V3's value residual connections.
    """
    def __init__(self, d_model, n_heads=8, n_kv_heads=4, window=256):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads
        self.heads_per_kv = n_heads // n_kv_heads
        self.window = window
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.d_head)
        self.k_norm = nn.RMSNorm(self.d_head)

        # Learned mixing weight for value residual
        self.v_alpha = nn.Parameter(torch.tensor(0.8))

    def forward(self, x, embed=None):
        B, L, D = x.shape

        # Value gets mixed input
        if embed is not None:
            alpha = torch.sigmoid(self.v_alpha)
            v_input = alpha * x + (1 - alpha) * embed
        else:
            v_input = x

        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.d_head)
        v = self.v_proj(v_input).view(B, L, self.n_kv_heads, self.d_head)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
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


class FlexBlockWithEmbed(nn.Module):
    """FlexBlock that passes embedding to mixer if it supports it."""
    def __init__(self, d, d_ff, mixer, rs=1.0, passes_embed=False):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = mixer
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias=True)
        self.rs = rs
        self.passes_embed = passes_embed

    def forward(self, x, embed=None):
        normed = self.n1(x)
        if self.passes_embed and embed is not None:
            x = x + self.rs * self.mix(normed, embed)
        else:
            x = x + self.rs * self.mix(normed)
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("ValueResidualLM", "novel", "Conv + value-residual GQA (embed→value highway)")
class ValueResidualLM(FrontierModel):
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
                mixer = ValueResidualGQA(d, nh, n_kv, window=256)
                blocks.append(FlexBlockWithEmbed(d, config.d_ff, mixer, rs=rs, passes_embed=True))
            else:
                mixer = MultiHeadConvMixer(d, nh)
                blocks.append(FlexBlockWithEmbed(d, config.d_ff, mixer, rs=rs, passes_embed=False))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_init_weights)

    def forward(self, x):
        embed_out = self.embed(x)
        h = embed_out
        for b in self.blocks:
            h = b(h, embed=embed_out)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"ValueResidual: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"
