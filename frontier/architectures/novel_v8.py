"""
Novel Architectures — Batch 8 (Break the 4.0 Barrier)
======================================================
Conv-based architectures plateau at ~4.04. The issue: pure local mixing
has limited receptive field per layer. Need global interaction without O(n²).

Strategy: Combine the best conv mixer with global information flow:
1. MHConvLinAttnLM: MHConv layers + linear attention every 4th layer (O(n) global)
2. MHConvPoolDeep24LM: Push MHConvPool to 24L with even narrower FFN
3. ConvGatedRetentionLM: Conv + RetNet-style retention (causal, O(n))
4. MHConvCumsumLM: MHConv + learned causal cumulative sums for global aggregation
5. MHConvPool2xLM: MHConvPool but pool every 2nd layer instead of every 3rd
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


class MultiHeadConvMixerB8(nn.Module):
    """MHConv with gating — reused across batch 8 architectures."""
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
# 1. MHCONV + LINEAR ATTENTION — Global context via O(n) attention
# ═══════════════════════════════════════════════════════

class CausalLinearAttention(nn.Module):
    """
    Linear attention with ELU feature map, causal via cumulative sum.
    O(n) complexity. Provides true global context that conv can't.
    """
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.scale = self.d_head ** -0.5

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # ELU feature map (ensures positive features for linear attention)
        q = F.elu(q) + 1.0  # (B, H, L, d_head)
        k = F.elu(k) + 1.0

        # Causal linear attention via cumulative sum
        # kv[t] = sum_{s<=t} k[s]^T @ v[s] — accumulated key-value pairs
        kv = torch.einsum('bhld,bhlm->bhldm', k, v)  # (B, H, L, d_head, d_head)
        kv_cum = torch.cumsum(kv, dim=2)  # causal accumulation
        # Denominator: sum_{s<=t} q[t] @ k[s]
        k_cum = torch.cumsum(k, dim=2)
        denom = torch.einsum('bhld,bhld->bhl', q, k_cum).unsqueeze(-1).clamp(min=1e-6)
        # Numerator: q[t] @ kv_cum[t]
        numer = torch.einsum('bhld,bhldm->bhlm', q, kv_cum)
        attn_out = numer / denom

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(attn_out)


class MHConvLinAttnBlock(nn.Module):
    def __init__(self, d, d_ff, layer_idx, n_heads=8, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        # Every 4th layer = linear attention, rest = MHConv
        if layer_idx % 4 == 3:
            self.mix = CausalLinearAttention(d, n_heads)
        else:
            self.mix = MultiHeadConvMixerB8(d, n_heads)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("MHConvLinAttnLM", "novel", "MHConv + linear attention every 4th layer for global context")
class MHConvLinAttnLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            MHConvLinAttnBlock(d, config.d_ff, i, nh, rs, ac.get("use_bias", True))
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
    def describe(self): return f"MHConvLinAttn: {self.config.n_layers}L, conv + linear attention"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 2. MHCONV POOL DEEP 24L
# ═══════════════════════════════════════════════════════

class CausalPoolMixerB8(nn.Module):
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


class MHConvPoolBlockB8(nn.Module):
    def __init__(self, d, d_ff, layer_idx, n_heads=8, pool_every=3, scales=(2,4,8,16), rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        if layer_idx % pool_every == pool_every - 1:
            self.mix = CausalPoolMixerB8(d, scales)
        else:
            self.mix = MultiHeadConvMixerB8(d, n_heads)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("MHConvPoolDeep24LM", "novel", "MHConvPool at 24 layers with very narrow FFN")
class MHConvPoolDeep24LM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        ps = tuple(ac.get("pool_scales", [2,4,8,16]))
        pe = ac.get("pool_every", 3)
        rs = ac.get("residual_scale", 0.5)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            MHConvPoolBlockB8(d, config.d_ff, i, nh, pe, ps, rs, ac.get("use_bias", True))
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
    def describe(self): return f"MHConvPoolDeep24: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 3. CONV + GATED RETENTION — RetNet-style decay
# ═══════════════════════════════════════════════════════

class GatedRetentionMixer(nn.Module):
    """
    RetNet-style multi-scale retention with exponential decay.
    Uses the parallel form: retention(q, k, v) with decay mask.
    But keeps it O(n) via chunked computation.
    """
    def __init__(self, d_model, n_heads=8, chunk_size=64):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # Per-head decay rates
        decay_init = torch.log(1 - 2**(-5 - torch.arange(n_heads, dtype=torch.float32)))
        self.log_decay = nn.Parameter(decay_init)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Compute decay mask for attention-like computation within chunks
        decay = torch.exp(self.log_decay).clamp(0, 0.999)  # (n_heads,)

        # Process in chunks for memory efficiency
        C = self.chunk_size
        n_chunks = (L + C - 1) // C
        outputs = []

        # Running KV state across chunks
        kv_state = torch.zeros(B, self.n_heads, self.d_head, self.d_head, device=x.device, dtype=x.dtype)

        for c in range(n_chunks):
            start = c * C
            end = min(start + C, L)
            clen = end - start

            qc = q[:, :, start:end]  # (B, H, clen, d_head)
            kc = k[:, :, start:end]
            vc = v[:, :, start:end]

            # Intra-chunk: causal decay mask
            positions = torch.arange(clen, device=x.device, dtype=x.dtype)
            # decay_mask[i,j] = decay^(i-j) for j <= i, else 0
            dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (clen, clen)
            decay_h = decay.view(1, self.n_heads, 1, 1)
            mask = torch.where(dist >= 0, decay_h ** dist.unsqueeze(0).unsqueeze(0), torch.zeros_like(dist.unsqueeze(0).unsqueeze(0).expand_as(decay_h ** dist.abs().unsqueeze(0).unsqueeze(0))))

            # Intra-chunk attention with decay
            attn = torch.matmul(qc, kc.transpose(-1, -2)) * mask * (self.d_head ** -0.5)
            intra = torch.matmul(attn, vc)

            # Cross-chunk: apply state
            cross = torch.matmul(qc, kv_state) * (self.d_head ** -0.5)
            # Decay cross contribution by position within chunk
            pos_decay = (decay_h ** positions.view(1, 1, -1, 1))
            cross = cross * pos_decay

            outputs.append(intra + cross)

            # Update state: decay old state and add new chunk's contribution
            chunk_decay = decay_h.squeeze(-1).squeeze(-1) ** clen  # (1, H)
            kv_state = kv_state * chunk_decay.unsqueeze(-1).unsqueeze(-1)
            # Add new kv pairs with appropriate decay
            for i in range(clen):
                d_factor = decay.view(1, self.n_heads, 1, 1) ** (clen - 1 - i)
                kv_state = kv_state + d_factor * torch.einsum('bhd,bhm->bhdm', kc[:,:,i], vc[:,:,i])

        out = torch.cat(outputs, dim=2).transpose(1, 2).contiguous().view(B, L, D)
        return self.out(out * torch.sigmoid(self.gate(x)))


class ConvRetentionBlock(nn.Module):
    def __init__(self, d, d_ff, layer_idx, n_heads=8, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        if layer_idx % 3 == 2:
            self.mix = GatedRetentionMixer(d, n_heads)
        else:
            self.mix = MultiHeadConvMixerB8(d, n_heads)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("ConvGatedRetentionLM", "novel", "MHConv + RetNet-style gated retention every 3rd layer")
class ConvGatedRetentionLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            ConvRetentionBlock(d, config.d_ff, i, nh, rs, ac.get("use_bias", True))
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
    def describe(self): return f"ConvGatedRetention: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 4. MHCONV + CAUSAL CUMSUM AGGREGATION
# ═══════════════════════════════════════════════════════

class CausalCumsumMixer(nn.Module):
    """
    Learned causal cumulative sums: project input to multiple channels,
    take cumsum along sequence (inherently causal), then project back.
    Simple but provides true global context.
    """
    def __init__(self, d_model, n_channels=4):
        super().__init__()
        self.n_channels = n_channels
        self.in_proj = nn.Linear(d_model, d_model * n_channels, bias=False)
        self.channel_gates = nn.Linear(d_model, n_channels)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        projected = self.in_proj(x).view(B, L, self.n_channels, D)  # (B, L, C, D)

        # Apply different "decay" via learnable gating before cumsum
        channel_sums = []
        for c in range(self.n_channels):
            ch = projected[:, :, c]  # (B, L, D)
            # Causal cumsum with normalization
            cum = torch.cumsum(ch, dim=1)
            # Normalize by position (running average)
            denom = torch.arange(1, L+1, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(-1)
            channel_sums.append(cum / denom)

        stacked = torch.stack(channel_sums, dim=-1)  # (B, L, D, C)
        gates = F.softmax(self.channel_gates(x), dim=-1)  # (B, L, C)
        result = (stacked * gates.unsqueeze(2)).sum(-1)
        return self.out_proj(F.silu(result) * torch.sigmoid(self.gate(x)))


class MHConvCumsumBlock(nn.Module):
    def __init__(self, d, d_ff, layer_idx, n_heads=8, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        if layer_idx % 3 == 2:
            self.mix = CausalCumsumMixer(d)
        else:
            self.mix = MultiHeadConvMixerB8(d, n_heads)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("MHConvCumsumLM", "novel", "MHConv + causal cumsum aggregation every 3rd layer")
class MHConvCumsumLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            MHConvCumsumBlock(d, config.d_ff, i, nh, rs, ac.get("use_bias", True))
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
    def describe(self): return f"MHConvCumsum: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 5. MHCONV POOL 2X — Pool every 2nd layer
# ═══════════════════════════════════════════════════════

@register_arch("MHConvPool2xLM", "novel", "MHConvPool with pool every 2nd layer (more global mixing)")
class MHConvPool2xLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        ps = tuple(ac.get("pool_scales", [2,4,8,16]))
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            MHConvPoolBlockB8(d, config.d_ff, i, nh, 2, ps, rs, ac.get("use_bias", True))
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
    def describe(self): return f"MHConvPool2x: {self.config.n_layers}L, pool every 2nd"
    def sequence_mixing_complexity(self): return "O(n)"
