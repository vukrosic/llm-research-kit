"""
V6: Minimal Attention Hybrids
==============================
V5 proved: no O(n) mechanism beats GatedMHConv (3.915).
The 0.131 gap to transformer (3.784) requires content-content comparison.

V6 asks: how LITTLE attention do you need to close this gap?
We test various minimal-attention hybrids:
- 1 attn layer among 21 conv layers
- 2 attn layers among 20 conv layers
- Chunked-local attention (sub-quadratic)
- Low-rank attention (d_head=16)
- Single-head attention
- Linear attention variants

Goal: find the CHEAPEST way to get attention's benefit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from frontier.architectures.base import FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
from frontier.architectures.registry import register_arch
from frontier.architectures.batch100 import SwiGLU, Block, _init


# ═══════════════════════════════════════════════════════════════════════
# Shared: GatedMHConv mixer (the V2_04 winner)
# ═══════════════════════════════════════════════════════════════════════
class GatedMHConvMixer(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks
        ])
        self.head_gates = nn.Linear(d, nh)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).view(B, L, self.nh, self.dh)
        gates = torch.sigmoid(self.head_gates(x))
        hs = []
        for h in range(self.nh):
            hv = v[:, :, h, :].transpose(1, 2)
            conv_out = F.silu(self.convs[h](hv)[:, :, :L].transpose(1, 2))
            hs.append(conv_out * gates[:, :, h:h+1])
        return self.out(torch.cat(hs, -1))


# ═══════════════════════════════════════════════════════════════════════
# Shared: Causal self-attention mixer (standard, cheap)
# ═══════════════════════════════════════════════════════════════════════
class CausalSelfAttnMixer(nn.Module):
    """Standard causal self-attention with QK-norm."""
    def __init__(self, d, nh=8):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.q_norm = nn.RMSNorm(self.dh)
        self.k_norm = nn.RMSNorm(self.dh)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.nh, self.dh)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        q = self.q_norm(q); k = self.k_norm(k)
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        # Efficient causal attention via SDPA
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1,2).contiguous().view(B, L, D))


# ═══════════════════════════════════════════════════════════════════════
# Shared: Low-rank attention (d_head=16, cheaper O(n²))
# ═══════════════════════════════════════════════════════════════════════
class LowRankAttnMixer(nn.Module):
    """Attention with very small d_head (16 instead of 64). 4x cheaper."""
    def __init__(self, d, n_heads=8, d_head=16):
        super().__init__()
        self.nh = n_heads; self.dh = d_head
        self.q_proj = nn.Linear(d, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)  # full-rank values
        # Value needs to be reshaped to match n_heads
        self.v_dh = d // n_heads
        self.q_norm = nn.RMSNorm(d_head)
        self.k_norm = nn.RMSNorm(d_head)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.nh, self.dh)
        k = self.k_proj(x).view(B, L, self.nh, self.dh)
        v = self.v_proj(x).view(B, L, self.nh, self.v_dh)
        q = self.q_norm(q); k = self.k_norm(k)
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        # Standard causal attention
        scale = self.dh ** -0.5
        attn = torch.matmul(q, k.transpose(-1,-2)) * scale
        causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.out(out.transpose(1,2).contiguous().view(B, L, D))


# ═══════════════════════════════════════════════════════════════════════
# Shared: Chunked-local attention (only within windows)
# ═══════════════════════════════════════════════════════════════════════
class ChunkedLocalAttnMixer(nn.Module):
    """Attention only within local windows of size W. O(n*W) cost."""
    def __init__(self, d, nh=8, window=128):
        super().__init__()
        self.nh = nh; self.dh = d // nh; self.w = window
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.q_norm = nn.RMSNorm(self.dh)
        self.k_norm = nn.RMSNorm(self.dh)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.nh, self.dh)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        q = self.q_norm(q); k = self.k_norm(k)
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)  # (B, nh, L, dh)

        W = self.w
        # Pad to multiple of window
        pad_len = (W - L % W) % W
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        Lp = q.shape[2]
        n_windows = Lp // W

        # Reshape into windows
        q = q.view(B, self.nh, n_windows, W, self.dh)
        k = k.view(B, self.nh, n_windows, W, self.dh)
        v = v.view(B, self.nh, n_windows, W, self.dh)

        # Attention within each window (causal)
        out = F.scaled_dot_product_attention(
            q.reshape(B * self.nh * n_windows, W, self.dh),
            k.reshape(B * self.nh * n_windows, W, self.dh),
            v.reshape(B * self.nh * n_windows, W, self.dh),
            is_causal=True
        )
        out = out.view(B, self.nh, Lp, self.dh)[:, :, :L]
        return self.out(out.transpose(1,2).contiguous().view(B, L, D))


# ═══════════════════════════════════════════════════════════════════════
# Shared: Single-head attention (cheapest possible attention)
# ═══════════════════════════════════════════════════════════════════════
class SingleHeadAttnMixer(nn.Module):
    """Single attention head. Minimal compute for content-content interaction."""
    def __init__(self, d, d_head=64):
        super().__init__()
        self.dh = d_head
        self.q = nn.Linear(d, d_head, bias=False)
        self.k = nn.Linear(d, d_head, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.q_norm = nn.RMSNorm(d_head)
        self.k_norm = nn.RMSNorm(d_head)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        q = self.q_norm(self.q(x)).unsqueeze(1)  # (B, 1, L, dh)
        k = self.k_norm(self.k(x)).unsqueeze(1)
        v = self.v(x).unsqueeze(1)  # (B, 1, L, D)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.squeeze(1))


# ═══════════════════════════════════════════════════════════════════════
# Shared: Chunked linear attention (ELU+1 feature map, O(n))
# ═══════════════════════════════════════════════════════════════════════
class ChunkedLinAttnMixer(nn.Module):
    """O(n) linear attention via chunked processing."""
    def __init__(self, d, nh=8, chunk=128):
        super().__init__()
        self.nh = nh; self.dh = d // nh; self.chunk = chunk
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.q_norm = nn.RMSNorm(self.dh)
        self.k_norm = nn.RMSNorm(self.dh)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.nh, self.dh)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        q = F.elu(self.q_norm(q)) + 1.0
        k = F.elu(self.k_norm(k)) + 1.0
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)

        C = self.chunk
        n_chunks = (L + C - 1) // C
        if L % C != 0:
            p = n_chunks * C - L
            q = F.pad(q, (0,0,0,p)); k = F.pad(k, (0,0,0,p)); v = F.pad(v, (0,0,0,p))

        Lp = n_chunks * C
        q = q.view(B, self.nh, n_chunks, C, self.dh)
        k = k.view(B, self.nh, n_chunks, C, self.dh)
        v = v.view(B, self.nh, n_chunks, C, self.dh)

        state = torch.zeros(B, self.nh, self.dh, self.dh, device=x.device, dtype=x.dtype)
        outputs = []
        for c in range(n_chunks):
            qc, kc, vc = q[:,:,c], k[:,:,c], v[:,:,c]
            # Intra-chunk causal
            intra = torch.matmul(qc, kc.transpose(-1,-2))
            intra = intra * torch.tril(torch.ones(C, C, device=x.device, dtype=x.dtype))
            denom = intra.sum(-1, keepdim=True).clamp(min=1e-6)
            intra_out = torch.matmul(intra, vc) / denom
            # Inter-chunk via state
            inter_out = torch.matmul(qc, state)
            outputs.append(intra_out + inter_out * 0.1)
            state = state + torch.matmul(kc.transpose(-1,-2), vc)

        out = torch.cat(outputs, dim=2)[:,:,:L]
        return self.out(out.transpose(1,2).contiguous().view(B, L, D))


# ═══════════════════════════════════════════════════════════════════════
# V6 Architectures: Minimal Attention Hybrids
# ═══════════════════════════════════════════════════════════════════════

# V6_01: 21 conv + 1 attn (at the end)
@register_arch("V6_01_Conv21Attn1LM", "novel_v6", "21 GatedMHConv + 1 standard attention at end")
class V6_01(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [Block(d, dff, GatedMHConvMixer(d)) for _ in range(21)]
        blocks.append(Block(d, dff, CausalSelfAttnMixer(d)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v6"
    def describe(self): return "V6_01: 21 GatedMHConv + 1 standard attention at end"

# V6_02: 20 conv + 2 attn (last 2)
@register_arch("V6_02_Conv20Attn2LM", "novel_v6", "20 GatedMHConv + 2 standard attention at end")
class V6_02(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [Block(d, dff, GatedMHConvMixer(d)) for _ in range(20)]
        blocks += [Block(d, dff, CausalSelfAttnMixer(d)) for _ in range(2)]
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v6"
    def describe(self): return "V6_02: 20 GatedMHConv + 2 standard attention at end"

# V6_03: 18 conv + 4 attn (last 4)
@register_arch("V6_03_Conv18Attn4LM", "novel_v6", "18 GatedMHConv + 4 standard attention at end")
class V6_03(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [Block(d, dff, GatedMHConvMixer(d)) for _ in range(18)]
        blocks += [Block(d, dff, CausalSelfAttnMixer(d)) for _ in range(4)]
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v6"
    def describe(self): return "V6_03: 18 GatedMHConv + 4 standard attention at end"

# V6_04: Conv + attn every 5th layer (interleaved)
@register_arch("V6_04_ConvAttnInterleavedLM", "novel_v6", "Conv + attention every 5th layer (4 attn among 18 conv)")
class V6_04(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        blocks = []
        for i in range(22):
            if (i + 1) % 5 == 0:  # layers 4, 9, 14, 19 get attention
                blocks.append(Block(d, dff, CausalSelfAttnMixer(d)))
            else:
                blocks.append(Block(d, dff, GatedMHConvMixer(d)))
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v6"
    def describe(self): return "V6_04: Conv + attention every 5th layer"

# V6_05: 21 conv + 1 low-rank attn (d_head=16)
@register_arch("V6_05_ConvLowRankAttnLM", "novel_v6", "21 GatedMHConv + 1 low-rank attention (d_head=16)")
class V6_05(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [Block(d, dff, GatedMHConvMixer(d)) for _ in range(21)]
        blocks.append(Block(d, dff, LowRankAttnMixer(d, d_head=16)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v6"
    def describe(self): return "V6_05: 21 GatedMHConv + 1 low-rank attn (d_head=16)"

# V6_06: 21 conv + 1 single-head attn
@register_arch("V6_06_ConvSingleHeadLM", "novel_v6", "21 GatedMHConv + 1 single-head attention")
class V6_06(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [Block(d, dff, GatedMHConvMixer(d)) for _ in range(21)]
        blocks.append(Block(d, dff, SingleHeadAttnMixer(d)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v6"
    def describe(self): return "V6_06: 21 GatedMHConv + 1 single-head attention"

# V6_07: 20 conv + 2 chunked-local attn (window=128)
@register_arch("V6_07_ConvChunkedAttnLM", "novel_v6", "20 GatedMHConv + 2 chunked-local attn (W=128)")
class V6_07(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [Block(d, dff, GatedMHConvMixer(d)) for _ in range(20)]
        blocks += [Block(d, dff, ChunkedLocalAttnMixer(d, window=128)) for _ in range(2)]
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v6"
    def describe(self): return "V6_07: 20 GatedMHConv + 2 chunked-local attn (W=128)"

# V6_08: 20 conv + 2 linear attn (ELU+1, O(n))
@register_arch("V6_08_ConvLinearAttnLM", "novel_v6", "20 GatedMHConv + 2 chunked linear attention")
class V6_08(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [Block(d, dff, GatedMHConvMixer(d)) for _ in range(20)]
        blocks += [Block(d, dff, ChunkedLinAttnMixer(d)) for _ in range(2)]
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v6"
    def describe(self): return "V6_08: 20 GatedMHConv + 2 chunked linear attention"

# V6_09: Pure GatedMHConv baseline (for fair comparison)
@register_arch("V6_09_PureGatedMHConvLM", "novel_v6", "22 GatedMHConv (baseline for V6 comparison)")
class V6_09(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        self.blocks = nn.ModuleList([Block(d, dff, GatedMHConvMixer(d)) for _ in range(22)])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v6"
    def describe(self): return "V6_09: Pure GatedMHConv baseline (22L)"

# V6_10: Pure transformer baseline (for fair comparison)
@register_arch("V6_10_PureTransformerLM", "novel_v6", "22 standard attention (transformer baseline)")
class V6_10(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        self.blocks = nn.ModuleList([Block(d, dff, CausalSelfAttnMixer(d)) for _ in range(22)])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v6"
    def describe(self): return "V6_10: Pure transformer baseline (22L)"

# V6_11: Conv with PolyConv + 1 attn (combine best V5 + minimal attention)
@register_arch("V6_11_PolyConvAttn1LM", "novel_v6", "21 PolyConv + 1 attention at end")
class V6_11(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        from frontier.architectures.novel_noattn_v5 import PolynomialConv
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [Block(d, dff, PolynomialConv(d)) for _ in range(21)]
        blocks.append(Block(d, dff, CausalSelfAttnMixer(d)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v6"
    def describe(self): return "V6_11: 21 PolyConv + 1 attention at end"

# V6_12: 16 conv + 6 attn (progressive, like proven winner at 12M tokens)
@register_arch("V6_12_Conv16Attn6LM", "novel_v6", "16 GatedMHConv + 6 attention (progressive)")
class V6_12(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [Block(d, dff, GatedMHConvMixer(d)) for _ in range(16)]
        blocks += [Block(d, dff, CausalSelfAttnMixer(d)) for _ in range(6)]
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v6"
    def describe(self): return "V6_12: 16 GatedMHConv + 6 attention (progressive)"
