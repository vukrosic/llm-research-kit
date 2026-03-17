"""
V7: Exploiting the SingleHead Breakthrough
============================================
V6_06 (21 GatedMHConv + 1 SingleHeadAttn) = 3.768, beating transformer (3.784).

Key insight: 1 single-head attention (d_qk=64, d_v=512) at the end of a conv stack
gives the model ONE full-width global content-content comparison. This is more
effective than 8 narrow-value heads because after 21 layers of multi-scale conv,
the model needs to integrate ALL features globally, not 8 feature slices.

V7 explores:
- Position of single-head layer(s)
- Number of single-head layers
- Q/K dimension
- Combining with best V4 tweaks
- Width scaling
- Value residual for attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from frontier.architectures.base import FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
from frontier.architectures.registry import register_arch
from frontier.architectures.batch100 import SwiGLU, Block, _init


# ═══════════════════════════════════════════════════════════════════════
# Reuse from V6
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


class GatedMHConvScaledRes(nn.Module):
    """GatedMHConv with per-dimension learned output scaling (best V4 tweak)."""
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
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).view(B, L, self.nh, self.dh)
        gates = torch.sigmoid(self.head_gates(x))
        hs = []
        for h in range(self.nh):
            hv = v[:, :, h, :].transpose(1, 2)
            conv_out = F.silu(self.convs[h](hv)[:, :, :L].transpose(1, 2))
            hs.append(conv_out * gates[:, :, h:h+1])
        return self.out(torch.cat(hs, -1)) * self.scale


class SingleHeadAttnMixer(nn.Module):
    """Single attention head. d_head for Q/K, full d for V."""
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
        q = self.q_norm(self.q(x)).unsqueeze(1)
        k = self.k_norm(self.k(x)).unsqueeze(1)
        v = self.v(x).unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.squeeze(1))


class SingleHeadAttnValueRes(nn.Module):
    """Single-head attention with value residual from initial embedding."""
    def __init__(self, d, d_head=64, alpha=0.5):
        super().__init__()
        self.dh = d_head
        self.q = nn.Linear(d, d_head, bias=False)
        self.k = nn.Linear(d, d_head, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.q_norm = nn.RMSNorm(d_head)
        self.k_norm = nn.RMSNorm(d_head)
        self.out = nn.Linear(d, d, bias=False)
        self.alpha = alpha  # blend factor for value residual

    def forward(self, x, embed=None, **kw):
        B, L, D = x.shape
        q = self.q_norm(self.q(x)).unsqueeze(1)
        k = self.k_norm(self.k(x)).unsqueeze(1)
        v = self.v(x)
        if embed is not None:
            v = (1 - self.alpha) * v + self.alpha * embed
        v = v.unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.squeeze(1))


class CausalSelfAttnMixer(nn.Module):
    """Standard 8-head causal attention."""
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
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1,2).contiguous().view(B, L, D))


# Block that can pass embedding for value residual
class BlockVR(nn.Module):
    """Block that optionally passes embedding to mixer for value residual."""
    def __init__(self, d, dff, mixer, pass_embed=False):
        super().__init__()
        self.norm1 = nn.RMSNorm(d)
        self.mixer = mixer
        self.norm2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, dff)
        self.pass_embed = pass_embed

    def forward(self, x, embed=None):
        if self.pass_embed and embed is not None:
            x = x + self.mixer(self.norm1(x), embed=embed)
        else:
            x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ═══════════════════════════════════════════════════════════════════════
# V7 Architectures
# ═══════════════════════════════════════════════════════════════════════

def _make_v7(config, conv_layers, attn_positions, mixer_fn=None, conv_fn=None):
    """Helper to build conv+single-head-attn models with flexible placement."""
    d, dff = config.d_model, config.d_ff
    total = conv_layers + len(attn_positions)
    if conv_fn is None:
        conv_fn = lambda: GatedMHConvMixer(d)
    if mixer_fn is None:
        mixer_fn = lambda: SingleHeadAttnMixer(d)
    blocks = []
    attn_set = set(attn_positions)
    for i in range(total):
        if i in attn_set:
            blocks.append(Block(d, dff, mixer_fn()))
        else:
            blocks.append(Block(d, dff, conv_fn()))
    return blocks


# V7_01: SingleHead at MIDDLE (layer 11) instead of end
@register_arch("V7_01_SingleHeadMidLM", "novel_v7", "21 GatedMHConv + 1 single-head attn at middle")
class V7_01(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        self.blocks = nn.ModuleList(_make_v7(config, 21, [11]))
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v7"
    def describe(self): return "V7_01: 21 conv + 1 single-head attn at MIDDLE (layer 11)"


# V7_02: SingleHead at BEGINNING (layer 0)
@register_arch("V7_02_SingleHeadStartLM", "novel_v7", "21 GatedMHConv + 1 single-head attn at beginning")
class V7_02(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        self.blocks = nn.ModuleList(_make_v7(config, 21, [0]))
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v7"
    def describe(self): return "V7_02: 1 single-head attn at BEGINNING + 21 conv"


# V7_03: 2 SingleHead layers (middle + end)
@register_arch("V7_03_TwoSingleHeadLM", "novel_v7", "20 GatedMHConv + 2 single-head attn (mid+end)")
class V7_03(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        self.blocks = nn.ModuleList(_make_v7(config, 20, [10, 21]))
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v7"
    def describe(self): return "V7_03: 20 conv + 2 single-head attn (layers 10, 21)"


# V7_04: 3 SingleHead layers (evenly spaced: 7, 14, 21)
@register_arch("V7_04_ThreeSingleHeadLM", "novel_v7", "19 GatedMHConv + 3 single-head attn (7,14,21)")
class V7_04(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        self.blocks = nn.ModuleList(_make_v7(config, 19, [7, 14, 21]))
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v7"
    def describe(self): return "V7_04: 19 conv + 3 single-head attn (layers 7, 14, 21)"


# V7_05: SingleHead with d_head=32 (smaller Q/K, cheaper)
@register_arch("V7_05_SingleHeadSmallQKLM", "novel_v7", "21 GatedMHConv + 1 single-head attn (d_qk=32)")
class V7_05(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [Block(d, dff, GatedMHConvMixer(d)) for _ in range(21)]
        blocks.append(Block(d, dff, SingleHeadAttnMixer(d, d_head=32)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v7"
    def describe(self): return "V7_05: 21 conv + 1 single-head attn (d_qk=32)"


# V7_06: SingleHead with d_head=128 (larger Q/K, more expressive routing)
@register_arch("V7_06_SingleHeadLargeQKLM", "novel_v7", "21 GatedMHConv + 1 single-head attn (d_qk=128)")
class V7_06(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [Block(d, dff, GatedMHConvMixer(d)) for _ in range(21)]
        blocks.append(Block(d, dff, SingleHeadAttnMixer(d, d_head=128)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v7"
    def describe(self): return "V7_06: 21 conv + 1 single-head attn (d_qk=128)"


# V7_07: SingleHead + ScaledRes conv layers (best V4 tweak)
@register_arch("V7_07_ScaledResConvLM", "novel_v7", "21 GatedMHConvScaledRes + 1 single-head attn")
class V7_07(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [Block(d, dff, GatedMHConvScaledRes(d)) for _ in range(21)]
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
    def arch_family(cls): return "novel_v7"
    def describe(self): return "V7_07: 21 GatedMHConvScaledRes + 1 single-head attn at end"


# V7_08: SingleHead with value residual from embedding
@register_arch("V7_08_SingleHeadVRLM", "novel_v7", "21 GatedMHConv + 1 single-head attn with value residual")
class V7_08(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        self.conv_blocks = nn.ModuleList([Block(d, dff, GatedMHConvMixer(d)) for _ in range(21)])
        self.attn_block = BlockVR(d, dff, SingleHeadAttnValueRes(d), pass_embed=True)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        embed = h.detach()  # save for value residual
        for b in self.conv_blocks: h = b(h)
        h = self.attn_block(h, embed=embed)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v7"
    def describe(self): return "V7_08: 21 conv + 1 single-head attn with value residual from embedding"


# V7_09: 2 heads (not 1, not 8) with full-width V per head — 2-head attention
@register_arch("V7_09_TwoHeadFullVLM", "novel_v7", "21 GatedMHConv + 1 two-head attention (d_v=256 per head)")
class V7_09(FrontierModel):
    class TwoHeadAttnMixer(nn.Module):
        def __init__(self, d, d_head=64):
            super().__init__()
            self.nh = 2; self.dh = d_head; self.dv = d // 2
            self.q = nn.Linear(d, 2 * d_head, bias=False)
            self.k = nn.Linear(d, 2 * d_head, bias=False)
            self.v = nn.Linear(d, d, bias=False)
            self.q_norm = nn.RMSNorm(d_head)
            self.k_norm = nn.RMSNorm(d_head)
            self.out = nn.Linear(d, d, bias=False)
        def forward(self, x, **kw):
            B, L, D = x.shape
            q = self.q(x).view(B, L, 2, self.dh)
            k = self.k(x).view(B, L, 2, self.dh)
            v = self.v(x).view(B, L, 2, self.dv)
            q = self.q_norm(q); k = self.k_norm(k)
            q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.out(out.transpose(1,2).contiguous().view(B, L, D))

    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [Block(d, dff, GatedMHConvMixer(d)) for _ in range(21)]
        blocks.append(Block(d, dff, self.TwoHeadAttnMixer(d)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v7"
    def describe(self): return "V7_09: 21 conv + 1 two-head attn (d_v=256 per head)"


# V7_10: 4 heads with d_v=128 per head (between 1-head and 8-head)
@register_arch("V7_10_FourHeadLM", "novel_v7", "21 GatedMHConv + 1 four-head attention (d_v=128 per head)")
class V7_10(FrontierModel):
    class FourHeadAttnMixer(nn.Module):
        def __init__(self, d, d_head=64):
            super().__init__()
            self.nh = 4; self.dh = d_head; self.dv = d // 4
            self.q = nn.Linear(d, 4 * d_head, bias=False)
            self.k = nn.Linear(d, 4 * d_head, bias=False)
            self.v = nn.Linear(d, d, bias=False)
            self.q_norm = nn.RMSNorm(d_head)
            self.k_norm = nn.RMSNorm(d_head)
            self.out = nn.Linear(d, d, bias=False)
        def forward(self, x, **kw):
            B, L, D = x.shape
            q = self.q(x).view(B, L, 4, self.dh)
            k = self.k(x).view(B, L, 4, self.dh)
            v = self.v(x).view(B, L, 4, self.dv)
            q = self.q_norm(q); k = self.k_norm(k)
            q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.out(out.transpose(1,2).contiguous().view(B, L, D))

    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [Block(d, dff, GatedMHConvMixer(d)) for _ in range(21)]
        blocks.append(Block(d, dff, self.FourHeadAttnMixer(d)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v7"
    def describe(self): return "V7_10: 21 conv + 1 four-head attn (d_v=128 per head)"


# V7_11: SingleHead + 8-head standard at end (2 attention layers: one global, one multi-pattern)
@register_arch("V7_11_SinglePlusMultiHeadLM", "novel_v7", "20 GatedMHConv + 1 single-head + 1 multi-head attn")
class V7_11(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [Block(d, dff, GatedMHConvMixer(d)) for _ in range(20)]
        blocks.append(Block(d, dff, SingleHeadAttnMixer(d)))
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
    def arch_family(cls): return "novel_v7"
    def describe(self): return "V7_11: 20 conv + single-head attn + multi-head attn at end"


# V7_12: Reproduce V6_06 exactly (control — verify consistency)
@register_arch("V7_12_V6ControlLM", "novel_v7", "Control: exact copy of V6_06 (21 conv + 1 single-head)")
class V7_12(FrontierModel):
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
    def arch_family(cls): return "novel_v7"
    def describe(self): return "V7_12: Control — exact copy of V6_06 (21 conv + 1 single-head attn)"


# V7_13: SingleHead with d_head=256 (very wide Q/K)
@register_arch("V7_13_SingleHeadWideQKLM", "novel_v7", "21 GatedMHConv + 1 single-head attn (d_qk=256)")
class V7_13(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [Block(d, dff, GatedMHConvMixer(d)) for _ in range(21)]
        blocks.append(Block(d, dff, SingleHeadAttnMixer(d, d_head=256)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v7"
    def describe(self): return "V7_13: 21 conv + 1 single-head attn (d_qk=256)"


# V7_14: 2 SingleHead at end (both last 2 layers)
@register_arch("V7_14_TwoSingleHeadEndLM", "novel_v7", "20 GatedMHConv + 2 single-head attn at end")
class V7_14(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [Block(d, dff, GatedMHConvMixer(d)) for _ in range(20)]
        blocks.append(Block(d, dff, SingleHeadAttnMixer(d)))
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
    def arch_family(cls): return "novel_v7"
    def describe(self): return "V7_14: 20 conv + 2 single-head attn at end"


# V7_15: ScaledRes + SingleHead + value residual (combine all best ideas)
@register_arch("V7_15_KitchenSinkLM", "novel_v7", "21 ScaledRes conv + 1 single-head attn with VR")
class V7_15(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        self.conv_blocks = nn.ModuleList([Block(d, dff, GatedMHConvScaledRes(d)) for _ in range(21)])
        self.attn_block = BlockVR(d, dff, SingleHeadAttnValueRes(d), pass_embed=True)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        embed = h.detach()
        for b in self.conv_blocks: h = b(h)
        h = self.attn_block(h, embed=embed)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v7"
    def describe(self): return "V7_15: 21 ScaledRes conv + 1 single-head attn with value residual"
