"""
V11: Beyond the Plateau — Novel Mechanisms
============================================
V10 confirmed the conv+SingleHead+VR paradigm is plateaued at ~3.665.
V11 tries fundamentally new ideas:

1. PARALLEL conv+attn within a single layer (not sequential)
2. Learned mixture-of-mechanisms (router picks conv vs attn per token)
3. Additive attention (no softmax, O(n) but content-dependent)
4. Cross-layer connections (DenseNet-style for conv stack)
5. Progressive widening (narrow early, wide late)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from frontier.architectures.base import FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
from frontier.architectures.registry import register_arch
from frontier.architectures.batch100 import SwiGLU, Block, _init


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


class SingleHeadAttnVR(nn.Module):
    def __init__(self, d, d_head=64, alpha=0.5):
        super().__init__()
        self.dh = d_head; self.alpha = alpha
        self.q = nn.Linear(d, d_head, bias=False)
        self.k = nn.Linear(d, d_head, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.q_norm = nn.RMSNorm(d_head)
        self.k_norm = nn.RMSNorm(d_head)
        self.out = nn.Linear(d, d, bias=False)
    def forward(self, x, embed=None, **kw):
        B, L, D = x.shape
        q = self.q_norm(self.q(x)).unsqueeze(1)
        k = self.k_norm(self.k(x)).unsqueeze(1)
        v = self.v(x)
        if embed is not None: v = (1 - self.alpha) * v + self.alpha * embed
        return self.out(F.scaled_dot_product_attention(q, k, v.unsqueeze(1), is_causal=True).squeeze(1))


class MixerBlock(nn.Module):
    def __init__(self, d, dff, mixer):
        super().__init__()
        self.norm1 = nn.RMSNorm(d)
        self.mixer = mixer
        self.norm2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, dff)
    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ═══════════════════════════════════════════════════════════════════════
# NEW MECHANISM: Parallel Conv+Attn within one layer
# ═══════════════════════════════════════════════════════════════════════
class ParallelConvAttnMixer(nn.Module):
    """Run conv and single-head attn in parallel, blend outputs."""
    def __init__(self, d, d_head=64, alpha_vr=0.5):
        super().__init__()
        self.conv = GatedMHConvMixer(d)
        self.attn = SingleHeadAttnVR(d, d_head=d_head, alpha=alpha_vr)
        self.gate = nn.Linear(d, 1)  # learned blend

    def forward(self, x, embed=None, **kw):
        conv_out = self.conv(x)
        attn_out = self.attn(x, embed=embed)
        g = torch.sigmoid(self.gate(x))
        return g * attn_out + (1 - g) * conv_out


# ═══════════════════════════════════════════════════════════════════════
# NEW MECHANISM: Token-Routed (each token picks conv or attn)
# ═══════════════════════════════════════════════════════════════════════
class RoutedConvAttnMixer(nn.Module):
    """Each token routes to conv OR attn via straight-through estimator."""
    def __init__(self, d, d_head=64, alpha_vr=0.5):
        super().__init__()
        self.conv = GatedMHConvMixer(d)
        self.attn = SingleHeadAttnVR(d, d_head=d_head, alpha=alpha_vr)
        self.router = nn.Linear(d, 1)

    def forward(self, x, embed=None, **kw):
        conv_out = self.conv(x)
        attn_out = self.attn(x, embed=embed)
        # Soft routing (hard routing would need custom backward)
        r = torch.sigmoid(self.router(x))
        return r * attn_out + (1 - r) * conv_out


# ═══════════════════════════════════════════════════════════════════════
# NEW MECHANISM: Additive attention (no softmax, content-dependent, O(n))
# ═══════════════════════════════════════════════════════════════════════
class AdditiveAttnMixer(nn.Module):
    """Content-dependent weighting without softmax.
    Each token produces a "relevance score" via learned projection,
    and values are weighted by sigmoid(score) * causal_decay."""
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
        q = self.q_norm(self.q(x))  # (B, L, dh)
        k = self.k_norm(self.k(x))
        v = self.v(x)
        # Content-based scoring: sigmoid(q·k^T / sqrt(d)) but O(n²)
        # For O(n): approximate with cumulative weighted sum
        # Each position attends to weighted average of all previous positions
        # Weight = sigmoid(q_i · k_j / sqrt(d)) — but that's O(n²)
        # Instead: factored — weight = f(q_i) * g(k_j)
        q_gate = torch.sigmoid(q.sum(-1, keepdim=True) / (self.dh ** 0.5))  # (B, L, 1)
        k_gate = torch.sigmoid(k.sum(-1, keepdim=True) / (self.dh ** 0.5))  # (B, L, 1)
        # Cumulative weighted sum: sum(k_gate_j * v_j for j<=i) / sum(k_gate_j for j<=i)
        weighted_v = (k_gate * v).cumsum(dim=1)  # (B, L, D)
        k_sum = k_gate.cumsum(dim=1).clamp(min=1e-6)  # (B, L, 1)
        avg_v = weighted_v / k_sum
        return self.out(q_gate * avg_v)


# ═══════════════════════════════════════════════════════════════════════
# NEW MECHANISM: Cross-layer dense connections for late conv
# ═══════════════════════════════════════════════════════════════════════
class DenseConvBlock(nn.Module):
    """Conv block that takes concatenated input from multiple prior layers.
    Uses a projection to compress the dense connections."""
    def __init__(self, d, dff, n_dense=3):
        super().__init__()
        self.proj = nn.Linear(d * (n_dense + 1), d, bias=False) if n_dense > 0 else None
        self.norm1 = nn.RMSNorm(d)
        self.mixer = GatedMHConvMixer(d)
        self.norm2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, dff)

    def forward(self, x, dense_inputs=None):
        if dense_inputs and self.proj is not None:
            combined = torch.cat([x] + dense_inputs, dim=-1)
            x = self.proj(combined)
        h = x + self.mixer(self.norm1(x))
        h = h + self.ffn(self.norm2(h))
        return h


# ═══════════════════════════════════════════════════════════════════════
# V11 Architectures
# ═══════════════════════════════════════════════════════════════════════

# V11_01: All layers get parallel conv+attn (expensive but explores the idea)
@register_arch("V11_01_ParallelAllLM", "novel_v11", "d=640 16L: all layers parallel conv+attn")
class V11_01(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d = 640; dff = 2560
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        self.blocks = nn.ModuleList([MixerBlock(d, dff, ParallelConvAttnMixer(d, d_head=80)) for _ in range(16)])
        self.vr_all = set(range(16))
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x); emb = h.detach()
        for i, b in enumerate(self.blocks):
            h = h + b.mixer(b.norm1(h), embed=emb); h = h + b.ffn(b.norm2(h))
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v11"
    def describe(self): return "V11_01: all 16 layers parallel conv+attn"


# V11_02: Last 4 layers get parallel conv+attn, first 12 conv only
@register_arch("V11_02_ParallelLast4LM", "novel_v11", "d=640 16L: 12 conv + 4 parallel conv+attn")
class V11_02(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d = 640; dff = 2560
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [MixerBlock(d, dff, GatedMHConvMixer(d)) for _ in range(12)]
        blocks += [MixerBlock(d, dff, ParallelConvAttnMixer(d, d_head=80)) for _ in range(4)]
        self.blocks = nn.ModuleList(blocks)
        self.vr_set = set(range(12, 16))
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x); emb = h.detach()
        for i, b in enumerate(self.blocks):
            if i in self.vr_set:
                h = h + b.mixer(b.norm1(h), embed=emb); h = h + b.ffn(b.norm2(h))
            else: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v11"
    def describe(self): return "V11_02: 12 conv + 4 parallel conv+attn at end"


# V11_03: Parallel at layer 10 only (minimal parallel, compare with pure single-head)
@register_arch("V11_03_Parallel1LM", "novel_v11", "d=640 16L: 15 conv + 1 parallel conv+attn at layer 10")
class V11_03(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d = 640; dff = 2560
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = []
        for i in range(16):
            if i == 10:
                blocks.append(MixerBlock(d, dff, ParallelConvAttnMixer(d, d_head=80)))
            else:
                blocks.append(MixerBlock(d, dff, GatedMHConvMixer(d)))
        self.blocks = nn.ModuleList(blocks)
        self.vr_set = {10}
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x); emb = h.detach()
        for i, b in enumerate(self.blocks):
            if i in self.vr_set:
                h = h + b.mixer(b.norm1(h), embed=emb); h = h + b.ffn(b.norm2(h))
            else: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v11"
    def describe(self): return "V11_03: 15 conv + 1 parallel conv+attn at layer 10"


# V11_04: 2 single-head VR at layers 5,10 + conv VR on 11-14 (best V10 combo at wider scale)
@register_arch("V11_04_TwoVRConvVRLM", "novel_v11", "d=640 16L: 2 VR attn + conv VR late")
class V11_04(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d = 640; dff = 2560
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        from frontier.architectures.novel_noattn_v9 import GatedMHConvVR
        blocks = []
        self.vr_all = set()
        for i in range(16):
            if i in (5, 10):
                blocks.append(MixerBlock(d, dff, SingleHeadAttnVR(d, d_head=80, alpha=0.5)))
                self.vr_all.add(i)
            elif i in (11, 12, 13, 14):
                blocks.append(MixerBlock(d, dff, GatedMHConvVR(d, alpha=0.15)))
                self.vr_all.add(i)
            else:
                blocks.append(MixerBlock(d, dff, GatedMHConvMixer(d)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x); emb = h.detach()
        for i, b in enumerate(self.blocks):
            if i in self.vr_all:
                h = h + b.mixer(b.norm1(h), embed=emb); h = h + b.ffn(b.norm2(h))
            else: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v11"
    def describe(self): return "V11_04: 2 VR attn@5,10 + conv VR@11-14"


# V11_05: Additive attention (O(n), content-dependent) replaces single-head
@register_arch("V11_05_AdditiveAttnLM", "novel_v11", "d=640 16L: 15 conv + 1 additive attn at layer 10")
class V11_05(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d = 640; dff = 2560
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = []
        for i in range(16):
            if i == 10:
                blocks.append(MixerBlock(d, dff, AdditiveAttnMixer(d, d_head=80)))
            else:
                blocks.append(MixerBlock(d, dff, GatedMHConvMixer(d)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v11"
    def describe(self): return "V11_05: 15 conv + 1 additive attn (O(n)) at layer 10"


# V11_06: 3 additive attention layers spread out (O(n) everywhere)
@register_arch("V11_06_AdditiveTripleLM", "novel_v11", "d=640 16L: 13 conv + 3 additive attn")
class V11_06(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d = 640; dff = 2560
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = []
        for i in range(16):
            if i in (5, 10, 14):
                blocks.append(MixerBlock(d, dff, AdditiveAttnMixer(d, d_head=80)))
            else:
                blocks.append(MixerBlock(d, dff, GatedMHConvMixer(d)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v11"
    def describe(self): return "V11_06: 13 conv + 3 additive attn (O(n))"


# V11_07: Dense connections from layers 7,8,9 to layer 10 (before attn)
@register_arch("V11_07_DenseToAttnLM", "novel_v11", "d=640 16L: dense skip connections to attn layer")
class V11_07(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d = 640; dff = 2560
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        self.early_blocks = nn.ModuleList([MixerBlock(d, dff, GatedMHConvMixer(d)) for _ in range(10)])
        # Dense projection: concat layers 7,8,9,10 → compress to d
        self.dense_proj = nn.Linear(d * 4, d, bias=False)
        self.attn_block = MixerBlock(d, dff, SingleHeadAttnVR(d, d_head=80, alpha=0.5))
        self.late_blocks = nn.ModuleList([MixerBlock(d, dff, GatedMHConvMixer(d)) for _ in range(5)])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)

    def forward(self, x):
        h = self.embed(x); emb = h.detach()
        history = []
        for i, b in enumerate(self.early_blocks):
            h = b(h)
            if i >= 7: history.append(h)
        # Dense connection: project concatenated last 3 conv outputs + current
        dense = self.dense_proj(torch.cat(history, dim=-1))
        h = h + dense  # residual add
        # Attention layer with VR
        h = h + self.attn_block.mixer(self.attn_block.norm1(h), embed=emb)
        h = h + self.attn_block.ffn(self.attn_block.norm2(h))
        for b in self.late_blocks: h = b(h)
        return self.head(self.norm(h))

    @classmethod
    def arch_family(cls): return "novel_v11"
    def describe(self): return "V11_07: dense skip (L7-9) → attn@10 + 5 conv"


# V11_08: Baseline control — exact V9_13 at d=640
@register_arch("V11_08_ControlLM", "novel_v11", "Control: V9_13 copy (d=640 16L VR@10)")
class V11_08(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d = 640; dff = 2560
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = []
        for i in range(16):
            if i == 10:
                blocks.append(MixerBlock(d, dff, SingleHeadAttnVR(d, d_head=80, alpha=0.5)))
            else:
                blocks.append(MixerBlock(d, dff, GatedMHConvMixer(d)))
        self.blocks = nn.ModuleList(blocks)
        self.vr_set = {10}
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x); emb = h.detach()
        for i, b in enumerate(self.blocks):
            if i in self.vr_set:
                h = h + b.mixer(b.norm1(h), embed=emb); h = h + b.ffn(b.norm2(h))
            else: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v11"
    def describe(self): return "V11_08: Control (V9_13 copy)"
