"""
V13: Radical Topology Changes
==============================
V10-V11 confirm conv+SingleHead+VR is plateaued. V12 tests multi-query, ALiBi, recurrence.

V13 tries fundamentally different information flow:
1. Two-stage: first 8L conv → attention → second 8L conv (reset residual)
2. Bottleneck attention: compress d→d/2 before attn, expand after (cheaper attn, more capacity)
3. Repeated attention: pass through attention TWICE (2 passes, same weights)
4. Conv layers with INCREASING width (narrow→wide progressive)
5. Attention on difference (x - embed) rather than x — attend to what conv CHANGED
6. Dual-head: two single-head attentions at same layer with different d_qk, gated
7. Pre-attention feature extraction: dedicated projection before attn
8. Higher LR for attention params (split optimizer groups)
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
# NEW: Attention on DIFFERENCE (what conv changed, not raw hidden state)
# ═══════════════════════════════════════════════════════════════════════
class DiffAttnVR(nn.Module):
    """Attend to (x - embed) = what the conv layers actually learned.
    V still uses VR blend for raw token identity."""
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
        # Q and K from the difference (what conv learned)
        diff = x - embed if embed is not None else x
        q = self.q_norm(self.q(diff)).unsqueeze(1)
        k = self.k_norm(self.k(diff)).unsqueeze(1)
        # V still blends with embedding
        v = self.v(x)
        if embed is not None: v = (1 - self.alpha) * v + self.alpha * embed
        return self.out(F.scaled_dot_product_attention(q, k, v.unsqueeze(1), is_causal=True).squeeze(1))


# ═══════════════════════════════════════════════════════════════════════
# NEW: Dual single-head attention (2 heads with different d_qk, gated)
# ═══════════════════════════════════════════════════════════════════════
class DualHeadAttnVR(nn.Module):
    """Two single-head attentions: one wide (d_qk=128) for coarse routing,
    one narrow (d_qk=32) for fine-grained. Both share V, gated output."""
    def __init__(self, d, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        # Wide head for coarse global routing
        self.q1 = nn.Linear(d, 128, bias=False)
        self.k1 = nn.Linear(d, 128, bias=False)
        self.q1_norm = nn.RMSNorm(128)
        self.k1_norm = nn.RMSNorm(128)
        # Narrow head for fine-grained
        self.q2 = nn.Linear(d, 32, bias=False)
        self.k2 = nn.Linear(d, 32, bias=False)
        self.q2_norm = nn.RMSNorm(32)
        self.k2_norm = nn.RMSNorm(32)
        # Shared V
        self.v = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, 1)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, embed=None, **kw):
        B, L, D = x.shape
        v = self.v(x)
        if embed is not None: v = (1 - self.alpha) * v + self.alpha * embed

        # Head 1: wide QK
        q1 = self.q1_norm(self.q1(x)).unsqueeze(1)
        k1 = self.k1_norm(self.k1(x)).unsqueeze(1)
        out1 = F.scaled_dot_product_attention(q1, k1, v.unsqueeze(1), is_causal=True).squeeze(1)

        # Head 2: narrow QK
        q2 = self.q2_norm(self.q2(x)).unsqueeze(1)
        k2 = self.k2_norm(self.k2(x)).unsqueeze(1)
        out2 = F.scaled_dot_product_attention(q2, k2, v.unsqueeze(1), is_causal=True).squeeze(1)

        g = torch.sigmoid(self.gate(x))
        return self.out(g * out1 + (1 - g) * out2)


# ═══════════════════════════════════════════════════════════════════════
# NEW: Bottleneck attention (compress → attend → expand)
# ═══════════════════════════════════════════════════════════════════════
class BottleneckAttnVR(nn.Module):
    """Compress d→d_bottleneck before attention, expand after.
    Attention is O(L² × d_bottleneck) instead of O(L² × d)."""
    def __init__(self, d, d_bn=320, d_head=64, alpha=0.5):
        super().__init__()
        self.dh = d_head; self.alpha = alpha
        self.compress = nn.Linear(d, d_bn, bias=False)
        self.q = nn.Linear(d_bn, d_head, bias=False)
        self.k = nn.Linear(d_bn, d_head, bias=False)
        self.v = nn.Linear(d_bn, d_bn, bias=False)
        self.q_norm = nn.RMSNorm(d_head)
        self.k_norm = nn.RMSNorm(d_head)
        self.expand = nn.Linear(d_bn, d, bias=False)
        self.compress_emb = nn.Linear(d, d_bn, bias=False)

    def forward(self, x, embed=None, **kw):
        B, L, D = x.shape
        x_c = self.compress(x)  # (B, L, d_bn)
        q = self.q_norm(self.q(x_c)).unsqueeze(1)
        k = self.k_norm(self.k(x_c)).unsqueeze(1)
        v = self.v(x_c)
        if embed is not None:
            emb_c = self.compress_emb(embed)
            v = (1 - self.alpha) * v + self.alpha * emb_c
        out = F.scaled_dot_product_attention(q, k, v.unsqueeze(1), is_causal=True).squeeze(1)
        return self.expand(out)


# ═══════════════════════════════════════════════════════════════════════
# NEW: Repeated attention (same weights, 2 forward passes)
# ═══════════════════════════════════════════════════════════════════════
class RepeatedAttnVR(nn.Module):
    """Run the same attention twice. Second pass sees output of first pass."""
    def __init__(self, d, d_head=64, alpha=0.5):
        super().__init__()
        self.attn = SingleHeadAttnVR(d, d_head=d_head, alpha=alpha)
        self.norm_mid = nn.RMSNorm(d)

    def forward(self, x, embed=None, **kw):
        # First pass
        out1 = self.attn(x, embed=embed)
        # Second pass on first output
        out2 = self.attn(self.norm_mid(x + out1), embed=embed)
        return out1 + out2


# ═══════════════════════════════════════════════════════════════════════
# V13 Architecture Definitions
# ═══════════════════════════════════════════════════════════════════════

class V13Base(FrontierModel):
    def _build(self, d, dff, vocab_size, n_total, special_layers):
        self.embed = EmbeddingWithScale(vocab_size, d)
        self.vr_set = set(special_layers.keys())
        blocks = []
        for i in range(n_total):
            if i in special_layers:
                blocks.append(MixerBlock(d, dff, special_layers[i]))
            else:
                blocks.append(MixerBlock(d, dff, GatedMHConvMixer(d)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, vocab_size, self.embed.embedding.weight)
        self.apply(_init)

    def forward(self, x):
        h = self.embed(x); emb = h.detach()
        for i, b in enumerate(self.blocks):
            if i in self.vr_set:
                h = h + b.mixer(b.norm1(h), embed=emb); h = h + b.ffn(b.norm2(h))
            else: h = b(h)
        return self.head(self.norm(h))


# V13_01: Attention on difference (what conv learned, not raw hidden state)
@register_arch("V13_01_DiffAttnVRLM", "novel_v13", "d=640 16L, attention on (x-embed) at layer 10")
class V13_01(V13Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        self._build(640, 2560, c.vocab_size, 16,
                    {10: DiffAttnVR(640, d_head=80, alpha=0.5)})
    @classmethod
    def arch_family(cls): return "novel_v13"
    def describe(self): return "V13_01: Attention on diff (x-embed) + VR at layer 10"


# V13_02: Dual-head attention (wide+narrow QK, shared V)
@register_arch("V13_02_DualHeadVRLM", "novel_v13", "d=640 16L, dual-head (d128+d32) VR at layer 10")
class V13_02(V13Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        self._build(640, 2560, c.vocab_size, 16,
                    {10: DualHeadAttnVR(640, alpha=0.5)})
    @classmethod
    def arch_family(cls): return "novel_v13"
    def describe(self): return "V13_02: Dual-head (d128+d32) VR at layer 10"


# V13_03: Bottleneck attention (d=640→320 before attn)
@register_arch("V13_03_BottleneckAttnLM", "novel_v13", "d=640 16L, bottleneck (640→320) attn at layer 10")
class V13_03(V13Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        self._build(640, 2560, c.vocab_size, 16,
                    {10: BottleneckAttnVR(640, d_bn=320, d_head=80, alpha=0.5)})
    @classmethod
    def arch_family(cls): return "novel_v13"
    def describe(self): return "V13_03: Bottleneck (640→320) attention at layer 10"


# V13_04: Repeated attention (same weights, 2 passes)
@register_arch("V13_04_RepeatedAttnLM", "novel_v13", "d=640 16L, repeated attn (2 passes) at layer 10")
class V13_04(V13Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        self._build(640, 2560, c.vocab_size, 16,
                    {10: RepeatedAttnVR(640, d_head=80, alpha=0.5)})
    @classmethod
    def arch_family(cls): return "novel_v13"
    def describe(self): return "V13_04: Repeated attention (2 passes, shared weights)"


# V13_05: Two-stage architecture — conv(8L) → attn → conv(8L) with residual reset
@register_arch("V13_05_TwoStageLM", "novel_v13", "d=640 8+1+7=16L, two-stage conv→attn→conv")
class V13_05(FrontierModel):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        d = 640; dff = 2560
        self.embed = EmbeddingWithScale(c.vocab_size, d)
        # Stage 1: 8 conv layers
        self.stage1 = nn.ModuleList([MixerBlock(d, dff, GatedMHConvMixer(d)) for _ in range(8)])
        # Bridge: attention
        self.attn = SingleHeadAttnVR(d, d_head=80, alpha=0.5)
        self.attn_norm = nn.RMSNorm(d)
        self.attn_ffn_norm = nn.RMSNorm(d)
        self.attn_ffn = SwiGLU(d, dff)
        # Stage 2: 7 conv layers
        self.stage2 = nn.ModuleList([MixerBlock(d, dff, GatedMHConvMixer(d)) for _ in range(7)])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, c.vocab_size, self.embed.embedding.weight)
        self.apply(_init)

    def forward(self, x):
        h = self.embed(x); emb = h.detach()
        # Stage 1
        for b in self.stage1: h = b(h)
        # Attention bridge with VR
        h = h + self.attn(self.attn_norm(h), embed=emb)
        h = h + self.attn_ffn(self.attn_ffn_norm(h))
        # Stage 2
        for b in self.stage2: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v13"
    def describe(self): return "V13_05: Two-stage (8conv→attn→7conv)"


# V13_06: Higher LR for attention (tag attn params for optimizer split)
@register_arch("V13_06_HighLRAttnLM", "novel_v13", "d=640 16L, VR@10 with 2x attn LR")
class V13_06(V13Base):
    """Same as control but attention params get 2x learning rate."""
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        self._build(640, 2560, c.vocab_size, 16,
                    {10: SingleHeadAttnVR(640, d_head=80, alpha=0.5)})
        # Tag attention params
        for name, param in self.named_parameters():
            if 'blocks.10.mixer' in name:
                param._is_attn = True

    def get_optimizer_groups(self):
        """Override to give attention params 2x LR."""
        muon_params = []
        adamw_params = []
        attn_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad: continue
            if hasattr(p, '_is_attn') and p._is_attn:
                attn_params.append(p)
            elif p.ndim >= 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        # Return attn_params separately — runner will need to handle this
        return muon_params, adamw_params, attn_params

    @classmethod
    def arch_family(cls): return "novel_v13"
    def describe(self): return "V13_06: VR@10 with 2x attention LR"


# V13_07: Progressive widening (narrow early, wide late)
@register_arch("V13_07_ProgressiveWideLM", "novel_v13", "Progressive: d=512→768 over 16 layers")
class V13_07(FrontierModel):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        # Layers 0-7: d=512, Layers 8-15: d=768
        # Need projection at transition
        d1 = 512; d2 = 768; dff1 = 2048; dff2 = 3072
        self.embed = EmbeddingWithScale(c.vocab_size, d1)
        self.stage1 = nn.ModuleList([MixerBlock(d1, dff1, GatedMHConvMixer(d1)) for _ in range(8)])
        self.proj_up = nn.Linear(d1, d2, bias=False)
        self.stage2_conv = nn.ModuleList()
        for i in range(7):
            self.stage2_conv.append(MixerBlock(d2, dff2, GatedMHConvMixer(d2)))
        # Attention at layer 10 (2nd in stage2)
        self.attn = SingleHeadAttnVR(d2, d_head=96, alpha=0.5)
        self.attn_norm = nn.RMSNorm(d2)
        self.attn_ffn_norm = nn.RMSNorm(d2)
        self.attn_ffn = SwiGLU(d2, dff2)
        self.norm = nn.RMSNorm(d2)
        self.head_proj = nn.Linear(d2, c.vocab_size, bias=False)
        self.apply(_init)

    def forward(self, x):
        h = self.embed(x); emb_narrow = h.detach()
        for b in self.stage1: h = b(h)
        h = self.proj_up(h)
        emb_wide = self.proj_up(emb_narrow).detach()
        for i, b in enumerate(self.stage2_conv):
            if i == 2:  # attention at position ~10
                h = h + self.attn(self.attn_norm(h), embed=emb_wide)
                h = h + self.attn_ffn(self.attn_ffn_norm(h))
            h = b(h)
        return self.head_proj(self.norm(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def arch_family(cls): return "novel_v13"
    def describe(self): return "V13_07: Progressive width (512→768)"


# V13_08: VR with alpha=0.5 but using LEARNED projection for VR instead of raw embed
@register_arch("V13_08_ProjectedVRLM", "novel_v13", "d=640 16L, projected VR (learned transform of embed)")
class V13_08(FrontierModel):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        d = 640; dff = 2560
        self.embed = EmbeddingWithScale(c.vocab_size, d)
        blocks = []
        for i in range(16):
            if i == 10:
                blocks.append(MixerBlock(d, dff, SingleHeadAttnVR(d, d_head=80, alpha=0.5)))
            else:
                blocks.append(MixerBlock(d, dff, GatedMHConvMixer(d)))
        self.blocks = nn.ModuleList(blocks)
        # Learned VR projection
        self.vr_proj = nn.Sequential(
            nn.Linear(d, d, bias=False),
            nn.SiLU(),
            nn.Linear(d, d, bias=False)
        )
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, c.vocab_size, self.embed.embedding.weight)
        self.apply(_init)

    def forward(self, x):
        h = self.embed(x)
        emb = self.vr_proj(h.detach())  # Learned transform of embedding
        for i, b in enumerate(self.blocks):
            if i == 10:
                h = h + b.mixer(b.norm1(h), embed=emb); h = h + b.ffn(b.norm2(h))
            else: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v13"
    def describe(self): return "V13_08: Projected VR (learned embed transform)"


# V13_09: Control (V9_13 copy)
@register_arch("V13_09_ControlLM", "novel_v13", "Control: V9_13 copy (d=640 16L VR@10)")
class V13_09(V13Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        self._build(640, 2560, c.vocab_size, 16,
                    {10: SingleHeadAttnVR(640, d_head=80, alpha=0.5)})
    @classmethod
    def arch_family(cls): return "novel_v13"
    def describe(self): return "V13_09: Control (V9_13 copy)"
