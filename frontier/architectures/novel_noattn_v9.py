"""
V9: Width Scaling + VR Optimization
=====================================
V8 winner: V8_13 (d=640, 16L, 1 VR single-head at end) = 3.684
Key findings to combine:
1. Width scaling: d=640 >> d=512 at equal compute
2. VR at middle (layer 11 of 22) = best placement
3. 2 VR layers (mid+end) marginally helps
4. Alpha 0.5-0.9 all good

V9 explores:
- Width scaling + middle VR placement
- Even wider (d=704, 768) with fewer layers
- 12M token scale test on best architecture
- Conv layers with value residual too
- Different layer counts at d=640
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class GatedMHConvVR(nn.Module):
    """GatedMHConv with value residual — blend embedding into conv values."""
    def __init__(self, d, nh=8, max_k=65, alpha=0.3):
        super().__init__()
        self.nh = nh; self.dh = d // nh; self.alpha = alpha
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks
        ])
        self.head_gates = nn.Linear(d, nh)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, embed=None, **kw):
        B, L, D = x.shape
        v = self.v(x)
        if embed is not None:
            v = (1 - self.alpha) * v + self.alpha * embed
        v = v.view(B, L, self.nh, self.dh)
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
        if embed is not None:
            v = (1 - self.alpha) * v + self.alpha * embed
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


# Base class for V9 models
class V9Base(FrontierModel):
    def _build(self, d, dff, vocab_size, n_conv, attn_positions, vr_conv_positions=None,
               d_head=None, alpha=0.5, conv_vr_alpha=0.3):
        if d_head is None:
            d_head = max(64, d // 8)
        if vr_conv_positions is None:
            vr_conv_positions = set()
        else:
            vr_conv_positions = set(vr_conv_positions)

        self.embed = EmbeddingWithScale(vocab_size, d)
        total = n_conv + len(attn_positions)
        attn_set = set(attn_positions)
        self.attn_set = attn_set
        self.vr_conv_set = vr_conv_positions
        self.vr_all = attn_set | vr_conv_positions

        blocks = []
        for i in range(total):
            if i in attn_set:
                blocks.append(MixerBlock(d, dff, SingleHeadAttnVR(d, d_head=d_head, alpha=alpha)))
            elif i in vr_conv_positions:
                blocks.append(MixerBlock(d, dff, GatedMHConvVR(d, alpha=conv_vr_alpha)))
            else:
                blocks.append(MixerBlock(d, dff, GatedMHConvMixer(d)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, vocab_size, self.embed.embedding.weight)
        self.apply(_init)

    def forward(self, x):
        h = self.embed(x); emb = h.detach()
        for i, b in enumerate(self.blocks):
            if i in self.vr_all:
                h = h + b.mixer(b.norm1(h), embed=emb); h = h + b.ffn(b.norm2(h))
            else:
                h = b(h)
        return self.head(self.norm(h))


# V9_01: d=640 16L, VR at MIDDLE (layer 8)
@register_arch("V9_01_W640VRMidLM", "novel_v9", "d=640 16L: 15 conv + 1 VR single-head at middle")
class V9_01(V9Base):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        self._build(640, 2560, config.vocab_size, 15, [8], d_head=80)
    @classmethod
    def arch_family(cls): return "novel_v9"
    def describe(self): return "V9_01: d=640 16L, VR at middle (layer 8)"


# V9_02: d=640 16L, 2 VR (mid+end)
@register_arch("V9_02_W640TwoVRLM", "novel_v9", "d=640 16L: 14 conv + 2 VR single-head (mid+end)")
class V9_02(V9Base):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        self._build(640, 2560, config.vocab_size, 14, [7, 15], d_head=80)
    @classmethod
    def arch_family(cls): return "novel_v9"
    def describe(self): return "V9_02: d=640 16L, 2 VR at layers 7,15"


# V9_03: d=704 14L, VR at end
@register_arch("V9_03_W704VREndLM", "novel_v9", "d=704 14L: 13 conv + 1 VR single-head at end")
class V9_03(V9Base):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        self._build(704, 2816, config.vocab_size, 13, [13], d_head=88)
    @classmethod
    def arch_family(cls): return "novel_v9"
    def describe(self): return "V9_03: d=704 14L, VR at end"


# V9_04: d=704 14L, VR at middle
@register_arch("V9_04_W704VRMidLM", "novel_v9", "d=704 14L: 13 conv + 1 VR single-head at middle")
class V9_04(V9Base):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        self._build(704, 2816, config.vocab_size, 13, [7], d_head=88)
    @classmethod
    def arch_family(cls): return "novel_v9"
    def describe(self): return "V9_04: d=704 14L, VR at middle"


# V9_05: d=768 12L, VR at end
@register_arch("V9_05_W768VREndLM", "novel_v9", "d=768 12L: 11 conv + 1 VR single-head at end")
class V9_05(V9Base):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        self._build(768, 3072, config.vocab_size, 11, [11], d_head=96)
    @classmethod
    def arch_family(cls): return "novel_v9"
    def describe(self): return "V9_05: d=768 12L, VR at end"


# V9_06: d=768 12L, VR at middle
@register_arch("V9_06_W768VRMidLM", "novel_v9", "d=768 12L: 11 conv + 1 VR single-head at middle")
class V9_06(V9Base):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        self._build(768, 3072, config.vocab_size, 11, [6], d_head=96)
    @classmethod
    def arch_family(cls): return "novel_v9"
    def describe(self): return "V9_06: d=768 12L, VR at middle"


# V9_07: d=640 18L, VR at end (deeper than V8_13)
@register_arch("V9_07_W640D18VREndLM", "novel_v9", "d=640 18L: 17 conv + 1 VR single-head at end")
class V9_07(V9Base):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        self._build(640, 2560, config.vocab_size, 17, [17], d_head=80)
    @classmethod
    def arch_family(cls): return "novel_v9"
    def describe(self): return "V9_07: d=640 18L, VR at end"


# V9_08: d=640 18L, VR at middle
@register_arch("V9_08_W640D18VRMidLM", "novel_v9", "d=640 18L: 17 conv + 1 VR single-head at middle")
class V9_08(V9Base):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        self._build(640, 2560, config.vocab_size, 17, [9], d_head=80)
    @classmethod
    def arch_family(cls): return "novel_v9"
    def describe(self): return "V9_08: d=640 18L, VR at middle"


# V9_09: d=640 16L, conv VR at all layers + attn VR at end
@register_arch("V9_09_ConvVRLM", "novel_v9", "d=640 16L: 15 conv(VR) + 1 attn(VR) at end")
class V9_09(V9Base):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        self._build(640, 2560, config.vocab_size, 15, [15],
                    vr_conv_positions=list(range(15)), d_head=80, conv_vr_alpha=0.2)
    @classmethod
    def arch_family(cls): return "novel_v9"
    def describe(self): return "V9_09: d=640 16L, ALL conv+attn have VR (conv alpha=0.2)"


# V9_10: d=640 16L, conv VR at last 5 layers only + attn VR
@register_arch("V9_10_ConvVRLate5LM", "novel_v9", "d=640 16L: 10 conv + 5 conv(VR) + 1 attn(VR)")
class V9_10(V9Base):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        self._build(640, 2560, config.vocab_size, 15, [15],
                    vr_conv_positions=[10, 11, 12, 13, 14], d_head=80, conv_vr_alpha=0.2)
    @classmethod
    def arch_family(cls): return "novel_v9"
    def describe(self): return "V9_10: d=640 16L, last 5 conv + attn have VR"


# V9_11: d=640 14L (less layers, wider FFN = d*5 instead of d*4)
@register_arch("V9_11_W640D14WideFFNLM", "novel_v9", "d=640 14L: 13 conv + 1 VR, wider FFN (d*5)")
class V9_11(V9Base):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        self._build(640, 3200, config.vocab_size, 13, [13], d_head=80)  # dff=5*d
    @classmethod
    def arch_family(cls): return "novel_v9"
    def describe(self): return "V9_11: d=640 14L, VR end, wider FFN (dff=3200)"


# V9_12: d=640 20L, VR at middle (deeper but narrower FFN)
@register_arch("V9_12_W640D20VRMidLM", "novel_v9", "d=640 20L: 19 conv + 1 VR at middle, narrow FFN")
class V9_12(V9Base):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        self._build(640, 2048, config.vocab_size, 19, [10], d_head=80)  # narrower FFN to match params
    @classmethod
    def arch_family(cls): return "novel_v9"
    def describe(self): return "V9_12: d=640 20L, VR at middle, narrow FFN (dff=2048)"


# V9_13: d=640 16L, VR at layer 10 (slightly later than mid)
@register_arch("V9_13_W640VR10LM", "novel_v9", "d=640 16L: 15 conv + 1 VR at layer 10")
class V9_13(V9Base):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        self._build(640, 2560, config.vocab_size, 15, [10], d_head=80)
    @classmethod
    def arch_family(cls): return "novel_v9"
    def describe(self): return "V9_13: d=640 16L, VR at layer 10"


# V9_14: V8_13 control (d=640 16L VR end) for reproducibility
@register_arch("V9_14_ControlLM", "novel_v9", "Control: d=640 16L VR end (V8_13 copy)")
class V9_14(V9Base):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        self._build(640, 2560, config.vocab_size, 15, [15], d_head=80)
    @classmethod
    def arch_family(cls): return "novel_v9"
    def describe(self): return "V9_14: Control — V8_13 copy (d=640 16L VR end)"


# V9_15: d=640 16L, VR at mid + alpha=0.7
@register_arch("V9_15_W640VRMidA07LM", "novel_v9", "d=640 16L: 15 conv + 1 VR at mid (alpha=0.7)")
class V9_15(V9Base):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        self._build(640, 2560, config.vocab_size, 15, [8], d_head=80, alpha=0.7)
    @classmethod
    def arch_family(cls): return "novel_v9"
    def describe(self): return "V9_15: d=640 16L, VR at mid, alpha=0.7"
