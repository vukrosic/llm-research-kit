"""
V10: Final Exploitation + Scale Tests
=======================================
V9 winner: V9_13 (d=640, 16L, VR at layer 10) = 3.668 (Δtrans = -0.116)

V10 goals:
1. Fine-grained placement sweep at d=640 16L (layers 9, 10, 11, 12)
2. Combine conv VR (late) + attn VR + optimal placement
3. Conv VR alpha sweep for late layers
4. Best config with 2 VR attention layers at optimal positions
5. Scale test: run best at 10 minutes
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
    def __init__(self, d, nh=8, max_k=65, alpha=0.2):
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


class V10Base(FrontierModel):
    def _build(self, d, dff, vocab_size, n_total, attn_positions, conv_vr_positions=None,
               d_head=None, alpha=0.5, conv_vr_alpha=0.2):
        if d_head is None: d_head = max(64, d // 8)
        if conv_vr_positions is None: conv_vr_positions = set()
        else: conv_vr_positions = set(conv_vr_positions)
        attn_set = set(attn_positions)
        self.embed = EmbeddingWithScale(vocab_size, d)
        self.vr_all = attn_set | conv_vr_positions
        blocks = []
        for i in range(n_total):
            if i in attn_set:
                blocks.append(MixerBlock(d, dff, SingleHeadAttnVR(d, d_head=d_head, alpha=alpha)))
            elif i in conv_vr_positions:
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
            else: h = b(h)
        return self.head(self.norm(h))


# V10_01: VR at layer 9 (fine-grained placement)
@register_arch("V10_01_VR9LM", "novel_v10", "d=640 16L, VR at layer 9")
class V10_01(V10Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c); self._build(640, 2560, c.vocab_size, 16, [9], d_head=80)
    @classmethod
    def arch_family(cls): return "novel_v10"
    def describe(self): return "V10_01: d=640 16L, VR at layer 9"

# V10_02: VR at layer 11
@register_arch("V10_02_VR11LM", "novel_v10", "d=640 16L, VR at layer 11")
class V10_02(V10Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c); self._build(640, 2560, c.vocab_size, 16, [11], d_head=80)
    @classmethod
    def arch_family(cls): return "novel_v10"
    def describe(self): return "V10_02: d=640 16L, VR at layer 11"

# V10_03: VR at layer 12
@register_arch("V10_03_VR12LM", "novel_v10", "d=640 16L, VR at layer 12")
class V10_03(V10Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c); self._build(640, 2560, c.vocab_size, 16, [12], d_head=80)
    @classmethod
    def arch_family(cls): return "novel_v10"
    def describe(self): return "V10_03: d=640 16L, VR at layer 12"

# V10_04: VR@10 + conv VR on layers 11-14 (combine two best findings)
@register_arch("V10_04_VR10ConvVRLM", "novel_v10", "d=640 16L, attn VR@10 + conv VR@11-14")
class V10_04(V10Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        self._build(640, 2560, c.vocab_size, 16, [10],
                    conv_vr_positions=[11, 12, 13, 14], d_head=80, conv_vr_alpha=0.15)
    @classmethod
    def arch_family(cls): return "novel_v10"
    def describe(self): return "V10_04: attn VR@10 + conv VR@11-14"

# V10_05: VR@10 + conv VR on layers 8-9,11-14 (more conv VR)
@register_arch("V10_05_VR10ConvVR7LM", "novel_v10", "d=640 16L, attn VR@10 + conv VR@8-9,11-14")
class V10_05(V10Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        self._build(640, 2560, c.vocab_size, 16, [10],
                    conv_vr_positions=[8, 9, 11, 12, 13, 14], d_head=80, conv_vr_alpha=0.15)
    @classmethod
    def arch_family(cls): return "novel_v10"
    def describe(self): return "V10_05: attn VR@10 + conv VR@8-14"

# V10_06: 2 VR attn at layers 7 and 13 (thirds placement)
@register_arch("V10_06_TwoVR713LM", "novel_v10", "d=640 16L, 2 VR attn at layers 7,13")
class V10_06(V10Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c); self._build(640, 2560, c.vocab_size, 16, [7, 13], d_head=80)
    @classmethod
    def arch_family(cls): return "novel_v10"
    def describe(self): return "V10_06: 2 VR attn at layers 7, 13"

# V10_07: 2 VR attn at layers 8 and 12
@register_arch("V10_07_TwoVR812LM", "novel_v10", "d=640 16L, 2 VR attn at layers 8,12")
class V10_07(V10Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c); self._build(640, 2560, c.vocab_size, 16, [8, 12], d_head=80)
    @classmethod
    def arch_family(cls): return "novel_v10"
    def describe(self): return "V10_07: 2 VR attn at layers 8, 12"

# V10_08: VR@10 with alpha=0.6 (fine-tune alpha)
@register_arch("V10_08_VR10A06LM", "novel_v10", "d=640 16L, VR@10 alpha=0.6")
class V10_08(V10Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c); self._build(640, 2560, c.vocab_size, 16, [10], d_head=80, alpha=0.6)
    @classmethod
    def arch_family(cls): return "novel_v10"
    def describe(self): return "V10_08: VR@10 alpha=0.6"

# V10_09: VR@10 with alpha=0.8
@register_arch("V10_09_VR10A08LM", "novel_v10", "d=640 16L, VR@10 alpha=0.8")
class V10_09(V10Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c); self._build(640, 2560, c.vocab_size, 16, [10], d_head=80, alpha=0.8)
    @classmethod
    def arch_family(cls): return "novel_v10"
    def describe(self): return "V10_09: VR@10 alpha=0.8"

# V10_10: VR@10 + conv VR@12-14 with conv alpha=0.1
@register_arch("V10_10_VR10ConvA01LM", "novel_v10", "d=640 16L, attn VR@10 + conv VR@12-14 (a=0.1)")
class V10_10(V10Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        self._build(640, 2560, c.vocab_size, 16, [10],
                    conv_vr_positions=[12, 13, 14], d_head=80, conv_vr_alpha=0.1)
    @classmethod
    def arch_family(cls): return "novel_v10"
    def describe(self): return "V10_10: VR@10 + conv VR@12-14 (alpha=0.1)"

# V10_11: Control — exact V9_13 copy
@register_arch("V10_11_ControlLM", "novel_v10", "Control: V9_13 copy (d=640 16L VR@10)")
class V10_11(V10Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c); self._build(640, 2560, c.vocab_size, 16, [10], d_head=80)
    @classmethod
    def arch_family(cls): return "novel_v10"
    def describe(self): return "V10_11: Control — V9_13 copy"

# V10_12: VR@10 seed=43
@register_arch("V10_12_Seed43LM", "novel_v10", "d=640 16L VR@10 (different seed verification)")
class V10_12(V10Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c); self._build(640, 2560, c.vocab_size, 16, [10], d_head=80)
    @classmethod
    def arch_family(cls): return "novel_v10"
    def describe(self): return "V10_12: V9_13 copy (seed 43 — runner handles seed)"
