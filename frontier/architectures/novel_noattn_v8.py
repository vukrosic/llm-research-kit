"""
V8: Value Residual + Placement Exploitation
=============================================
V7 winner: V7_08 (21 GatedMHConv + 1 SingleHead with ValueResidual) = 3.713
Beats transformer (3.784) by 0.071.

Two independent winning factors:
1. VALUE RESIDUAL: Blending embedding into V gives +0.065 (3.777 → 3.713)
2. MIDDLE PLACEMENT: Placing attention at mid gives +0.033 (3.777 → 3.744)

V8 combines these and explores further:
- VR at middle placement
- VR alpha tuning (0.3, 0.5, 0.7)
- 2 VR single-head layers at different positions
- d_qk=128 + VR
- Width scaling (d=640, fewer layers)
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


class SingleHeadAttnVR(nn.Module):
    """Single-head attention with value residual."""
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


class SingleHeadAttnPlain(nn.Module):
    """Single-head attention without value residual."""
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
        return self.out(F.scaled_dot_product_attention(q, k, v, is_causal=True).squeeze(1))


# Helper: build model with embed-passing for VR layers
class V8Model(FrontierModel):
    """Base for V8 models that need to pass embedding for value residual."""
    def __init__(self, config, blocks, vr_indices, embed=None, norm=None, head=None):
        super().__init__(config)
        self.embed_layer = embed
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm
        self.lm_head = head
        self.vr_indices = set(vr_indices)

    def forward(self, x):
        h = self.embed_layer(x)
        emb = h.detach()
        for i, b in enumerate(self.blocks):
            if i in self.vr_indices:
                h = h + b.mixer(b.norm1(h), embed=emb)
                h = h + b.ffn(b.norm2(h))
            else:
                h = b(h)
        return self.lm_head(self.norm(h))


class MixerBlock(nn.Module):
    """Block with exposed norm1/norm2/mixer/ffn for VR models."""
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


def _build_v8(config, n_conv, attn_positions, d_head=64, alpha=0.5, use_vr=True):
    """Build blocks with VR attention at specified positions."""
    d, dff = config.d_model, config.d_ff
    total = n_conv + len(attn_positions)
    attn_set = set(attn_positions)
    blocks = []
    for i in range(total):
        if i in attn_set:
            if use_vr:
                blocks.append(MixerBlock(d, dff, SingleHeadAttnVR(d, d_head=d_head, alpha=alpha)))
            else:
                blocks.append(MixerBlock(d, dff, SingleHeadAttnPlain(d, d_head=d_head)))
        else:
            blocks.append(MixerBlock(d, dff, GatedMHConvMixer(d)))
    return blocks, attn_set if use_vr else set()


# ═══════════════════════════════════════════════════════════════════════
# V8_01: VR + Middle placement (combine two best V7 findings)
# ═══════════════════════════════════════════════════════════════════════
@register_arch("V8_01_VRMidLM", "novel_v8", "21 conv + 1 single-head VR attn at MIDDLE")
class V8_01(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks, self.vr_set = _build_v8(config, 21, [11], alpha=0.5)
        self.blocks = nn.ModuleList(blocks)
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
    def arch_family(cls): return "novel_v8"
    def describe(self): return "V8_01: VR + Middle (layer 11)"


# V8_02: VR at layer 15 (between middle and end)
@register_arch("V8_02_VR15LM", "novel_v8", "21 conv + 1 single-head VR attn at layer 15")
class V8_02(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks, self.vr_set = _build_v8(config, 21, [15], alpha=0.5)
        self.blocks = nn.ModuleList(blocks)
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
    def arch_family(cls): return "novel_v8"
    def describe(self): return "V8_02: VR at layer 15"


# V8_03: VR alpha=0.3 (less embedding, more processed)
@register_arch("V8_03_VRAlpha03LM", "novel_v8", "21 conv + 1 VR single-head (alpha=0.3) at end")
class V8_03(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks, self.vr_set = _build_v8(config, 21, [21], alpha=0.3)
        self.blocks = nn.ModuleList(blocks)
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
    def arch_family(cls): return "novel_v8"
    def describe(self): return "V8_03: VR alpha=0.3 at end"


# V8_04: VR alpha=0.7 (more embedding)
@register_arch("V8_04_VRAlpha07LM", "novel_v8", "21 conv + 1 VR single-head (alpha=0.7) at end")
class V8_04(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks, self.vr_set = _build_v8(config, 21, [21], alpha=0.7)
        self.blocks = nn.ModuleList(blocks)
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
    def arch_family(cls): return "novel_v8"
    def describe(self): return "V8_04: VR alpha=0.7 at end"


# V8_05: VR alpha=0.9 (mostly embedding)
@register_arch("V8_05_VRAlpha09LM", "novel_v8", "21 conv + 1 VR single-head (alpha=0.9) at end")
class V8_05(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks, self.vr_set = _build_v8(config, 21, [21], alpha=0.9)
        self.blocks = nn.ModuleList(blocks)
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
    def arch_family(cls): return "novel_v8"
    def describe(self): return "V8_05: VR alpha=0.9 at end"


# V8_06: 2 VR single-head at mid and end
@register_arch("V8_06_TwoVRLM", "novel_v8", "20 conv + 2 VR single-head (mid + end)")
class V8_06(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks, self.vr_set = _build_v8(config, 20, [10, 21], alpha=0.5)
        self.blocks = nn.ModuleList(blocks)
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
    def arch_family(cls): return "novel_v8"
    def describe(self): return "V8_06: 2 VR single-head at layers 10 and 21"


# V8_07: VR + d_qk=128 at end (combine both improvements)
@register_arch("V8_07_VR128LM", "novel_v8", "21 conv + 1 VR single-head (d_qk=128) at end")
class V8_07(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks, self.vr_set = _build_v8(config, 21, [21], d_head=128, alpha=0.5)
        self.blocks = nn.ModuleList(blocks)
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
    def arch_family(cls): return "novel_v8"
    def describe(self): return "V8_07: VR + d_qk=128 at end"


# V8_08: VR + d_qk=128 at middle
@register_arch("V8_08_VR128MidLM", "novel_v8", "21 conv + 1 VR single-head (d_qk=128) at middle")
class V8_08(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks, self.vr_set = _build_v8(config, 21, [11], d_head=128, alpha=0.5)
        self.blocks = nn.ModuleList(blocks)
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
    def arch_family(cls): return "novel_v8"
    def describe(self): return "V8_08: VR + d_qk=128 at middle"


# V8_09: VR with LEARNED alpha (per-dimension)
@register_arch("V8_09_VRLearnedAlphaLM", "novel_v8", "21 conv + 1 VR single-head (learned alpha)")
class V8_09(FrontierModel):
    class SingleHeadLearnedVR(nn.Module):
        def __init__(self, d, d_head=64):
            super().__init__()
            self.dh = d_head
            self.q = nn.Linear(d, d_head, bias=False)
            self.k = nn.Linear(d, d_head, bias=False)
            self.v = nn.Linear(d, d, bias=False)
            self.q_norm = nn.RMSNorm(d_head)
            self.k_norm = nn.RMSNorm(d_head)
            self.out = nn.Linear(d, d, bias=False)
            self.alpha = nn.Parameter(torch.full((d,), 0.5))  # per-dim learned
        def forward(self, x, embed=None, **kw):
            B, L, D = x.shape
            q = self.q_norm(self.q(x)).unsqueeze(1)
            k = self.k_norm(self.k(x)).unsqueeze(1)
            v = self.v(x)
            if embed is not None:
                a = torch.sigmoid(self.alpha)
                v = (1 - a) * v + a * embed
            return self.out(F.scaled_dot_product_attention(q, k, v.unsqueeze(1), is_causal=True).squeeze(1))

    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks = [MixerBlock(d, dff, GatedMHConvMixer(d)) for _ in range(21)]
        blocks.append(MixerBlock(d, dff, self.SingleHeadLearnedVR(d)))
        self.blocks = nn.ModuleList(blocks)
        self.vr_set = {21}
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
    def arch_family(cls): return "novel_v8"
    def describe(self): return "V8_09: VR with learned per-dim alpha"


# V8_10: VR with hidden state from layer 10 (not embedding)
@register_arch("V8_10_VRFromMidLM", "novel_v8", "21 conv + 1 VR single-head using layer-10 hidden state")
class V8_10(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        self.blocks = nn.ModuleList([MixerBlock(d, dff, GatedMHConvMixer(d)) for _ in range(21)]
                                     + [MixerBlock(d, dff, SingleHeadAttnVR(d, alpha=0.5))])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size, self.embed.embedding.weight)
        self.apply(_init)
    def forward(self, x):
        h = self.embed(x)
        mid_state = None
        for i, b in enumerate(self.blocks):
            if i == 10:
                mid_state = h.detach()
            if i == 21:
                h = h + b.mixer(b.norm1(h), embed=mid_state); h = h + b.ffn(b.norm2(h))
            else:
                h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v8"
    def describe(self): return "V8_10: VR from layer-10 hidden state (not embedding)"


# V8_11: VR at layer 7 (earlier placement)
@register_arch("V8_11_VR7LM", "novel_v8", "21 conv + 1 VR single-head at layer 7")
class V8_11(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks, self.vr_set = _build_v8(config, 21, [7], alpha=0.5)
        self.blocks = nn.ModuleList(blocks)
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
    def arch_family(cls): return "novel_v8"
    def describe(self): return "V8_11: VR at layer 7"


# V8_12: 3 VR single-head layers spread out (7, 14, 21)
@register_arch("V8_12_ThreeVRLM", "novel_v8", "19 conv + 3 VR single-head (7, 14, 21)")
class V8_12(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks, self.vr_set = _build_v8(config, 19, [7, 14, 21], alpha=0.5)
        self.blocks = nn.ModuleList(blocks)
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
    def arch_family(cls): return "novel_v8"
    def describe(self): return "V8_12: 3 VR single-head at layers 7, 14, 21"


# V8_13: Wider model (d=640, 16 layers) + VR single-head at end
@register_arch("V8_13_Wide640VR_LM", "novel_v8", "15 conv + 1 VR single-head (d=640, 16L)")
class V8_13(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d = 640; dff = 2560  # wider model
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        self.blocks = nn.ModuleList(
            [MixerBlock(d, dff, GatedMHConvMixer(d)) for _ in range(15)]
            + [MixerBlock(d, dff, SingleHeadAttnVR(d, d_head=80, alpha=0.5))]
        )
        self.vr_set = {15}
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
    def arch_family(cls): return "novel_v8"
    def describe(self): return "V8_13: Wide d=640, 16L, VR single-head at end"


# V8_14: VR mid + d_qk=128 (best placement + best d_qk + VR)
@register_arch("V8_14_VR128Mid_LM", "novel_v8", "21 conv + 1 VR single-head (d_qk=128, mid)")
class V8_14(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks, self.vr_set = _build_v8(config, 21, [11], d_head=128, alpha=0.5)
        self.blocks = nn.ModuleList(blocks)
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
    def arch_family(cls): return "novel_v8"
    def describe(self): return "V8_14: VR + d_qk=128 at middle (layer 11)"


# V8_15: VR control (exact copy of V7_08 for reproducibility check)
@register_arch("V8_15_ControlLM", "novel_v8", "Control: exact V7_08 (21 conv + 1 VR single-head end)")
class V8_15(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d, dff = config.d_model, config.d_ff
        self.embed = EmbeddingWithScale(config.vocab_size, d)
        blocks, self.vr_set = _build_v8(config, 21, [21], alpha=0.5)
        self.blocks = nn.ModuleList(blocks)
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
    def arch_family(cls): return "novel_v8"
    def describe(self): return "V8_15: Control — exact copy of V7_08"
