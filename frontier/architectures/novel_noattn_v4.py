"""
Novel Non-Attention Batch 4: Exploit GatedMHConv
=================================================
V2_04_GatedMHConv = 3.915 is the BEST non-attention architecture found.
Gap to transformer: 0.131 (3.915 vs 3.784).

This batch explores variations of GatedMHConv to close the remaining gap.
Every architecture here is a variant of the winning formula:
  Multi-head depthwise causal conv + per-head content-dependent gating.

We vary: gating mechanism, kernel sizes, number of heads, head interaction,
projection structure, normalization, activation functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from frontier.architectures.base import FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
from frontier.architectures.registry import register_arch
from frontier.architectures.batch100 import SwiGLU, Block, EmbedBlock, _init


# ════════════════════════════════════════════════════════════════════════
# Base: GatedMHConv with variations
# ════════════════════════════════════════════════════════════════════════

class GatedMHConv_Base(nn.Module):
    """Base GatedMHConv — this is V2_04 (the winner)."""
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


# V4_01: More heads (16 instead of 8) — finer-grained scale selection
class GatedMHConv16H(nn.Module):
    def __init__(self, d, nh=16, max_k=129):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        ks = [min(2**((i*8)//nh + 1) + 1, max_k) for i in range(nh)]
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
        hs = [F.silu(self.convs[h](v[:,:,h].transpose(1,2))[:,:,:L].transpose(1,2)) * gates[:,:,h:h+1]
              for h in range(self.nh)]
        return self.out(torch.cat(hs, -1))


# V4_02: Softmax gates instead of sigmoid (competitive head selection)
class GatedMHConvSoftmax(nn.Module):
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
        gates = F.softmax(self.head_gates(x), dim=-1)  # competitive!
        hs = [F.silu(self.convs[h](v[:,:,h].transpose(1,2))[:,:,:L].transpose(1,2)) * gates[:,:,h:h+1]
              for h in range(self.nh)]
        return self.out(torch.cat(hs, -1))


# V4_03: Per-head per-CHANNEL gates (more fine-grained than per-head)
class GatedMHConvChannelGate(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks
        ])
        self.channel_gates = nn.Linear(d, d)  # per-channel, not just per-head
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).view(B, L, self.nh, self.dh)
        cg = torch.sigmoid(self.channel_gates(x)).view(B, L, self.nh, self.dh)
        hs = [F.silu(self.convs[h](v[:,:,h].transpose(1,2))[:,:,:L].transpose(1,2)) * cg[:,:,h]
              for h in range(self.nh)]
        return self.out(torch.cat(hs, -1))


# V4_04: GatedMHConv + token shift
class GatedMHConvTS(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.ts_w = nn.Parameter(torch.ones(d) * 0.5)
        self.inner = GatedMHConv_Base(d, nh, max_k)

    def forward(self, x, **kw):
        w = torch.sigmoid(self.ts_w)
        shifted = F.pad(x[:, :-1], (0, 0, 1, 0))
        return self.inner(w * x + (1 - w) * shifted)


# V4_05: GatedMHConv + value residual from embedding
class GatedMHConvVR(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.inner = GatedMHConv_Base(d, nh, max_k)
        self.alpha = nn.Parameter(torch.tensor(0.8))

    def forward(self, x, embed=None, **kw):
        if embed is not None:
            a = torch.sigmoid(self.alpha)
            x = a * x + (1 - a) * embed
        return self.inner(x)


# V4_06: GatedMHConv + cross-head mixing AFTER gating
class GatedMHConvCrossPost(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks
        ])
        self.head_gates = nn.Linear(d, nh)
        self.head_mix = nn.Linear(nh, nh, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).view(B, L, self.nh, self.dh)
        gates = torch.sigmoid(self.head_gates(x))
        conv_outs = []
        for h in range(self.nh):
            co = F.silu(self.convs[h](v[:,:,h].transpose(1,2))[:,:,:L].transpose(1,2))
            conv_outs.append(co * gates[:,:,h:h+1])
        stacked = torch.stack(conv_outs, dim=2)  # (B, L, nh, dh)
        mixed = self.head_mix(stacked.permute(0,1,3,2)).permute(0,1,3,2)
        return self.out(mixed.reshape(B, L, D))


# V4_07: GatedMHConv with GLU-style gating (gate * silu(value))
class GatedMHConvGLU(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        self.v = nn.Linear(d, d, bias=False)
        self.g = nn.Linear(d, d, bias=False)  # separate gate projection
        self.convs = nn.ModuleList([
            nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks
        ])
        self.head_gates = nn.Linear(d, nh)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).view(B, L, self.nh, self.dh)
        g = torch.sigmoid(self.g(x)).view(B, L, self.nh, self.dh)  # GLU gate
        hg = torch.sigmoid(self.head_gates(x))
        hs = []
        for h in range(self.nh):
            co = self.convs[h](v[:,:,h].transpose(1,2))[:,:,:L].transpose(1,2)
            hs.append(F.silu(co) * g[:,:,h] * hg[:,:,h:h+1])
        return self.out(torch.cat(hs, -1))


# V4_08: GatedMHConv with RMSNorm per head (stabilize training)
class GatedMHConvNormed(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks
        ])
        self.head_norms = nn.ModuleList([nn.RMSNorm(self.dh) for _ in range(nh)])
        self.head_gates = nn.Linear(d, nh)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).view(B, L, self.nh, self.dh)
        gates = torch.sigmoid(self.head_gates(x))
        hs = []
        for h in range(self.nh):
            co = self.convs[h](v[:,:,h].transpose(1,2))[:,:,:L].transpose(1,2)
            co = self.head_norms[h](co)  # normalize per head
            hs.append(F.silu(co) * gates[:,:,h:h+1])
        return self.out(torch.cat(hs, -1))


# V4_09: GatedMHConv + output residual (skip connection from input)
class GatedMHConvResidual(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.inner = GatedMHConv_Base(d, nh, max_k)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, **kw):
        return self.inner(x) + torch.sigmoid(self.res_scale) * x


# V4_10: GatedMHConv with 2 separate value projections (richer features)
class GatedMHConvDualV(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        self.v1 = nn.Linear(d, d, bias=False)
        self.v2 = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks
        ])
        self.head_gates = nn.Linear(d, nh)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v1 = self.v1(x).view(B, L, self.nh, self.dh)
        v2 = torch.sigmoid(self.v2(x)).view(B, L, self.nh, self.dh)
        gates = torch.sigmoid(self.head_gates(x))
        hs = []
        for h in range(self.nh):
            co = self.convs[h](v1[:,:,h].transpose(1,2))[:,:,:L].transpose(1,2)
            hs.append(co * v2[:,:,h] * gates[:,:,h:h+1])  # double gating
        return self.out(torch.cat(hs, -1))


# V4_11: GatedMHConv + cumsum global path (hybrid)
class GatedMHConvPlusCumsum(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.inner = GatedMHConv_Base(d, nh, max_k)
        self.cs_proj = nn.Linear(d, d // 4, bias=False)
        self.cs_gate = nn.Linear(d, d // 4)
        self.merge = nn.Linear(d + d // 4, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        local = self.inner(x)
        gated = self.cs_proj(x) * torch.sigmoid(self.cs_gate(x))
        cs = torch.cumsum(gated, dim=1)
        pos = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, L, 1)
        global_ctx = cs / pos
        return self.merge(torch.cat([local, global_ctx], dim=-1))


# V4_12: Deeper GatedMHConv with 2 conv layers per head
class GatedMHConvDeep(nn.Module):
    def __init__(self, d, nh=8, max_k=33):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        self.v = nn.Linear(d, d, bias=False)
        self.convs1 = nn.ModuleList([nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(self.dh, self.dh, 3, padding=2, groups=self.dh) for _ in ks])
        self.head_gates = nn.Linear(d, nh)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).view(B, L, self.nh, self.dh)
        gates = torch.sigmoid(self.head_gates(x))
        hs = []
        for h in range(self.nh):
            hv = v[:,:,h].transpose(1,2)
            c1 = F.silu(self.convs1[h](hv)[:,:,:L])
            c2 = self.convs2[h](c1)[:,:,:L].transpose(1,2)
            hs.append(F.silu(c2) * gates[:,:,h:h+1])
        return self.out(torch.cat(hs, -1))


# V4_13: GatedMHConv with GELU instead of SiLU
class GatedMHConvGELU(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks])
        self.head_gates = nn.Linear(d, nh)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).view(B, L, self.nh, self.dh)
        gates = torch.sigmoid(self.head_gates(x))
        hs = [F.gelu(self.convs[h](v[:,:,h].transpose(1,2))[:,:,:L].transpose(1,2)) * gates[:,:,h:h+1]
              for h in range(self.nh)]
        return self.out(torch.cat(hs, -1))


# V4_14: GatedMHConv with learnable residual scale per layer
class GatedMHConvScaledRes(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.inner = GatedMHConv_Base(d, nh, max_k)
        # The Block already has rs parameter; this adds per-dim scaling
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x, **kw):
        return self.inner(x) * self.scale


# V4_15: GatedMHConv + TS + VR (exploit V2 lesson: simple > kitchen sink, but try smarter combo)
class GatedMHConvTSVR(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.ts_w = nn.Parameter(torch.ones(d) * 0.5)
        self.nh = nh; self.dh = d // nh
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks])
        self.head_gates = nn.Linear(d, nh)
        self.out = nn.Linear(d, d, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.9))

    def forward(self, x, embed=None, **kw):
        B, L, D = x.shape
        # Token shift
        w = torch.sigmoid(self.ts_w)
        shifted = F.pad(x[:, :-1], (0, 0, 1, 0))
        h = w * x + (1 - w) * shifted
        # Value residual
        if embed is not None:
            a = torch.sigmoid(self.alpha)
            h = a * h + (1 - a) * embed
        v = self.v(h).view(B, L, self.nh, self.dh)
        gates = torch.sigmoid(self.head_gates(h))
        hs = [F.silu(self.convs[i](v[:,:,i].transpose(1,2))[:,:,:L].transpose(1,2)) * gates[:,:,i:i+1]
              for i in range(self.nh)]
        return self.out(torch.cat(hs, -1))


# ════════════════════════════════════════════════════════════════════════
# MODEL & REGISTRATION
# ════════════════════════════════════════════════════════════════════════

NOVEL_V4_CONFIGS = {
    "V4_01_16Heads":       ("GatedMHConv with 16 heads (finer scale selection)", GatedMHConv16H, False),
    "V4_02_SoftmaxGate":   ("GatedMHConv with softmax (competitive) gates", GatedMHConvSoftmax, False),
    "V4_03_ChannelGate":   ("GatedMHConv with per-channel gates", GatedMHConvChannelGate, False),
    "V4_04_TokenShift":    ("GatedMHConv + token shift", GatedMHConvTS, False),
    "V4_05_ValueRes":      ("GatedMHConv + value residual from embedding", GatedMHConvVR, True),
    "V4_06_CrossPost":     ("GatedMHConv + cross-head mixing after gating", GatedMHConvCrossPost, False),
    "V4_07_GLU":           ("GatedMHConv with GLU-style dual gating", GatedMHConvGLU, False),
    "V4_08_HeadNorm":      ("GatedMHConv + RMSNorm per head", GatedMHConvNormed, False),
    "V4_09_OutResidual":   ("GatedMHConv + learned output residual", GatedMHConvResidual, False),
    "V4_10_DualValue":     ("GatedMHConv with dual value projections", GatedMHConvDualV, False),
    "V4_11_PlusCumsum":    ("GatedMHConv + cumsum global context path", GatedMHConvPlusCumsum, False),
    "V4_12_DeepConv":      ("GatedMHConv with 2 conv layers per head", GatedMHConvDeep, False),
    "V4_13_GELU":          ("GatedMHConv with GELU activation", GatedMHConvGELU, False),
    "V4_14_ScaledRes":     ("GatedMHConv with per-dim learned scale", GatedMHConvScaledRes, False),
    "V4_15_TSVR":          ("GatedMHConv + TokenShift + ValueRes (smart combo)", GatedMHConvTSVR, True),
}


class NovelV4Model(FrontierModel):
    def __init__(self, config: FrontierConfig, arch_name: str):
        super().__init__(config)
        self._arch_name = arch_name
        desc, mixer_cls, needs_embed = NOVEL_V4_CONFIGS[arch_name]
        d = config.d_model; n = config.n_layers
        self._needs_embed = needs_embed
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        if needs_embed:
            self.blocks = nn.ModuleList([EmbedBlock(d, config.d_ff, mixer_cls(d)) for _ in range(n)])
        else:
            self.blocks = nn.ModuleList([Block(d, config.d_ff, mixer_cls(d)) for _ in range(n)])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_init)

    def forward(self, x):
        h = self.embed(x); embed_out = h
        for block in self.blocks:
            if self._needs_embed: h = block(h, embed=embed_out)
            else: h = block(h)
        return self.head(self.norm(h))

    @classmethod
    def arch_family(cls): return "novel_noattn_v4"
    def describe(self): return f"{self._arch_name}: {NOVEL_V4_CONFIGS[self._arch_name][0]}"
    def sequence_mixing_complexity(self): return "O(n)"


for name, (desc, _, _) in NOVEL_V4_CONFIGS.items():
    @register_arch(f"{name}LM", "novel_noattn_v4", desc)
    class _M(NovelV4Model):
        _arch_key = name
        def __init__(self, config):
            super().__init__(config, self.__class__._arch_key)
    _M.__name__ = f"{name}LM"
    _M.__qualname__ = f"{name}LM"
    globals()[f"_{name}_cls"] = _M
