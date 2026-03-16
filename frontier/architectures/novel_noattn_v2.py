"""
Novel Non-Attention Batch 2: Closing the Gap
=============================================
Batch 1 best: LearnedDecayConv = 4.004 (gap to transformer: 0.22)
PureConv (MHConv) = 3.920 (gap to transformer: 0.136)
Transformer baseline = 3.784

Strategy:
A. Enhance MHConv — push PureConv's 3.920 closer to 3.784
B. Combine best novel mechanisms with each other
C. New mechanisms focused on content-dependent processing WITHOUT attention
D. Wider kernels, value residual for conv, cross-head interactions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from frontier.architectures.base import FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
from frontier.architectures.registry import register_arch
from frontier.architectures.batch100 import SwiGLU, Block, EmbedBlock, _init, MHConv, TokenShiftMHConv


# ════════════════════════════════════════════════════════════════════════
# A1. WideKernelConv — MHConv with much wider kernels (up to 511)
# ════════════════════════════════════════════════════════════════════════
# THESIS: MHConv caps at k=65. Language has dependencies spanning hundreds
# of tokens (paragraphs, long sentences). Wider kernels capture these.
# The GPU can handle k=511 efficiently via depthwise conv.

class WideKernelConv(nn.Module):
    def __init__(self, d, nh=8):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        ks = [3, 7, 15, 31, 63, 127, 255, 511][:nh]
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks
        ])
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).view(B, L, self.nh, self.dh)
        hs = []
        for h in range(self.nh):
            hv = v[:, :, h, :].transpose(1, 2)
            hs.append(F.silu(self.convs[h](hv)[:, :, :L].transpose(1, 2)))
        return self.out(torch.cat(hs, -1) * torch.sigmoid(self.gate(x)))


# ════════════════════════════════════════════════════════════════════════
# A2. MHConvCrossHead — MHConv + cross-head interaction
# ════════════════════════════════════════════════════════════════════════
# THESIS: Standard MHConv processes heads independently. Cross-head
# interaction lets different kernel sizes share information, creating
# richer multi-scale representations. Like talking heads for conv.

class MHConvCrossHead(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks
        ])
        # Cross-head mixing (small MLP)
        self.head_mix = nn.Linear(nh, nh, bias=False)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).view(B, L, self.nh, self.dh)
        conv_outs = []
        for h in range(self.nh):
            hv = v[:, :, h, :].transpose(1, 2)
            conv_outs.append(F.silu(self.convs[h](hv)[:, :, :L].transpose(1, 2)))
        stacked = torch.stack(conv_outs, dim=2)  # (B, L, nh, dh)
        # Cross-head interaction: mix across heads per position
        # Reshape for head_mix: (B, L, dh, nh) -> mix -> (B, L, dh, nh)
        mixed = self.head_mix(stacked.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        mixed = mixed.reshape(B, L, D)
        return self.out(mixed * torch.sigmoid(self.gate(x)))


# ════════════════════════════════════════════════════════════════════════
# A3. ValueResConv — MHConv with value residual from embedding
# ════════════════════════════════════════════════════════════════════════
# THESIS: Value residual was the biggest single improvement for attention
# (+0.1 val_loss). The same principle should help conv: feed the raw
# embedding into the value projection to create a gradient highway.

class ValueResConv(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.conv = MHConv(d, nh, max_k)
        self.alpha = nn.Parameter(torch.tensor(0.8))

    def forward(self, x, embed=None, **kw):
        if embed is not None:
            a = torch.sigmoid(self.alpha)
            # Mix input with embedding before conv
            x_mixed = a * x + (1 - a) * embed
            return self.conv(x_mixed)
        return self.conv(x)


# ════════════════════════════════════════════════════════════════════════
# A4. GatedMHConv — MHConv with per-head input-dependent gating
# ════════════════════════════════════════════════════════════════════════
# THESIS: Standard MHConv uses a single output gate. Per-HEAD gating
# lets the model dynamically control which kernel sizes matter for each
# token — content-dependent multi-scale processing without attention.

class GatedMHConv(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks
        ])
        # Per-head gates (content-dependent)
        self.head_gates = nn.Linear(d, nh)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).view(B, L, self.nh, self.dh)
        gates = torch.sigmoid(self.head_gates(x))  # (B, L, nh)
        hs = []
        for h in range(self.nh):
            hv = v[:, :, h, :].transpose(1, 2)
            conv_out = F.silu(self.convs[h](hv)[:, :, :L].transpose(1, 2))
            hs.append(conv_out * gates[:, :, h:h+1])
        return self.out(torch.cat(hs, -1))


# ════════════════════════════════════════════════════════════════════════
# B1. DecayConvPlusShift — Combine LearnedDecayConv + TokenShiftPyramid
# ════════════════════════════════════════════════════════════════════════
# THESIS: LearnedDecayConv was #1 novel (4.004), TokenShiftPyramid was #4
# (4.055). Combine both: decay conv for smooth local patterns, shift pyramid
# for sparse global access. They're complementary.

class DecayConvPlusShift(nn.Module):
    def __init__(self, d, n_groups=16, max_kernel=64, n_shifts=8):
        super().__init__()
        # Decay conv path
        self.ng = n_groups; self.gsize = d // n_groups; self.mk = max_kernel
        self.proj = nn.Linear(d, d, bias=False)
        self.amp = nn.Parameter(torch.ones(n_groups) * 0.5)
        self.alpha = nn.Parameter(torch.linspace(0.01, 0.2, n_groups))
        self.omega = nn.Parameter(torch.linspace(0.0, 2.0, n_groups))
        self.phase = nn.Parameter(torch.zeros(n_groups))
        # Shift path
        self.n_shifts = n_shifts
        self.shift_proj = nn.Linear(d, d, bias=False)
        shifts = [2 ** k for k in range(n_shifts)]
        self.register_buffer('shifts', torch.tensor(shifts))
        self.shift_weights = nn.Parameter(torch.randn(n_shifts, d) * 0.01)
        # Merge
        self.merge_gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        # Decay conv
        h = self.proj(x)
        t = torch.arange(self.mk, device=x.device, dtype=torch.float32)
        alpha = F.softplus(self.alpha)
        kernel = self.amp.abs().unsqueeze(1) * torch.exp(-alpha.unsqueeze(1) * t.unsqueeze(0)) * \
                 torch.cos(self.omega.unsqueeze(1) * t.unsqueeze(0) + self.phase.unsqueeze(1))
        kernel = kernel.unsqueeze(1).expand(-1, self.gsize, -1).reshape(D, self.mk)
        kernel = kernel.flip(1).unsqueeze(1)
        ht = h.transpose(1, 2)
        decay_out = F.conv1d(ht, kernel.to(ht.dtype), padding=self.mk - 1, groups=D)[:, :, :L].transpose(1, 2)

        # Shift pyramid
        sp = self.shift_proj(x)
        shift_out = torch.zeros(B, L, D, device=x.device, dtype=x.dtype)
        for i, s in enumerate(self.shifts.tolist()):
            s = int(s)
            if s >= L: continue
            shifted = F.pad(sp[:, :-s], (0, 0, s, 0))
            shift_out = shift_out + shifted * self.shift_weights[i]

        # Merge
        g = torch.sigmoid(self.merge_gate(x))
        return self.out(g * decay_out + (1 - g) * shift_out)


# ════════════════════════════════════════════════════════════════════════
# B2. ConvBankPlusEMA — DynamicConvBank + MultiScaleEMA
# ════════════════════════════════════════════════════════════════════════
# THESIS: DynamicConvBank gives content-dependent local processing.
# MultiScaleEMA gives multi-timescale global context. Together: content-
# dependent local features informed by global context.

class ConvBankPlusEMA(nn.Module):
    def __init__(self, d, kernel_sizes=(3, 7, 15, 31), scales=(4, 16, 64, 256)):
        super().__init__()
        self.nk = len(kernel_sizes)
        self.ns = len(scales)
        self.scales = scales
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(d, d, k, padding=k - 1, groups=d) for k in kernel_sizes
        ])
        self.router = nn.Linear(d, self.nk)
        self.ema_weights = nn.Linear(d, self.ns)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).transpose(1, 2)
        routes = F.softmax(self.router(x), dim=-1)

        # Conv bank (local)
        conv_outs = []
        for i, conv in enumerate(self.convs):
            co = conv(v)[:, :, :L].transpose(1, 2)
            conv_outs.append(co)
        stacked_conv = torch.stack(conv_outs, dim=-1)
        local_out = (stacked_conv * routes.unsqueeze(2)).sum(-1)

        # EMA (global)
        cs = torch.cumsum(x, dim=1)
        ema_feats = []
        for s in self.scales:
            if s >= L:
                pos = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, L, 1)
                ema_feats.append(cs / pos)
            else:
                shifted = F.pad(cs[:, :-s], (0, 0, s, 0))
                windowed = cs - shifted
                win_size = torch.clamp(
                    torch.arange(1, L + 1, device=x.device, dtype=x.dtype), max=s
                ).view(1, L, 1)
                ema_feats.append(windowed / win_size)
        stacked_ema = torch.stack(ema_feats, dim=-1)
        ema_w = F.softmax(self.ema_weights(x), dim=-1)
        global_out = (stacked_ema * ema_w.unsqueeze(2)).sum(-1)

        return self.out(F.silu(local_out + global_out))


# ════════════════════════════════════════════════════════════════════════
# C1. ContentGatedConv — Conv with content-dependent kernel weighting
# ════════════════════════════════════════════════════════════════════════
# THESIS: The key missing ingredient vs attention is CONTENT-DEPENDENCE.
# Standard conv has fixed kernels. Here: a small MLP generates per-position
# weights that scale the conv output channels. This is different from
# DynamicConvBank (which routes between kernels) — here every kernel element
# is content-modulated.

class ContentGatedConv(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks
        ])
        # Content-dependent channel gating
        self.content_gate = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.SiLU(),
            nn.Linear(d // 4, d),
            nn.Sigmoid(),
        )
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).view(B, L, self.nh, self.dh)
        cg = self.content_gate(x)  # (B, L, D) — content-dependent per-channel gate
        hs = []
        for h in range(self.nh):
            hv = v[:, :, h, :].transpose(1, 2)
            hs.append(F.silu(self.convs[h](hv)[:, :, :L].transpose(1, 2)))
        conv_out = torch.cat(hs, -1)
        return self.out(conv_out * cg)


# ════════════════════════════════════════════════════════════════════════
# C2. GatedLinearRecurrence — Minimal gated recurrence via parallel scan
# ════════════════════════════════════════════════════════════════════════
# THESIS: The simplest recurrence: h_t = gate_t * h_{t-1} + (1-gate_t) * x_t
# This is a linear recurrence that CAN be parallelized via prefix sum.
# With input-dependent gates, this provides content-dependent temporal
# aggregation — the key ingredient attention has that conv doesn't.
# Implementation: chunked parallel approach for GPU efficiency.

class GatedLinearRecurrence(nn.Module):
    def __init__(self, d, chunk_size=64):
        super().__init__()
        self.cs = chunk_size
        self.proj = nn.Linear(d, d, bias=False)
        self.gate_proj = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        val = self.proj(x)
        gate = torch.sigmoid(self.gate_proj(x))  # forget gate

        # Parallel within chunks, sequential across chunks
        CS = self.cs
        output = torch.zeros_like(val)
        state = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        for start in range(0, L, CS):
            end = min(start + CS, L)
            g_chunk = gate[:, start:end]  # (B, chunk, D)
            v_chunk = val[:, start:end]   # (B, chunk, D)

            # Within chunk: parallel via cumulative product of gates
            # h[0] = g[0]*state + (1-g[0])*v[0]
            # h[1] = g[1]*h[0] + (1-g[1])*v[1]
            # h[t] = prod(g[s:t])*h[s] + sum_{j=s}^{t} prod(g[j+1:t]) * (1-g[j])*v[j]
            chunk_len = end - start
            h = torch.zeros(B, chunk_len, D, device=x.device, dtype=x.dtype)
            h[:, 0] = g_chunk[:, 0] * state + (1 - g_chunk[:, 0]) * v_chunk[:, 0]
            for t in range(1, chunk_len):
                h[:, t] = g_chunk[:, t] * h[:, t-1] + (1 - g_chunk[:, t]) * v_chunk[:, t]

            output[:, start:end] = h
            state = h[:, -1]

        return self.out(output)


# ════════════════════════════════════════════════════════════════════════
# C3. ConvPlusScan — MHConv for local + gated linear scan for global
# ════════════════════════════════════════════════════════════════════════
# THESIS: MHConv is great for local (3.920). But it can't do content-
# dependent global aggregation. Add a lightweight gated scan for global
# context on top of conv features. This gives conv+global for conv-only
# architectures — analogous to how progressive conv+attn works.

class ConvPlusScan(nn.Module):
    def __init__(self, d, nh=8, max_k=65, chunk_size=64):
        super().__init__()
        self.conv = MHConv(d, nh, max_k)
        # Lightweight gated scan for global context (half channels)
        self.half = d // 2
        self.scan_gate = nn.Linear(d, self.half)
        self.scan_val = nn.Linear(d, self.half, bias=False)
        self.merge = nn.Linear(d + self.half, d, bias=False)
        self.cs = chunk_size

    def forward(self, x, **kw):
        B, L, D = x.shape
        # Local via conv
        local = self.conv(x)

        # Global via gated scan (half-dimensional for speed)
        gate = torch.sigmoid(self.scan_gate(x))  # (B, L, half)
        val = self.scan_val(x)  # (B, L, half)

        # Chunked scan
        CS = self.cs
        scan_out = torch.zeros(B, L, self.half, device=x.device, dtype=x.dtype)
        state = torch.zeros(B, self.half, device=x.device, dtype=x.dtype)
        for start in range(0, L, CS):
            end = min(start + CS, L)
            g = gate[:, start:end]
            v = val[:, start:end]
            chunk_len = end - start
            h = torch.zeros(B, chunk_len, self.half, device=x.device, dtype=x.dtype)
            h[:, 0] = g[:, 0] * state + (1 - g[:, 0]) * v[:, 0]
            for t in range(1, chunk_len):
                h[:, t] = g[:, t] * h[:, t - 1] + (1 - g[:, t]) * v[:, t]
            scan_out[:, start:end] = h
            state = h[:, -1]

        return self.merge(torch.cat([local, scan_out], dim=-1))


# ════════════════════════════════════════════════════════════════════════
# C4. MultiGrainConv — Process at multiple granularities simultaneously
# ════════════════════════════════════════════════════════════════════════
# THESIS: Language operates at multiple grains: characters → tokens →
# phrases → sentences. Process all grains in parallel using strided views
# of the sequence, each at a different "zoom level". Reconstruct at full
# resolution. This is like multi-scale processing but without pooling
# (which leaks future info).

class MultiGrainConv(nn.Module):
    def __init__(self, d, grains=(1, 2, 4, 8)):
        super().__init__()
        self.grains = grains
        ng = len(grains)
        dg = d // ng
        self.dg = dg
        self.proj = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(dg, dg, 7, padding=6, groups=dg) for _ in grains
        ])
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        h = self.proj(x).view(B, L, len(self.grains), self.dg)
        outs = []
        for i, grain in enumerate(self.grains):
            hg = h[:, :, i, :]  # (B, L, dg)
            if grain == 1:
                ht = hg.transpose(1, 2)
                out = self.convs[i](ht)[:, :, :L].transpose(1, 2)
            else:
                # Process every grain-th token (causal: take every grain-th from past)
                # Use strided avg pooling as input
                Lp = (L + grain - 1) // grain * grain
                hg_pad = F.pad(hg, (0, 0, 0, Lp - L))
                # Reshape to (B, L//grain, grain, dg) and take last (causal)
                hg_strided = hg_pad.view(B, Lp // grain, grain, self.dg)[:, :, -1, :]  # (B, L//g, dg)
                ht = hg_strided.transpose(1, 2)
                conv_out = self.convs[i](ht)[:, :, :Lp // grain].transpose(1, 2)
                # Upsample back to L via repeat
                out = conv_out.repeat_interleave(grain, dim=1)[:, :L]
            outs.append(F.silu(out))
        combined = torch.cat(outs, dim=-1)
        return self.out(combined * torch.sigmoid(self.gate(x)))


# ════════════════════════════════════════════════════════════════════════
# C5. ConvMoE — Mixture of Conv Experts
# ════════════════════════════════════════════════════════════════════════
# THESIS: Different tokens need different types of processing. A mixture
# of K conv "experts" (each with different kernel sizes / structures)
# selected per-token gives the model the ability to adapt its processing.
# This is MoE for conv — each expert is a different conv configuration.

class ConvMoE(nn.Module):
    def __init__(self, d, n_experts=4):
        super().__init__()
        self.ne = n_experts
        # Different expert types
        self.experts = nn.ModuleList([
            MHConv(d, nh=8, max_k=65),  # standard multi-scale
            MHConv(d, nh=4, max_k=255),  # fewer heads, wider kernels
            nn.Sequential(  # deep narrow
                nn.Conv1d(d, d, 3, padding=2, groups=d),
                nn.SiLU(),
                nn.Conv1d(d, d, 3, padding=2, groups=d),
            ),
            nn.Sequential(  # single wide kernel
                nn.Linear(d, d, bias=False),
                nn.SiLU(),
            ),
        ])
        self.router = nn.Linear(d, n_experts)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        routes = F.softmax(self.router(x), dim=-1)  # (B, L, ne)
        expert_outs = []
        for i, expert in enumerate(self.experts):
            if isinstance(expert, nn.Sequential) and isinstance(expert[0], nn.Conv1d):
                # Conv-based expert
                ht = x.transpose(1, 2)
                for mod in expert:
                    if isinstance(mod, nn.Conv1d):
                        ht = mod(ht)[:, :, :L]
                    else:
                        ht = mod(ht)
                expert_outs.append(ht.transpose(1, 2))
            elif isinstance(expert, MHConv):
                expert_outs.append(expert(x))
            else:
                expert_outs.append(expert(x))

        stacked = torch.stack(expert_outs, dim=-1)  # (B, L, D, ne)
        mixed = (stacked * routes.unsqueeze(2)).sum(-1)
        return self.out(mixed)


# ════════════════════════════════════════════════════════════════════════
# D1. TSConvWideVR — TokenShift + WideKernel + ValueResidual
# ════════════════════════════════════════════════════════════════════════
# THESIS: Combine the three strongest individual improvements:
# TokenShift (RWKV-style), wider kernels, and value residual.
# This is the "kitchen sink" for conv — but kitchen sink worked for
# attention (CosineAttnVR = best at 3.759).

class TSConvWideVR(nn.Module):
    def __init__(self, d, nh=8):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d) * 0.5)
        self.conv = WideKernelConv(d, nh)
        self.alpha = nn.Parameter(torch.tensor(0.8))

    def forward(self, x, embed=None, **kw):
        w = torch.sigmoid(self.w)
        shifted = F.pad(x[:, :-1], (0, 0, 1, 0))
        h = w * x + (1 - w) * shifted
        if embed is not None:
            a = torch.sigmoid(self.alpha)
            h = a * h + (1 - a) * embed
        return self.conv(h)


# ════════════════════════════════════════════════════════════════════════
# D2. TSMHConvVR — TokenShift + standard MHConv + ValueResidual
# ════════════════════════════════════════════════════════════════════════
# THESIS: Same as D1 but with standard MHConv kernels (not wide).
# Tests whether value residual helps conv as much as it helped attention.

class TSMHConvVR(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d) * 0.5)
        self.conv = MHConv(d, nh, max_k)
        self.alpha = nn.Parameter(torch.tensor(0.8))

    def forward(self, x, embed=None, **kw):
        w = torch.sigmoid(self.w)
        shifted = F.pad(x[:, :-1], (0, 0, 1, 0))
        h = w * x + (1 - w) * shifted
        if embed is not None:
            a = torch.sigmoid(self.alpha)
            h = a * h + (1 - a) * embed
        return self.conv(h)


# ════════════════════════════════════════════════════════════════════════
# D3. DoubleConv — Two MHConv passes with different kernel configs
# ════════════════════════════════════════════════════════════════════════
# THESIS: One conv pass may not extract all useful patterns. Two passes
# with different kernel configurations (one for short patterns, one for
# long) give richer features. Like depth in a ResNet.

class DoubleConv(nn.Module):
    def __init__(self, d, nh=8):
        super().__init__()
        self.conv1 = MHConv(d, nh=nh, max_k=17)  # short-range first
        self.conv2 = MHConv(d, nh=nh, max_k=255)  # long-range second
        self.mix = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        h1 = self.conv1(x)
        return self.mix(self.conv2(h1 + x))  # residual between conv passes


# ════════════════════════════════════════════════════════════════════════
# D4. ConvWithLearnedPositionBias — MHConv + relative position bias
# ════════════════════════════════════════════════════════════════════════
# THESIS: Attention has implicit position awareness through position
# embeddings. Conv has no position awareness — it processes all positions
# identically. Adding a learned position-dependent bias to conv output
# could help it differentiate early vs late positions in a sequence.

class ConvWithPositionBias(nn.Module):
    def __init__(self, d, nh=8, max_k=65, max_len=2048):
        super().__init__()
        self.conv = MHConv(d, nh, max_k)
        # Learned position bias (shared across batch)
        self.pos_bias = nn.Parameter(torch.zeros(max_len, d) * 0.01)
        self.gate = nn.Linear(d, d)

    def forward(self, x, **kw):
        B, L, D = x.shape
        conv_out = self.conv(x)
        bias = self.pos_bias[:L].unsqueeze(0)  # (1, L, D)
        return conv_out + torch.sigmoid(self.gate(x)) * bias


# ════════════════════════════════════════════════════════════════════════
# D5. TriplePathConv — Three parallel conv paths at different scales
# ════════════════════════════════════════════════════════════════════════
# THESIS: Instead of one MHConv with multiple heads, run three completely
# separate conv networks (short, medium, long range) and combine.
# Each path has its own value projection, giving more capacity.

class TriplePathConv(nn.Module):
    def __init__(self, d):
        super().__init__()
        third = d // 3
        self.t1 = third; self.t2 = third; self.t3 = d - 2 * third
        # Short range (k=3,5,7)
        self.short_v = nn.Linear(d, self.t1, bias=False)
        self.short_conv = nn.Conv1d(self.t1, self.t1, 7, padding=6, groups=self.t1)
        # Medium range (k=31)
        self.med_v = nn.Linear(d, self.t2, bias=False)
        self.med_conv = nn.Conv1d(self.t2, self.t2, 31, padding=30, groups=self.t2)
        # Long range (k=127)
        self.long_v = nn.Linear(d, self.t3, bias=False)
        self.long_conv = nn.Conv1d(self.t3, self.t3, 127, padding=126, groups=self.t3)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        s = F.silu(self.short_conv(self.short_v(x).transpose(1, 2))[:, :, :L].transpose(1, 2))
        m = F.silu(self.med_conv(self.med_v(x).transpose(1, 2))[:, :, :L].transpose(1, 2))
        l = F.silu(self.long_conv(self.long_v(x).transpose(1, 2))[:, :, :L].transpose(1, 2))
        combined = torch.cat([s, m, l], dim=-1)
        return self.out(combined * torch.sigmoid(self.gate(x)))


# ════════════════════════════════════════════════════════════════════════
# MODEL CLASS & REGISTRATION
# ════════════════════════════════════════════════════════════════════════

NOVEL_V2_CONFIGS = {
    # A: Enhanced MHConv
    "V2_01_WideKernel":     ("MHConv with kernels up to 511", WideKernelConv, False),
    "V2_02_CrossHead":      ("MHConv + cross-head interaction", MHConvCrossHead, False),
    "V2_03_ValueResConv":   ("MHConv + value residual from embedding", ValueResConv, True),
    "V2_04_GatedMHConv":    ("MHConv + per-head content-dependent gating", GatedMHConv, False),
    # B: Hybrid novel
    "V2_05_DecayShift":     ("LearnedDecayConv + TokenShiftPyramid hybrid", DecayConvPlusShift, False),
    "V2_06_ConvBankEMA":    ("DynamicConvBank + MultiScaleEMA hybrid", ConvBankPlusEMA, False),
    # C: New mechanisms
    "V2_07_ContentGated":   ("Content-dependent channel gating on MHConv", ContentGatedConv, False),
    "V2_08_GatedLinRec":    ("Gated linear recurrence via chunked scan", GatedLinearRecurrence, False),
    "V2_09_ConvPlusScan":   ("MHConv (local) + gated scan (global)", ConvPlusScan, False),
    "V2_10_MultiGrain":     ("Multi-granularity parallel conv processing", MultiGrainConv, False),
    "V2_11_ConvMoE":        ("Mixture of conv experts with per-token routing", ConvMoE, False),
    # D: Kitchen sink combos
    "V2_12_TSConvWideVR":   ("TokenShift + WideKernel + ValueRes (kitchen sink)", TSConvWideVR, True),
    "V2_13_TSMHConvVR":     ("TokenShift + MHConv + ValueRes", TSMHConvVR, True),
    "V2_14_DoubleConv":     ("Two MHConv passes (short→long range)", DoubleConv, False),
    "V2_15_TriplePath":     ("Three parallel conv paths (short/med/long)", TriplePathConv, False),
}


class NovelV2Model(FrontierModel):
    """Batch 2 novel non-attention model."""
    def __init__(self, config: FrontierConfig, arch_name: str):
        super().__init__(config)
        self._arch_name = arch_name
        desc, mixer_cls, needs_embed = NOVEL_V2_CONFIGS[arch_name]
        d = config.d_model
        n = config.n_layers
        self._needs_embed = needs_embed

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)

        if needs_embed:
            self.blocks = nn.ModuleList([
                EmbedBlock(d, config.d_ff, mixer_cls(d)) for _ in range(n)
            ])
        else:
            self.blocks = nn.ModuleList([
                Block(d, config.d_ff, mixer_cls(d)) for _ in range(n)
            ])

        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_init)

    def forward(self, x):
        h = self.embed(x)
        embed_out = h
        for block in self.blocks:
            if self._needs_embed:
                h = block(h, embed=embed_out)
            else:
                h = block(h)
        return self.head(self.norm(h))

    @classmethod
    def arch_family(cls): return "novel_noattn_v2"
    def describe(self):
        desc = NOVEL_V2_CONFIGS[self._arch_name][0]
        return f"{self._arch_name}: {desc}"
    def sequence_mixing_complexity(self): return "O(n)"


# Register all
for name, (desc, _, _) in NOVEL_V2_CONFIGS.items():
    @register_arch(f"{name}LM", "novel_noattn_v2", desc)
    class _M(NovelV2Model):
        _arch_key = name
        def __init__(self, config):
            super().__init__(config, self.__class__._arch_key)
    _M.__name__ = f"{name}LM"
    _M.__qualname__ = f"{name}LM"
    globals()[f"_{name}_cls"] = _M
