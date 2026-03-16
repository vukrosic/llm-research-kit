"""
15 Novel Non-Attention Architectures
=====================================
Every mixer uses ONLY O(n) or O(n log n) sequence mixing.
NO attention: no QKV dot-products, no softmax over positions, no pairwise token comparisons.

Each has a clear thesis for why it could outperform attention.
All must be fully parallelizable (no sequential loops over L).

Target: ~100-110M params at d=512, n_layers=22, d_ff=2048
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from frontier.architectures.base import FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
from frontier.architectures.registry import register_arch
from frontier.architectures.batch100 import SwiGLU, Block, _init


# ════════════════════════════════════════════════════════════════════════
# MIXER 1: ExpDecayMultiHead
# ════════════════════════════════════════════════════════════════════════
# THESIS: Language has strong recency bias. Most useful context is local.
# Multi-head exponential decay with different timescales (fast heads for
# local syntax, slow heads for global topic) naturally captures this
# hierarchy. Attention wastes capacity giving uniform access to all positions.
#
# MECHANISM: Each head has a learned decay rate λ_h. The output is an
# exponentially-weighted running sum: y_t = Σ_{s≤t} λ^{t-s} · gate(x_s) · proj(x_s)
# Implemented as causal conv with exponential kernel per head group.

class ExpDecayMultiHead(nn.Module):
    def __init__(self, d, n_groups=4):
        super().__init__()
        self.ng = n_groups
        self.dg = d // n_groups
        self.proj = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d)
        # Different kernel sizes per group (exponentially spaced)
        kernel_sizes = [7, 31, 127, 511][:n_groups]
        self.convs = nn.ModuleList()
        for i, ks in enumerate(kernel_sizes):
            conv = nn.Conv1d(self.dg, self.dg, ks, padding=ks - 1, groups=self.dg)
            # Initialize with exponential decay
            with torch.no_grad():
                decay = 0.7 + 0.29 * (i / max(n_groups - 1, 1))  # 0.7 to 0.99
                t = torch.arange(ks, dtype=torch.float32)
                kernel = decay ** t
                kernel = kernel / kernel.sum()  # normalize
                conv.weight.data[:, 0, :] = kernel.flip(0).unsqueeze(0).expand(self.dg, -1)
            self.convs.append(conv)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        h = self.proj(x) * torch.sigmoid(self.gate(x))
        h = h.view(B, L, self.ng, self.dg)
        outs = []
        for g in range(self.ng):
            hg = h[:, :, g, :].transpose(1, 2)  # (B, dg, L)
            og = self.convs[g](hg)[:, :, :L]  # causal trim
            outs.append(og.transpose(1, 2))  # (B, L, dg)
        return self.out(torch.cat(outs, dim=-1))


# ════════════════════════════════════════════════════════════════════════
# MIXER 2: GatedConvTower
# ════════════════════════════════════════════════════════════════════════
# THESIS: Deep composition of simple local operations can match global
# attention. Each sub-layer is a gated k=3 causal conv. Stacking 4 gives
# receptive field 9 per block; 22 blocks give 198. With residual connections,
# information flows globally. Simpler = faster = more tokens in 5 min.
#
# MECHANISM: 4 stacked gated causal convolutions within each mixer call.
# gate(conv(x)) * silu(conv(x)) — GLU-style gating.

class GatedConvTower(nn.Module):
    def __init__(self, d, n_sub=3, k=3):
        super().__init__()
        self.n_sub = n_sub
        self.convs = nn.ModuleList([nn.Conv1d(d, d, k, padding=k - 1, groups=d) for _ in range(n_sub)])
        # Single shared gate + projection
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        h = x
        for conv in self.convs:
            ht = h.transpose(1, 2)
            conv_out = conv(ht)[:, :, :h.shape[1]].transpose(1, 2)
            h = h + F.silu(conv_out) * 0.3  # fixed scaling for stability
        return self.out(h * torch.sigmoid(self.gate(x)))


# ════════════════════════════════════════════════════════════════════════
# MIXER 3: MultiScaleEMA
# ════════════════════════════════════════════════════════════════════════
# THESIS: Language has multi-scale structure (morpheme → word → phrase →
# sentence → paragraph). Attention processes all scales in one flat operation.
# Multi-scale EMA explicitly models different timescales, letting each
# specialize. This is the core idea behind MEGA (Multi-scale EMA with
# Gated Attention) — but we remove the attention entirely.
#
# MECHANISM: K groups, each with a different moving-average window.
# EMA approximated via cumsum: running_mean_k = cumsum(x) / position,
# windowed by using cumsum differences. Cross-scale gating combines them.

class MultiScaleEMA(nn.Module):
    def __init__(self, d, scales=(4, 16, 64, 256)):
        super().__init__()
        self.scales = scales
        self.ns = len(scales)
        self.proj = nn.Linear(d, d, bias=False)
        self.scale_weights = nn.Linear(d, self.ns)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        h = self.proj(x)  # (B, L, D)
        cs = torch.cumsum(h, dim=1)  # (B, L, D)

        features = []
        for s in self.scales:
            # Windowed running mean of last s tokens
            if s >= L:
                pos = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, L, 1)
                features.append(cs / pos)
            else:
                # cs[t] - cs[t-s] gives sum of last s tokens
                shifted = F.pad(cs[:, :-s], (0, 0, s, 0))  # shift right by s
                windowed = (cs - shifted)
                # Normalize by window size (clip for positions < s)
                win_size = torch.clamp(
                    torch.arange(1, L + 1, device=x.device, dtype=x.dtype),
                    max=s
                ).view(1, L, 1)
                features.append(windowed / win_size)

        # Stack and weight by input-dependent gate
        stacked = torch.stack(features, dim=-1)  # (B, L, D, ns)
        weights = F.softmax(self.scale_weights(x), dim=-1)  # (B, L, ns)
        out = (stacked * weights.unsqueeze(2)).sum(-1)  # (B, L, D)
        return self.out(out)


# ════════════════════════════════════════════════════════════════════════
# MIXER 4: DynamicConvBank
# ════════════════════════════════════════════════════════════════════════
# THESIS: Attention's key power is content-dependent processing — the same
# position gets processed differently depending on what's there. Convolutions
# are static. DynamicConvBank bridges this gap: a bank of K conv kernels
# with per-token soft routing. Each token selects its own mixture of
# convolution patterns based on its content. Content-dependent + O(n).
#
# MECHANISM: K causal convolutions with different kernel sizes. Per-token
# routing scores via a small projector. Apply all convs, combine with
# routing weights.

class DynamicConvBank(nn.Module):
    def __init__(self, d, kernel_sizes=(3, 7, 15, 31)):
        super().__init__()
        self.nk = len(kernel_sizes)
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(d, d, k, padding=k - 1, groups=d) for k in kernel_sizes
        ])
        self.router = nn.Linear(d, self.nk)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).transpose(1, 2)  # (B, D, L)
        routes = F.softmax(self.router(x), dim=-1)  # (B, L, nk)

        conv_outs = []
        for i, conv in enumerate(self.convs):
            co = conv(v)[:, :, :L].transpose(1, 2)  # (B, L, D)
            conv_outs.append(co)

        stacked = torch.stack(conv_outs, dim=-1)  # (B, L, D, nk)
        mixed = (stacked * routes.unsqueeze(2)).sum(-1)  # (B, L, D)
        return self.out(F.silu(mixed))


# ════════════════════════════════════════════════════════════════════════
# MIXER 5: PolynomialMixer
# ════════════════════════════════════════════════════════════════════════
# THESIS: Attention's power comes from the QUADRATIC interaction (Q·K^T).
# This is what makes it content-dependent. We can get quadratic interactions
# WITHOUT pairwise token comparison: project into two spaces, multiply
# element-wise (quadratic in input), then aggregate via cumsum. This
# captures second-order statistics over time — like a running covariance.
#
# MECHANISM: Two projections p1(x), p2(x). Element-wise product gives
# quadratic features. Gated cumsum aggregates. Output = query * state.

class PolynomialMixer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.p1 = nn.Linear(d, d, bias=False)
        self.p2 = nn.Linear(d, d, bias=False)
        self.decay_gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        # Quadratic interaction
        quad = self.p1(x) * self.p2(x)  # (B, L, D) — 2nd order in x
        # Gated accumulation
        gate = torch.sigmoid(self.decay_gate(x))
        state = torch.cumsum(quad * gate, dim=1)
        # Normalize
        pos = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, L, 1)
        state = state / pos
        return self.out(x * state)


# ════════════════════════════════════════════════════════════════════════
# MIXER 6: TokenShiftPyramid
# ════════════════════════════════════════════════════════════════════════
# THESIS: The simplest possible "attention" is just looking at specific
# past positions. A pyramid of shifts at powers-of-2 distances (1, 2, 4,
# 8, ..., 1024) gives log(n)-resolution coverage of the entire past.
# This is like a binary indexed tree — constant factor overhead for global
# access. RWKV's token shift works well; this extends it to multiple scales.
#
# MECHANISM: Shift input by 2^k for k=0,...,10. Each shift has a learned
# weight. All shifts are combined via gating.

class TokenShiftPyramid(nn.Module):
    def __init__(self, d, max_shifts=10):
        super().__init__()
        self.n_shifts = max_shifts
        shifts = [2 ** k for k in range(max_shifts)]  # 1, 2, 4, ..., 512
        self.register_buffer('shifts', torch.tensor(shifts))
        # Per-shift, per-channel weights
        self.shift_weights = nn.Parameter(torch.randn(max_shifts, d) * 0.02)
        self.gate = nn.Linear(d, d)
        self.mix = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        # Create shifted versions
        proj = self.mix(x)
        accumulated = torch.zeros(B, L, D, device=x.device, dtype=x.dtype)
        for i, s in enumerate(self.shifts.tolist()):
            s = int(s)
            if s >= L:
                continue
            shifted = F.pad(proj[:, :-s], (0, 0, s, 0))
            accumulated = accumulated + shifted * self.shift_weights[i].unsqueeze(0).unsqueeze(0)
        return self.out(accumulated * torch.sigmoid(self.gate(x)))


# ════════════════════════════════════════════════════════════════════════
# MIXER 7: CausalDiffusion
# ════════════════════════════════════════════════════════════════════════
# THESIS: Information in language flows like heat diffusion — nearby tokens
# share the most context. Attention is a "teleportation" mechanism (any
# position to any position). But diffusion with multiple steps creates
# smooth gradients of information that may better capture the continuous
# nature of meaning flow. Each step is a causal conv (k=3), and we stack
# K steps with learned diffusion rates + gating.
#
# MECHANISM: K iterations of: h += α * causal_conv(h, k=3) + β * x
# This is an unrolled causal heat equation with source term.

class CausalDiffusion(nn.Module):
    def __init__(self, d, n_steps=4, k=3):
        super().__init__()
        self.n_steps = n_steps
        self.convs = nn.ModuleList([nn.Conv1d(d, d, k, padding=k - 1, groups=d) for _ in range(n_steps)])
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor(0.3)) for _ in range(n_steps)])
        self.source_gates = nn.ModuleList([nn.Linear(d, d) for _ in range(n_steps)])
        self.proj = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        source = self.proj(x)
        h = source.clone()
        for i in range(self.n_steps):
            ht = h.transpose(1, 2)
            diff = self.convs[i](ht)[:, :, :L].transpose(1, 2)
            alpha = torch.sigmoid(self.alphas[i])
            h = h + alpha * (diff - h) + torch.sigmoid(self.source_gates[i](x)) * source * 0.1
        return self.out(h)


# ════════════════════════════════════════════════════════════════════════
# MIXER 8: ComplexRotator
# ════════════════════════════════════════════════════════════════════════
# THESIS: Language has periodic structure — repeating patterns, rhythmic
# constructions, rhymes, and self-referential structures. Complex rotations
# naturally encode periodicity. By treating channel pairs as complex numbers
# and applying learned rotations at each position, we create oscillatory
# states that capture periodic patterns. This is fundamentally different
# from attention's similarity-based matching.
#
# MECHANISM: Pair channels as (real, imag). Each pair has learned rotation
# angle θ and decay r. State evolves as z_t = r·e^{iθ} · z_{t-1} + x_t.
# Parallelized via cumsum in polar form.

class ComplexRotator(nn.Module):
    def __init__(self, d, n_heads=8):
        super().__init__()
        self.nh = n_heads
        self.dh = d // n_heads // 2  # pairs
        assert d % (n_heads * 2) == 0
        self.proj = nn.Linear(d, d, bias=False)
        # Learned rotation angles and decay rates per head
        self.theta = nn.Parameter(torch.linspace(0.01, 0.5, n_heads))  # rotation speed
        self.decay_logit = nn.Parameter(torch.linspace(0.5, 3.0, n_heads))  # decay rate
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        h = self.proj(x) * torch.sigmoid(self.gate(x))
        # Split into real and imaginary parts
        h = h.view(B, L, self.nh, 2, self.dh)
        h_real, h_imag = h[:, :, :, 0], h[:, :, :, 1]  # (B, L, nh, dh)

        # Build rotation + decay weights via cumsum
        decay = torch.sigmoid(self.decay_logit)  # (nh,) in (0, 1)
        theta = self.theta  # (nh,)

        t = torch.arange(L, device=x.device, dtype=torch.float32)  # (L,)
        # cos(t*θ) and sin(t*θ) for rotation
        cos_t = torch.cos(t.unsqueeze(1) * theta.unsqueeze(0))  # (L, nh)
        sin_t = torch.sin(t.unsqueeze(1) * theta.unsqueeze(0))  # (L, nh)
        # decay^t for amplitude
        decay_t = decay.unsqueeze(0) ** t.unsqueeze(1)  # (L, nh)

        # For causal processing: convolve with damped oscillator kernel
        # kernel[t] = decay^t * cos(t*theta) + i * decay^t * sin(t*theta)
        # Apply to real/imag parts separately:
        # out_real[t] = sum_{s<=t} decay^{t-s} * (cos((t-s)*θ)*in_real[s] - sin((t-s)*θ)*in_imag[s])
        # This requires relative position (t-s), which is a convolution!

        # Build conv kernels for each head
        max_k = min(L, 256)  # truncate kernel
        kt = torch.arange(max_k, device=x.device, dtype=torch.float32)
        k_decay = decay.unsqueeze(0) ** kt.unsqueeze(1)  # (max_k, nh)
        k_cos = torch.cos(kt.unsqueeze(1) * theta.unsqueeze(0)) * k_decay  # (max_k, nh)
        k_sin = torch.sin(kt.unsqueeze(1) * theta.unsqueeze(0)) * k_decay  # (max_k, nh)

        # Apply convolution per head
        out_parts = []
        for head in range(self.nh):
            # Kernels for this head
            kc = k_cos[:, head].flip(0).view(1, 1, max_k).to(h_real.dtype)
            ks = k_sin[:, head].flip(0).view(1, 1, max_k).to(h_imag.dtype)

            hr = h_real[:, :, head, :].transpose(1, 2)  # (B, dh, L)
            hi = h_imag[:, :, head, :].transpose(1, 2)

            # out_real = conv(real, cos_kernel) - conv(imag, sin_kernel)
            # out_imag = conv(real, sin_kernel) + conv(imag, cos_kernel)
            or_ = F.conv1d(hr, kc.expand(self.dh, -1, -1), padding=max_k - 1, groups=self.dh)[:, :, :L]
            oi = F.conv1d(hi, ks.expand(self.dh, -1, -1), padding=max_k - 1, groups=self.dh)[:, :, :L]
            out_real_h = (or_ - oi).transpose(1, 2)  # (B, L, dh)

            or2 = F.conv1d(hr, ks.expand(self.dh, -1, -1), padding=max_k - 1, groups=self.dh)[:, :, :L]
            oi2 = F.conv1d(hi, kc.expand(self.dh, -1, -1), padding=max_k - 1, groups=self.dh)[:, :, :L]
            out_imag_h = (or2 + oi2).transpose(1, 2)

            out_parts.append(out_real_h)
            out_parts.append(out_imag_h)

        out = torch.cat(out_parts, dim=-1)  # (B, L, D)
        return self.out(out)


# ════════════════════════════════════════════════════════════════════════
# MIXER 9: ConvStateHybrid
# ════════════════════════════════════════════════════════════════════════
# THESIS: Local and global context serve different roles in language.
# Local = syntax, agreement, word formation. Global = topic, entity tracking.
# Attention conflates both into one mechanism. We separate them:
# (a) Short causal conv for local patterns
# (b) Gated cumsum for global state accumulation
# Then combine with content-dependent gating.
#
# MECHANISM: Parallel paths with different receptive fields, merged by gate.

class ConvStateHybrid(nn.Module):
    def __init__(self, d, local_k=7):
        super().__init__()
        # Local path: short conv
        self.local_conv = nn.Conv1d(d, d, local_k, padding=local_k - 1, groups=d)
        # Global path: gated cumsum
        self.global_proj = nn.Linear(d, d, bias=False)
        self.global_gate = nn.Linear(d, d)
        # Merge
        self.merge_gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        # Local path
        local_out = F.silu(self.local_conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2))
        # Global path
        gated = self.global_proj(x) * torch.sigmoid(self.global_gate(x))
        cs = torch.cumsum(gated, dim=1)
        pos = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, L, 1)
        global_out = cs / pos
        # Merge with content-dependent gating
        g = torch.sigmoid(self.merge_gate(x))
        return self.out(g * local_out + (1 - g) * global_out)


# ════════════════════════════════════════════════════════════════════════
# MIXER 10: CumsumHierarchy
# ════════════════════════════════════════════════════════════════════════
# THESIS: Different levels of cumsum capture different statistics:
# - Level 0: x itself (instantaneous)
# - Level 1: cumsum(x)/t = running mean (1st moment)
# - Level 2: cumsum(cumsum(x))/(t*(t+1)/2) = smoothed mean (accumulated trend)
# - Level 3: triple cumsum = very smooth average
# Each level provides a different temporal "filter" — from sharp (level 0)
# to very smooth (level 3). This multi-resolution statistical view gives
# the model access to temporal features that attention cannot efficiently
# compute.
#
# MECHANISM: Stack of cumsums, each normalized. Combined via learned gating.

class CumsumHierarchy(nn.Module):
    def __init__(self, d, n_levels=4):
        super().__init__()
        self.nl = n_levels
        self.proj = nn.Linear(d, d, bias=False)
        self.level_weights = nn.Linear(d, n_levels)
        self.out = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d)

    def forward(self, x, **kw):
        B, L, D = x.shape
        h = self.proj(x)
        pos = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, L, 1)

        levels = [h]  # Level 0: raw
        cs = h
        for lev in range(1, self.nl):
            cs = torch.cumsum(cs, dim=1)
            # Normalize: k-th cumsum of constant 1 = binomial(t+k-1, k)
            # Approximate: divide by t^k / k!
            norm = (pos ** lev) / math.factorial(lev)
            levels.append(cs / norm.clamp(min=1.0))

        stacked = torch.stack(levels, dim=-1)  # (B, L, D, nl)
        weights = F.softmax(self.level_weights(x), dim=-1)  # (B, L, nl)
        out = (stacked * weights.unsqueeze(2)).sum(-1)
        return self.out(out * torch.sigmoid(self.gate(x)))


# ════════════════════════════════════════════════════════════════════════
# MIXER 11: SpectralConv
# ════════════════════════════════════════════════════════════════════════
# THESIS: Some patterns in language are better described in frequency domain
# — periodic structures (lists, enumerations), rhythmic patterns, repeated
# phrases. A conv kernel parameterized as sum of K sinusoids is more
# parameter-efficient for capturing long-range periodic patterns than
# arbitrary kernels. This is essentially a learnable spectral filter.
#
# MECHANISM: Construct causal conv kernel as sum of K damped sinusoids:
# kernel[t] = Σ_k A_k · exp(-α_k·t) · cos(ω_k·t + φ_k)
# Apply as depthwise causal conv.

class SpectralConv(nn.Module):
    def __init__(self, d, n_components=8, max_kernel=128):
        super().__init__()
        self.nc = n_components
        self.mk = max_kernel
        self.d = d
        self.proj = nn.Linear(d, d, bias=False)
        # Learnable spectral parameters per channel group
        n_groups = d // 8  # group channels for efficiency
        self.n_groups = n_groups
        self.gsize = 8
        self.amplitudes = nn.Parameter(torch.randn(n_groups, n_components) * 0.1)
        self.frequencies = nn.Parameter(torch.linspace(0.01, 1.0, n_components).unsqueeze(0).expand(n_groups, -1))
        self.phases = nn.Parameter(torch.zeros(n_groups, n_components))
        self.decays = nn.Parameter(torch.ones(n_groups, n_components) * 2.0)  # in logit space
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        h = self.proj(x)
        t = torch.arange(self.mk, device=x.device, dtype=torch.float32)

        # Build kernels from spectral components
        decay = torch.sigmoid(self.decays) * 0.1  # small decay rates
        amp = self.amplitudes
        freq = F.softplus(self.frequencies)
        phase = self.phases

        # kernel[g, t] = sum_k amp[g,k] * exp(-decay[g,k]*t) * cos(freq[g,k]*t + phase[g,k])
        # (n_groups, n_components, 1) * (1, 1, mk)
        exp_decay = torch.exp(-decay.unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0))
        cos_comp = torch.cos(freq.unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0) + phase.unsqueeze(-1))
        kernel = (amp.unsqueeze(-1) * exp_decay * cos_comp).sum(1)  # (n_groups, mk)

        # Expand to per-channel
        kernel = kernel.unsqueeze(1).expand(-1, self.gsize, -1).reshape(D, self.mk)
        kernel = kernel.flip(1).unsqueeze(1)  # (D, 1, mk) — causal kernel

        # Apply depthwise conv
        ht = h.transpose(1, 2)  # (B, D, L)
        out = F.conv1d(ht, kernel.to(ht.dtype), padding=self.mk - 1, groups=D)[:, :, :L]
        out = out.transpose(1, 2)
        return self.out(out * torch.sigmoid(self.gate(x)))


# ════════════════════════════════════════════════════════════════════════
# MIXER 12: ReactionDiffusion
# ════════════════════════════════════════════════════════════════════════
# THESIS: Reaction-diffusion systems (Turing patterns) create structured
# spatial patterns from uniform inputs. In language, "spatial" = sequential.
# An activator-inhibitor system with causal diffusion could learn to create
# topic/sentence boundaries, emphasis patterns, and other linguistic
# structures. This is fundamentally different from attention's lookup model.
#
# MECHANISM: Split channels into activator (A) and inhibitor (I).
# Diffusion via causal conv (k=5). Reaction: A' = A*σ(W_A·[A,I]), I' = I*σ(W_I·[A,I]).
# 3 reaction-diffusion steps per mixer call.

class ReactionDiffusion(nn.Module):
    def __init__(self, d, n_steps=3):
        super().__init__()
        half = d // 2
        self.half = half
        self.n_steps = n_steps
        self.proj = nn.Linear(d, d, bias=False)
        # Shared diffusion convs and reaction weights across steps
        self.diff_a = nn.Conv1d(half, half, 5, padding=4, groups=half)
        self.diff_i = nn.Conv1d(half, half, 5, padding=4, groups=half)
        self.react_gate = nn.Linear(d, d)  # shared reaction gate
        self.out = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d)

    def forward(self, x, **kw):
        B, L, D = x.shape
        h = self.proj(x)
        a, i = h[:, :, :self.half], h[:, :, self.half:]

        for _ in range(self.n_steps):
            da = self.diff_a(a.transpose(1, 2))[:, :, :L].transpose(1, 2)
            di = self.diff_i(i.transpose(1, 2))[:, :, :L].transpose(1, 2)
            a_new = a + 0.1 * (da - a)
            i_new = i + 0.1 * (di - i)
            combined = torch.cat([a_new, i_new], dim=-1)
            reaction = torch.sigmoid(self.react_gate(combined))
            r_a, r_i = reaction[:, :, :self.half], reaction[:, :, self.half:]
            a = a_new * r_a
            i = i_new * r_i

        out = torch.cat([a, i], dim=-1)
        return self.out(out * torch.sigmoid(self.gate(x)))


# ════════════════════════════════════════════════════════════════════════
# MIXER 13: RecurrentChannelMix
# ════════════════════════════════════════════════════════════════════════
# THESIS: Attention performs TWO operations simultaneously: temporal
# aggregation (weighted sum over positions) and feature transformation
# (value projection). Separating these should allow each to specialize.
# Temporal: gated cumsum aggregates past. Feature: MLP transforms channels
# using context-dependent weights derived from running statistics.
#
# MECHANISM: (1) Gated cumsum for temporal aggregation, (2) Channel mixing
# via small MLP whose weights depend on running mean (position-dependent).

class RecurrentChannelMix(nn.Module):
    def __init__(self, d, bottleneck=128):
        super().__init__()
        # Temporal aggregation path
        self.t_proj = nn.Linear(d, d, bias=False)
        self.t_gate = nn.Linear(d, d)
        # Channel mixing: context-dependent transformation
        self.c_down = nn.Linear(d, bottleneck, bias=False)
        self.c_up = nn.Linear(bottleneck, d, bias=False)
        # Context: from running stats
        self.ctx_proj = nn.Linear(d, bottleneck)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        # Temporal: gated cumsum
        gated = self.t_proj(x) * torch.sigmoid(self.t_gate(x))
        cs = torch.cumsum(gated, dim=1)
        pos = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, L, 1)
        temporal = cs / pos  # running mean

        # Channel mixing with context modulation
        ctx = torch.sigmoid(self.ctx_proj(temporal))  # (B, L, bottleneck)
        down = self.c_down(x)  # (B, L, bottleneck)
        modulated = down * ctx  # context-dependent channel interaction
        up = self.c_up(F.silu(modulated))
        return self.out(temporal + up)


# ════════════════════════════════════════════════════════════════════════
# MIXER 14: LearnedDecayConv
# ════════════════════════════════════════════════════════════════════════
# THESIS: The damped oscillator is the universal impulse response of any
# 2nd-order linear system: y(t) = A·exp(-α·t)·cos(ω·t + φ). Natural
# language is produced by 2nd-order physical systems (vocal tract resonance).
# By parameterizing conv kernels as damped oscillators, we provide a
# physics-motivated inductive bias that matches the structure of speech
# and language better than arbitrary conv weights or uniform attention.
#
# MECHANISM: Per-channel-group damped oscillator kernels. Learn A, α, ω, φ.
# Apply as depthwise causal convolution. Different groups = different resonances.

class LearnedDecayConv(nn.Module):
    def __init__(self, d, n_groups=16, max_kernel=64):
        super().__init__()
        self.ng = n_groups
        self.gsize = d // n_groups
        self.mk = max_kernel
        self.proj = nn.Linear(d, d, bias=False)
        # Per-group parameters: amplitude, decay, frequency, phase
        self.amp = nn.Parameter(torch.ones(n_groups) * 0.5)
        self.alpha = nn.Parameter(torch.linspace(0.01, 0.2, n_groups))  # decay rate
        self.omega = nn.Parameter(torch.linspace(0.0, 2.0, n_groups))  # frequency
        self.phase = nn.Parameter(torch.zeros(n_groups))
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        h = self.proj(x)
        t = torch.arange(self.mk, device=x.device, dtype=torch.float32)

        # Build per-group kernels
        amp = self.amp.abs()
        alpha = F.softplus(self.alpha)
        omega = self.omega
        phase = self.phase

        kernel = amp.unsqueeze(1) * torch.exp(-alpha.unsqueeze(1) * t.unsqueeze(0)) * \
                 torch.cos(omega.unsqueeze(1) * t.unsqueeze(0) + phase.unsqueeze(1))
        # (ng, mk) → expand to per-channel
        kernel = kernel.unsqueeze(1).expand(-1, self.gsize, -1).reshape(D, self.mk)
        kernel = kernel.flip(1).unsqueeze(1)  # (D, 1, mk) causal

        ht = h.transpose(1, 2)
        out = F.conv1d(ht, kernel.to(ht.dtype), padding=self.mk - 1, groups=D)[:, :, :L]
        out = out.transpose(1, 2)
        return self.out(out * torch.sigmoid(self.gate(x)))


# ════════════════════════════════════════════════════════════════════════
# MIXER 15: HyperConvMixer
# ════════════════════════════════════════════════════════════════════════
# THESIS: Static convolutions cannot adapt to context. Attention adapts
# but at O(n²) cost. HyperConvMixer bridges the gap: a hypernetwork
# generates convolution kernel weights conditioned on running statistics
# (cumsum-based running mean). The conv kernels literally change based on
# what the model has seen so far. Content-dependent convolution at O(n).
#
# MECHANISM: Running mean (via cumsum) → small MLP → conv kernel weights.
# Apply generated weights as depthwise causal conv. Different at every
# position because running mean evolves.

class HyperConvMixer(nn.Module):
    def __init__(self, d, k=7, n_groups=16):
        super().__init__()
        self.k = k
        self.ng = n_groups
        self.gsize = d // n_groups
        self.proj = nn.Linear(d, d, bias=False)
        # Hypernetwork: running_stats → conv kernel weights
        # Input: d features → output: n_groups * k kernel values
        self.hyper = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.SiLU(),
            nn.Linear(d // 4, n_groups * k),
        )
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        h = self.proj(x)

        # Running statistics (via cumsum)
        cs = torch.cumsum(x, dim=1)
        pos = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, L, 1)
        running_mean = cs / pos  # (B, L, D)

        # Generate conv kernels from running stats
        kernels = self.hyper(running_mean)  # (B, L, ng * k)
        kernels = kernels.view(B, L, self.ng, self.k)
        kernels = F.softmax(kernels, dim=-1)  # normalize per group

        # Apply position-dependent conv (manually, since kernels vary per position)
        # For efficiency, pad h and use gather
        h_padded = F.pad(h, (0, 0, self.k - 1, 0))  # (B, L+k-1, D)
        h_padded = h_padded.view(B, L + self.k - 1, self.ng, self.gsize)

        output = torch.zeros(B, L, self.ng, self.gsize, device=x.device, dtype=x.dtype)
        for j in range(self.k):
            # h_padded[:, j:j+L] is the input shifted by (k-1-j)
            shifted = h_padded[:, j:j + L]  # (B, L, ng, gsize)
            w = kernels[:, :, :, self.k - 1 - j:self.k - j]  # (B, L, ng, 1)
            output = output + shifted * w

        output = output.reshape(B, L, D)
        return self.out(output * torch.sigmoid(self.gate(x)))


# ════════════════════════════════════════════════════════════════════════
# MODEL CLASS & REGISTRATION
# ════════════════════════════════════════════════════════════════════════

NOVEL_CONFIGS = {
    "N01_ExpDecayMH":      ("Multi-head exponential decay (multi-timescale leaky integrators)", ExpDecayMultiHead),
    "N02_GatedConvTower":  ("Deep gated conv tower (4 stacked k=3 convs per layer)", GatedConvTower),
    "N03_MultiScaleEMA":   ("Multi-scale EMA with cross-scale gating", MultiScaleEMA),
    "N04_DynConvBank":     ("Dynamic conv bank with per-token routing", DynamicConvBank),
    "N05_PolyMixer":       ("Polynomial (quadratic) mixer with cumsum aggregation", PolynomialMixer),
    "N06_ShiftPyramid":    ("Token shift pyramid (log-n scale coverage)", TokenShiftPyramid),
    "N07_CausalDiffusion": ("Causal diffusion (unrolled heat equation)", CausalDiffusion),
    "N08_ComplexRotator":  ("Complex-valued oscillatory states (damped rotation)", ComplexRotator),
    "N09_ConvStateHybrid": ("Conv (local) + cumsum (global) hybrid paths", ConvStateHybrid),
    "N10_CumsumHierarchy": ("Hierarchical cumsum (multi-order temporal statistics)", CumsumHierarchy),
    "N11_SpectralConv":    ("Spectral conv (sum-of-sinusoids kernel)", SpectralConv),
    "N12_ReactDiffusion":  ("Reaction-diffusion system (activator-inhibitor)", ReactionDiffusion),
    "N13_RecChannelMix":   ("Separated temporal aggregation + context-dependent channel mixing", RecurrentChannelMix),
    "N14_LearnedDecayConv":("Damped oscillator kernels (physics-motivated conv)", LearnedDecayConv),
    "N15_HyperConv":       ("Hypernetwork-generated conv kernels (context-adaptive)", HyperConvMixer),
}


class NovelNoAttnModel(FrontierModel):
    """Model using one of the 15 novel non-attention mixers."""
    def __init__(self, config: FrontierConfig, arch_name: str):
        super().__init__(config)
        self._arch_name = arch_name
        desc, mixer_cls = NOVEL_CONFIGS[arch_name]
        d = config.d_model
        n = config.n_layers

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            Block(d, config.d_ff, mixer_cls(d)) for _ in range(n)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_init)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))

    @classmethod
    def arch_family(cls):
        return "novel_noattn"

    def describe(self):
        desc = NOVEL_CONFIGS[self._arch_name][0]
        return f"{self._arch_name}: {desc}"

    def sequence_mixing_complexity(self):
        return "O(n)"


# Register all 15
for name, (desc, _) in NOVEL_CONFIGS.items():
    @register_arch(f"{name}LM", "novel_noattn", desc)
    class _M(NovelNoAttnModel):
        _arch_key = name
        def __init__(self, config):
            super().__init__(config, self.__class__._arch_key)
    _M.__name__ = f"{name}LM"
    _M.__qualname__ = f"{name}LM"
    globals()[f"_{name}_cls"] = _M
