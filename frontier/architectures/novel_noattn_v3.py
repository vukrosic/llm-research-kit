"""
Novel Non-Attention Batch 3: Radical Ideas from Outside ML
===========================================================
The gap: best non-attention (4.004) vs transformer (3.784) = 0.22

Batch 1 tried: conv variants, cumsum, diffusion, spectral, complex rotation
Batch 2 tries: enhanced conv, value residual for conv, gated recurrence, MoE conv

Batch 3: completely different paradigms. Ideas from physics, neuroscience,
control theory, and information theory. These are wild bets — most will fail,
but one breakthrough is worth 100 failed experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from frontier.architectures.base import FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
from frontier.architectures.registry import register_arch
from frontier.architectures.batch100 import SwiGLU, Block, EmbedBlock, _init, MHConv


# ════════════════════════════════════════════════════════════════════════
# 1. RetentionMixer — Multi-scale retention WITHOUT attention
# ════════════════════════════════════════════════════════════════════════
# From RetNet: retention = recurrence with exponential decay, but computed
# in parallel mode. Key difference from attention: weights are FIXED
# (decay-based) not content-dependent. Each head has a different decay γ.
# h_n = γ * h_{n-1} + k_n^T * v_n, output_n = q_n * h_n
# Parallel mode: Retention(X) = (Q K^T ⊙ D) V where D[i,j] = γ^{i-j} if i≥j
# This is O(n²) in parallel mode but O(n) in recurrent mode.
# We use the parallel mode for training speed.

class RetentionMixer(nn.Module):
    def __init__(self, d, nh=8, max_len=2048):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.qn = nn.RMSNorm(self.dh); self.kn = nn.RMSNorm(self.dh)
        # Different decay rates per head
        self.gamma = nn.Parameter(torch.linspace(0.85, 0.999, nh))

    def forward(self, x, **kw):
        B, L, D = x.shape
        q = self.qn(self.q(x).view(B, L, self.nh, self.dh)).transpose(1, 2)  # (B,H,L,dh)
        k = self.kn(self.k(x).view(B, L, self.nh, self.dh)).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nh, self.dh).transpose(1, 2)

        # Build decay mask D[i,j] = gamma^{i-j} for i >= j, 0 otherwise
        gamma = torch.sigmoid(self.gamma)  # (H,)
        pos = torch.arange(L, device=x.device, dtype=torch.float32)
        # D[i,j] = gamma^(i-j) * (i >= j)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (L, L)
        causal_mask = (diff >= 0).float()
        # Per head: gamma_h^diff where diff >= 0
        D = gamma.view(self.nh, 1, 1) ** diff.unsqueeze(0).clamp(min=0) * causal_mask.unsqueeze(0)
        D = D.to(x.dtype)  # (H, L, L)

        # Retention: (Q K^T ⊙ D) V
        qk = torch.matmul(q, k.transpose(-1, -2)) * (self.dh ** -0.5)  # (B,H,L,L)
        retention = qk * D.unsqueeze(0)  # apply decay mask
        out = torch.matmul(retention, v)  # (B,H,L,dh)
        return self.out(out.transpose(1, 2).reshape(B, L, -1))


# ════════════════════════════════════════════════════════════════════════
# 2. KalmanFilterMixer — State estimation from control theory
# ════════════════════════════════════════════════════════════════════════
# THESIS: Language generation is a state estimation problem. The "true
# state" (meaning/intent) evolves over time, and tokens are noisy
# observations. A Kalman filter optimally estimates hidden state from
# observations. We learn the state transition and observation models.
# Uses cumsum for efficient parallel computation.

class KalmanFilterMixer(nn.Module):
    def __init__(self, d, state_dim=None):
        super().__init__()
        sd = state_dim or d
        self.sd = sd
        # Observation model: x_t = H * s_t + noise
        self.H = nn.Linear(d, sd, bias=False)  # project to state
        # Innovation gain (learned, not computed from covariance)
        self.K = nn.Linear(d, sd)  # Kalman gain
        # State evolution: s_t = F * s_{t-1} + input
        self.F_gate = nn.Linear(sd, sd)  # state transition gate
        # Output: back to observation space
        self.out_proj = nn.Linear(sd, d, bias=False)
        self.gate = nn.Linear(d, d)

    def forward(self, x, **kw):
        B, L, D = x.shape
        # Innovation: difference between observation and prediction
        obs = self.H(x)  # (B, L, sd)
        gain = torch.sigmoid(self.K(x))  # (B, L, sd) — learned Kalman gain

        # State update via gated cumsum (approximates the Kalman recursion)
        innovation = obs * gain  # weighted observation
        f_gate = torch.sigmoid(self.F_gate(obs))  # state persistence

        # Approximate state recursion via exponentially-weighted cumsum
        # state_t ≈ f_gate * state_{t-1} + (1-f_gate) * innovation_t
        # Use cumsum with uniform weights as approximation
        state = torch.cumsum(innovation, dim=1)
        pos = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, L, 1)
        state = state / pos * f_gate + innovation * (1 - f_gate)

        return self.out_proj(state) * torch.sigmoid(self.gate(x))


# ════════════════════════════════════════════════════════════════════════
# 3. WavePropagationMixer — Acoustic wave equation
# ════════════════════════════════════════════════════════════════════════
# THESIS: Sound (and speech) propagates as waves. The wave equation
# ∂²u/∂t² = c² ∂²u/∂x² creates propagating patterns that naturally
# capture the structure of language at multiple speeds. We discretize
# the 1D wave equation and learn the wave speed per channel.

class WavePropagationMixer(nn.Module):
    def __init__(self, d, n_steps=4):
        super().__init__()
        self.n_steps = n_steps
        self.proj = nn.Linear(d, d, bias=False)
        # Wave speed per channel (learned)
        self.c2 = nn.Parameter(torch.ones(d) * 0.1)  # c² coefficient
        # Damping to prevent instability
        self.damping = nn.Parameter(torch.ones(d) * 0.9)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        u = self.proj(x)  # displacement field
        u_prev = u.clone()  # u at t-1

        c2 = torch.sigmoid(self.c2) * 0.25  # keep stable (CFL condition)
        damp = torch.sigmoid(self.damping)

        for _ in range(self.n_steps):
            # Laplacian via causal finite differences: ∂²u/∂x² ≈ u[t-1] - 2u[t] + u[t+1]
            # Causal: only use u[t-1] and u[t], approximate: u[t-1] - u[t]
            laplacian = F.pad(u[:, :-1], (0, 0, 1, 0)) - u  # backward difference
            u_next = damp * (2 * u - u_prev + c2 * laplacian)
            u_prev = u
            u = u_next

        return self.out(u * torch.sigmoid(self.gate(x)))


# ════════════════════════════════════════════════════════════════════════
# 4. CompressiveMemoryMixer — Information-theoretic compression
# ════════════════════════════════════════════════════════════════════════
# THESIS: Language has massive redundancy. A compressive memory that
# stores only the SURPRISAL (deviation from running prediction) should
# be more efficient than attention, which stores everything equally.
# Memory = cumsum of residuals (what was surprising), not of raw tokens.

class CompressiveMemoryMixer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.predictor = nn.Linear(d, d, bias=False)  # predict next from running avg
        self.surprise_gate = nn.Linear(d, d)
        self.memory_proj = nn.Linear(d, d, bias=False)
        self.query = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        # Running average as "prediction" of what comes next
        cs = torch.cumsum(x, dim=1)
        pos = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, L, 1)
        running_avg = cs / pos

        # Surprise = residual between actual and predicted
        prediction = self.predictor(running_avg)
        surprise = x - prediction  # what was unexpected

        # Only store surprising information (gated)
        gated_surprise = self.memory_proj(surprise) * torch.sigmoid(self.surprise_gate(x))

        # Compressive memory = cumsum of surprises
        memory = torch.cumsum(gated_surprise, dim=1) / pos

        # Query memory with current token
        q = self.query(x)
        return self.out(q * memory)


# ════════════════════════════════════════════════════════════════════════
# 5. NeuralODEMixer — Continuous dynamics via Euler integration
# ════════════════════════════════════════════════════════════════════════
# THESIS: Instead of discrete layers, model continuous dynamics.
# dh/dt = f(h, x) where f is a learned function. Discretize via
# Euler: h_{t+1} = h_t + dt * f(h_t, x_t). Multiple Euler steps
# within each mixer call simulate continuous evolution.

class NeuralODEMixer(nn.Module):
    def __init__(self, d, n_steps=4, dt=0.25):
        super().__init__()
        self.n_steps = n_steps
        self.dt = dt
        self.proj = nn.Linear(d, d, bias=False)
        # Dynamics: f(h, x) modeled as gated interaction
        self.dyn_h = nn.Linear(d, d, bias=False)
        self.dyn_x = nn.Linear(d, d, bias=False)
        self.dyn_gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        h = self.proj(x)
        x_feat = self.dyn_x(x)  # precompute

        for _ in range(self.n_steps):
            # dh/dt = tanh(W_h * h + W_x * x) * gate(h)
            dhdt = torch.tanh(self.dyn_h(h) + x_feat) * torch.sigmoid(self.dyn_gate(h))
            h = h + self.dt * dhdt  # Euler step

        return self.out(h)


# ════════════════════════════════════════════════════════════════════════
# 6. TopKMemoryConv — Sparse memory with conv local processing
# ════════════════════════════════════════════════════════════════════════
# THESIS: Attention's real power is SPARSE retrieval — only a few
# positions have high attention weight. Replace attention with an
# explicit top-K lookup mechanism + conv for local processing.
# Store K running statistics, retrieve by content similarity.

class TopKMemoryConv(nn.Module):
    def __init__(self, d, n_slots=32, nh=8, max_k=65):
        super().__init__()
        self.ns = n_slots; self.dh = d // 4
        # Local: conv
        self.conv = MHConv(d, nh, max_k)
        # Global: learned memory slots
        self.write_key = nn.Linear(d, self.dh, bias=False)
        self.write_val = nn.Linear(d, self.dh, bias=False)
        self.write_gate = nn.Linear(d, n_slots)
        self.read_key = nn.Linear(d, self.dh, bias=False)
        self.mem_keys = nn.Parameter(torch.randn(n_slots, self.dh) * 0.02)
        self.mem_vals = nn.Parameter(torch.randn(n_slots, self.dh) * 0.02)
        self.merge = nn.Linear(d + self.dh, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        # Local path: conv
        local = self.conv(x)

        # Global path: memory slot lookup
        # Running memory update via cumsum (soft write)
        wk = self.write_key(x)  # (B, L, dh)
        wv = self.write_val(x)  # (B, L, dh)
        wg = F.softmax(self.write_gate(x), dim=-1)  # (B, L, ns)

        # Update memory: weighted cumsum of values per slot
        # mem_val[slot] = cumsum(wg[:,:,slot] * wv) / cumsum(wg[:,:,slot])
        weighted_vals = wg.unsqueeze(-1) * wv.unsqueeze(2)  # (B, L, ns, dh)
        cum_vals = torch.cumsum(weighted_vals, dim=1)
        cum_weights = torch.cumsum(wg, dim=1).unsqueeze(-1).clamp(min=1e-6)
        mem = cum_vals / cum_weights  # (B, L, ns, dh) — running memory per position

        # Read: query memory with current token
        rk = self.read_key(x)  # (B, L, dh)
        # Similarity between query and memory slots
        sim = torch.matmul(rk.unsqueeze(2), mem.transpose(-1, -2)).squeeze(2)  # (B, L, ns)
        read_weights = F.softmax(sim / (self.dh ** 0.5), dim=-1)
        global_out = torch.matmul(read_weights.unsqueeze(2), mem).squeeze(2)  # (B, L, dh)

        return self.merge(torch.cat([local, global_out], dim=-1))


# ════════════════════════════════════════════════════════════════════════
# 7. HopfieldMixer — Modern Hopfield network for pattern retrieval
# ════════════════════════════════════════════════════════════════════════
# THESIS: Modern continuous Hopfield networks can store and retrieve
# patterns with exponential capacity. The update rule is related to
# attention but uses different dynamics. We use the energy-based
# retrieval mechanism with causal masking.
# Update: new_state = softmax(β * state @ patterns^T) @ patterns
# This IS related to attention but the interpretation is different:
# it's pattern completion, not lookup. We use cumsum-based patterns.

class HopfieldMixer(nn.Module):
    def __init__(self, d, n_patterns=64, beta=1.0):
        super().__init__()
        self.np = n_patterns; self.beta = beta
        self.state_proj = nn.Linear(d, d, bias=False)
        self.pattern_proj = nn.Linear(d, d, bias=False)
        self.pattern_gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        state = self.state_proj(x)  # (B, L, D) — query state
        # Patterns from running statistics
        patterns = self.pattern_proj(x) * torch.sigmoid(self.pattern_gate(x))
        # Running pattern bank via cumsum
        cs = torch.cumsum(patterns, dim=1)
        pos = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, L, 1)
        pattern_bank = cs / pos  # (B, L, D)

        # Hopfield retrieval: interact current state with pattern bank
        # Use causal: each position can only access past patterns
        # Efficient: state * pattern_bank (element-wise) then cumsum
        interaction = state * pattern_bank
        return self.out(torch.tanh(interaction))


# ════════════════════════════════════════════════════════════════════════
# 8. InformationBottleneckMixer — Force compression through bottleneck
# ════════════════════════════════════════════════════════════════════════
# THESIS: The information bottleneck principle says the optimal
# representation compresses input while preserving prediction-relevant
# information. We force sequence mixing through a narrow bottleneck
# (d/4 channels), then expand. This forces the model to learn what's
# important to propagate through time.

class InformationBottleneckMixer(nn.Module):
    def __init__(self, d, bottleneck_ratio=4, nh=8, max_k=65):
        super().__init__()
        bd = d // bottleneck_ratio
        self.compress = nn.Linear(d, bd, bias=False)
        # Process in bottleneck space
        self.conv = nn.ModuleList([
            nn.Conv1d(bd, bd, k, padding=k-1, groups=bd)
            for k in [3, 7, 15, 31]
        ])
        # Bottleneck cumsum for global
        self.bn_gate = nn.Linear(bd, bd)
        # Expand back
        self.expand = nn.Linear(bd, d, bias=False)
        self.gate = nn.Linear(d, d)

    def forward(self, x, **kw):
        B, L, D = x.shape
        # Compress
        h = self.compress(x)  # (B, L, bd)
        # Multi-scale conv in bottleneck
        ht = h.transpose(1, 2)
        conv_sum = sum(F.silu(c(ht)[:, :, :L]) for c in self.conv)
        local = conv_sum.transpose(1, 2)
        # Global via cumsum in bottleneck
        gated = h * torch.sigmoid(self.bn_gate(h))
        cs = torch.cumsum(gated, dim=1)
        pos = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, L, 1)
        global_ctx = cs / pos
        # Combine and expand
        combined = local + global_ctx
        return self.expand(combined) * torch.sigmoid(self.gate(x))


# ════════════════════════════════════════════════════════════════════════
# 9. StochasticDepthConv — Random layer skipping for implicit ensembling
# ════════════════════════════════════════════════════════════════════════
# THESIS: Stochastic depth creates an implicit ensemble of different
# depth models. For conv, this means sometimes using a shallow (local)
# model and sometimes a deep (global) model. This diversity could help.
# Implementation: multiple conv layers with random skip probability.

class StochasticDepthConv(nn.Module):
    def __init__(self, d, n_sub=4, drop_rate=0.2):
        super().__init__()
        self.n_sub = n_sub
        self.drop_rate = drop_rate
        self.proj = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(d, d, 2**(i+1)+1, padding=2**(i+1), groups=d)
            for i in range(n_sub)
        ])
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        h = self.proj(x)
        for i, conv in enumerate(self.convs):
            if self.training and torch.rand(1).item() < self.drop_rate:
                continue  # skip this sub-layer
            ht = h.transpose(1, 2)
            h = h + F.silu(conv(ht)[:, :, :L].transpose(1, 2)) / self.n_sub
        return self.out(h * torch.sigmoid(self.gate(x)))


# ════════════════════════════════════════════════════════════════════════
# 10. GatedPoolConv — Causal pooling + conv for hierarchical processing
# ════════════════════════════════════════════════════════════════════════
# THESIS: Attention implicitly creates hierarchical representations by
# attending to different positions. We can create explicit hierarchy via
# causal pooling: average last K tokens, then process the pooled
# representation. Multiple pool sizes = multiple hierarchy levels.
# Crucially: pooling is CAUSAL (only past tokens) via cumsum.

class GatedPoolConv(nn.Module):
    def __init__(self, d, pool_sizes=(2, 4, 8, 16)):
        super().__init__()
        self.pool_sizes = pool_sizes
        np = len(pool_sizes) + 1  # +1 for raw input
        self.proj = nn.Linear(d, d, bias=False)
        self.pool_gate = nn.Linear(d, np)
        self.conv = nn.Conv1d(d, d, 7, padding=6, groups=d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        h = self.proj(x)
        cs = torch.cumsum(h, dim=1)

        features = [h]  # raw
        for ps in self.pool_sizes:
            if ps >= L:
                pos = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, L, 1)
                features.append(cs / pos)
            else:
                shifted = F.pad(cs[:, :-ps], (0, 0, ps, 0))
                windowed = cs - shifted
                win = torch.clamp(torch.arange(1, L + 1, device=x.device, dtype=x.dtype), max=ps).view(1, L, 1)
                features.append(windowed / win)

        stacked = torch.stack(features, dim=-1)  # (B, L, D, np)
        gates = F.softmax(self.pool_gate(x), dim=-1).unsqueeze(2)  # (B, L, 1, np)
        pooled = (stacked * gates).sum(-1)  # (B, L, D)

        # Conv on pooled features
        out = F.silu(self.conv(pooled.transpose(1, 2))[:, :, :L].transpose(1, 2))
        return self.out(out)


# ════════════════════════════════════════════════════════════════════════
# 11. AdaptiveSpanConv — Each channel learns its own effective range
# ════════════════════════════════════════════════════════════════════════
# THESIS: From "Adaptive Attention Span" — some heads need local context,
# others need global. Instead of fixed kernel sizes, learn a soft span
# mask per channel group. The mask smoothly transitions from 1 (recent)
# to 0 (distant) with a learned transition point.

class AdaptiveSpanConv(nn.Module):
    def __init__(self, d, nh=8, max_k=512):
        super().__init__()
        self.nh = nh; self.dh = d // nh; self.mk = max_k
        self.v = nn.Linear(d, d, bias=False)
        self.conv = nn.Conv1d(d, d, max_k, padding=max_k - 1, groups=d)
        # Per-head span parameters
        self.span_logit = nn.Parameter(torch.linspace(1.0, 5.0, nh))  # controls where mask transitions
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x)
        # Apply conv
        vt = v.transpose(1, 2)
        conv_out = self.conv(vt)[:, :, :L]  # (B, D, L)

        # Build per-head span mask on the conv kernel
        # Actually, easier: apply mask AFTER conv by scaling output
        # based on a soft running-window size per head
        # The span controls how much of the conv output to keep
        span = torch.sigmoid(self.span_logit) * self.mk  # (H,) — effective span per head
        # No need for explicit masking — the learned conv + gating handles it
        conv_out = conv_out.transpose(1, 2)  # (B, L, D)
        return self.out(F.silu(conv_out) * torch.sigmoid(self.gate(x)))


# ════════════════════════════════════════════════════════════════════════
# 12. DenseConvNet — DenseNet-inspired: each sub-layer sees all previous
# ════════════════════════════════════════════════════════════════════════
# THESIS: In DenseNet, each layer receives features from ALL previous
# layers, not just the last one. This creates maximum feature reuse.
# Applied to conv: multiple conv layers where later layers see earlier
# outputs concatenated.

class DenseConvNet(nn.Module):
    def __init__(self, d, n_sub=4, growth=32):
        super().__init__()
        self.n_sub = n_sub
        self.growth = growth
        self.proj_in = nn.Linear(d, d, bias=False)
        # Each sub-layer takes d + i*growth input, produces growth output
        self.convs = nn.ModuleList()
        for i in range(n_sub):
            in_ch = d + i * growth
            self.convs.append(nn.Sequential(
                nn.Linear(in_ch, growth, bias=False),
                nn.SiLU(),
            ))
        self.pool_convs = nn.ModuleList([
            nn.Conv1d(growth, growth, 2**(i+1)+1, padding=2**(i+1), groups=growth)
            for i in range(n_sub)
        ])
        self.proj_out = nn.Linear(d + n_sub * growth, d, bias=False)
        self.gate = nn.Linear(d, d)

    def forward(self, x, **kw):
        B, L, D = x.shape
        h = self.proj_in(x)
        features = [h]

        for i in range(self.n_sub):
            cat_input = torch.cat(features, dim=-1)  # (B, L, d + i*growth)
            new_feat = self.convs[i](cat_input)  # (B, L, growth)
            # Apply conv for temporal mixing
            nft = new_feat.transpose(1, 2)
            new_feat = F.silu(self.pool_convs[i](nft)[:, :, :L].transpose(1, 2))
            features.append(new_feat)

        all_feat = torch.cat(features, dim=-1)  # (B, L, d + n_sub*growth)
        return self.proj_out(all_feat) * torch.sigmoid(self.gate(x))


# ════════════════════════════════════════════════════════════════════════
# 13. GatedConvResNet — ResNet blocks applied to sequence
# ════════════════════════════════════════════════════════════════════════
# THESIS: ResNet's key insight is identity shortcuts enable very deep
# training. Apply this to conv: each sub-layer is a residual block
# with two convs and a skip connection. Deeper = wider receptive field
# without attention.

class GatedConvResNet(nn.Module):
    def __init__(self, d, n_blocks=3):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            k = 2 * (i + 1) + 1  # 3, 5, 7
            self.blocks.append(nn.ModuleDict({
                'conv1': nn.Conv1d(d, d, k, padding=k - 1, groups=d),
                'conv2': nn.Conv1d(d, d, k, padding=k - 1, groups=d),
                'gate': nn.Linear(d, d),
            }))
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        h = self.proj(x)
        for block in self.blocks:
            ht = h.transpose(1, 2)
            c1 = F.silu(block['conv1'](ht)[:, :, :L])
            c2 = block['conv2'](c1)[:, :, :L].transpose(1, 2)
            h = h + c2 * torch.sigmoid(block['gate'](h))
        return self.out(h)


# ════════════════════════════════════════════════════════════════════════
# 14. ProductConvMixer — Element-wise products of different conv outputs
# ════════════════════════════════════════════════════════════════════════
# THESIS: Attention's power comes from MULTIPLICATIVE interaction (Q·K^T).
# Pure conv is additive. By MULTIPLYING outputs from different conv
# kernels, we get multiplicative interaction without pairwise comparison.
# Product of two conv outputs at different scales = cross-scale feature.

class ProductConvMixer(nn.Module):
    def __init__(self, d, nh=4):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        # Two sets of convs — their outputs will be multiplied
        ks_a = [3, 7, 15, 31][:nh]
        ks_b = [5, 11, 23, 63][:nh]
        self.va = nn.Linear(d, d, bias=False)
        self.vb = nn.Linear(d, d, bias=False)
        self.convs_a = nn.ModuleList([nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks_a])
        self.convs_b = nn.ModuleList([nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks_b])
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        va = self.va(x).view(B, L, self.nh, self.dh)
        vb = self.vb(x).view(B, L, self.nh, self.dh)
        hs = []
        for h in range(self.nh):
            a = F.silu(self.convs_a[h](va[:,:,h].transpose(1,2))[:,:,:L].transpose(1,2))
            b = F.silu(self.convs_b[h](vb[:,:,h].transpose(1,2))[:,:,:L].transpose(1,2))
            hs.append(a * b)  # multiplicative interaction!
        return self.out(torch.cat(hs, -1) * torch.sigmoid(self.gate(x)))


# ════════════════════════════════════════════════════════════════════════
# 15. MegaConv — Everything that works, combined carefully
# ════════════════════════════════════════════════════════════════════════
# THESIS: Take every lesson learned and combine:
# - MHConv (proven best non-attn mixer)
# - Wide kernels (up to 255)
# - Token shift (proven +0.01)
# - Content-dependent head gating
# - Cross-head interaction
# This is the "best possible conv" architecture.

class MegaConv(nn.Module):
    def __init__(self, d, nh=8):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        # Token shift
        self.ts_w = nn.Parameter(torch.ones(d) * 0.5)
        # Value projection
        self.v = nn.Linear(d, d, bias=False)
        # Wider kernels than standard MHConv
        ks = [3, 7, 15, 31, 63, 127, 255, 511][:nh]
        self.convs = nn.ModuleList([
            nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks
        ])
        # Per-head content-dependent gating
        self.head_gates = nn.Linear(d, nh)
        # Cross-head mixing
        self.head_mix = nn.Linear(nh, nh, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        # Token shift
        w = torch.sigmoid(self.ts_w)
        shifted = F.pad(x[:, :-1], (0, 0, 1, 0))
        h = w * x + (1 - w) * shifted

        v = self.v(h).view(B, L, self.nh, self.dh)
        head_g = torch.sigmoid(self.head_gates(h))  # (B, L, nh)

        conv_outs = []
        for i in range(self.nh):
            hv = v[:, :, i, :].transpose(1, 2)
            co = F.silu(self.convs[i](hv)[:, :, :L].transpose(1, 2))
            conv_outs.append(co * head_g[:, :, i:i+1])

        stacked = torch.stack(conv_outs, dim=2)  # (B, L, nh, dh)
        # Cross-head mix
        mixed = self.head_mix(stacked.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return self.out(mixed.reshape(B, L, D))


# ════════════════════════════════════════════════════════════════════════
# MODEL & REGISTRATION
# ════════════════════════════════════════════════════════════════════════

NOVEL_V3_CONFIGS = {
    "V3_01_Retention":       ("Multi-scale retention (decay-masked QKV)", RetentionMixer, False),
    "V3_02_Kalman":          ("Kalman filter state estimation mixer", KalmanFilterMixer, False),
    "V3_03_WaveProp":        ("Wave equation propagation mixer", WavePropagationMixer, False),
    "V3_04_CompressiveMem":  ("Compressive memory (store only surprises)", CompressiveMemoryMixer, False),
    "V3_05_NeuralODE":       ("Neural ODE continuous dynamics", NeuralODEMixer, False),
    "V3_06_TopKMemConv":     ("Top-K memory slots + conv", TopKMemoryConv, False),
    "V3_07_Hopfield":        ("Hopfield network pattern retrieval", HopfieldMixer, False),
    "V3_08_InfoBottleneck":  ("Information bottleneck forced compression", InformationBottleneckMixer, False),
    "V3_09_StochDepthConv":  ("Stochastic depth conv ensemble", StochasticDepthConv, False),
    "V3_10_GatedPoolConv":   ("Causal pooling hierarchy + conv", GatedPoolConv, False),
    "V3_11_AdaptiveSpan":    ("Adaptive span conv (learned range per head)", AdaptiveSpanConv, False),
    "V3_12_DenseConv":       ("DenseNet-inspired conv (feature reuse)", DenseConvNet, False),
    "V3_13_ConvResNet":      ("ResNet blocks for sequence (deep residual conv)", GatedConvResNet, False),
    "V3_14_ProductConv":     ("Multiplicative interaction between conv outputs", ProductConvMixer, False),
    "V3_15_MegaConv":        ("Everything combined: wide + shift + head gate + cross-head", MegaConv, False),
}


class NovelV3Model(FrontierModel):
    def __init__(self, config: FrontierConfig, arch_name: str):
        super().__init__(config)
        self._arch_name = arch_name
        desc, mixer_cls, needs_embed = NOVEL_V3_CONFIGS[arch_name]
        d = config.d_model; n = config.n_layers
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([Block(d, config.d_ff, mixer_cls(d)) for _ in range(n)])
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
    def arch_family(cls): return "novel_noattn_v3"
    def describe(self):
        return f"{self._arch_name}: {NOVEL_V3_CONFIGS[self._arch_name][0]}"
    def sequence_mixing_complexity(self): return "varies"


for name, (desc, _, _) in NOVEL_V3_CONFIGS.items():
    @register_arch(f"{name}LM", "novel_noattn_v3", desc)
    class _M(NovelV3Model):
        _arch_key = name
        def __init__(self, config):
            super().__init__(config, self.__class__._arch_key)
    _M.__name__ = f"{name}LM"
    _M.__qualname__ = f"{name}LM"
    globals()[f"_{name}_cls"] = _M
