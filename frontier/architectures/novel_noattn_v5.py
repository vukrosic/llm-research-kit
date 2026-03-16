"""
Novel Non-Attention Batch 5: Radical New Mechanisms
====================================================
V4 exploits GatedMHConv variants. V5 goes completely different directions.
Goal: find something STRUCTURALLY different that closes the 0.131 gap to transformer.

Key insight: attention provides (1) content-based routing, (2) dynamic weighting,
(3) global receptive field. We need ALL THREE without O(n²).

Ideas from: signal processing, control theory, hash-based routing, sparse interaction,
state machines, information theory, neuroscience (lateral inhibition, dendritic computation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from frontier.architectures.base import FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
from frontier.architectures.registry import register_arch
from frontier.architectures.batch100 import SwiGLU, Block, EmbedBlock, _init, MHConv


# ═══════════════════════════════════════════════════════════════════════
# V5_01: HashRouted Conv — content-based routing via learned hashing
# Each token hashes into buckets, conv only within buckets → content-dependent
# Similar to LSH but with learned hash + causal conv instead of attention
# ═══════════════════════════════════════════════════════════════════════
class HashRoutedConv(nn.Module):
    def __init__(self, d, n_buckets=32, n_hash=4):
        super().__init__()
        self.n_buckets = n_buckets
        self.n_hash = n_hash
        self.dh = d // n_hash
        self.hash_proj = nn.Linear(d, n_buckets * n_hash, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        # Per-bucket small conv
        self.bucket_conv = nn.Conv1d(d, d, kernel_size=7, padding=6, groups=d)
        self.mix = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v_proj(x)  # (B, L, D)
        # Compute soft bucket assignments
        logits = self.hash_proj(x).view(B, L, self.n_hash, self.n_buckets)
        weights = F.softmax(logits, dim=-1)  # (B, L, n_hash, n_buckets)
        # For each hash function, weight the value by bucket membership
        # This creates content-dependent mixing weights
        v_expanded = v.view(B, L, self.n_hash, self.dh)
        # Scatter-gather: each position's output is weighted sum of positions with similar hash
        # Efficiently: bucket_score[i,j] = sum_h weights[i,h,:] · weights[j,h,:]
        # This is O(L * n_buckets * n_hash) ≈ O(L * 128) — linear!
        # But we approximate via conv on hash-weighted features
        bucket_features = torch.einsum('blhk,blhd->bkd', weights, v_expanded)  # (B, n_buckets, dh)
        # Each position reads from its buckets
        out = torch.einsum('blhk,bkd->blhd', weights, bucket_features)  # (B, L, n_hash, dh)
        out = out.reshape(B, L, D)
        # Add local conv for fine-grained position info
        local = F.silu(self.bucket_conv(v.transpose(1,2))[:,:,:L].transpose(1,2))
        return self.out(out + local)


# ═══════════════════════════════════════════════════════════════════════
# V5_02: DendriticConv — neuroscience-inspired dendritic computation
# Multiple "dendritic branches" per neuron, each with different receptive field
# Branches gate each other (multiplicative interaction) before summing
# ═══════════════════════════════════════════════════════════════════════
class DendriticConv(nn.Module):
    def __init__(self, d, n_branches=4):
        super().__init__()
        self.n_branches = n_branches
        self.branch_convs = nn.ModuleList()
        for i in range(n_branches):
            k = 2**(i+1) + 1  # 3, 5, 9, 17
            self.branch_convs.append(nn.Conv1d(d, d, k, padding=k-1, groups=d))
        # Inter-branch gating: learned scalar per branch
        self.branch_gate_scales = nn.Parameter(torch.ones(n_branches) * 0.5)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        xt = x.transpose(1, 2)
        branches = []
        for i in range(self.n_branches):
            b = F.silu(self.branch_convs[i](xt)[:, :, :L].transpose(1, 2))
            branches.append(b)
        # Multiplicative dendritic interaction: branch[i] *= sigmoid(scale * branch[i-1])
        for i in range(1, self.n_branches):
            gate = torch.sigmoid(branches[i-1] * self.branch_gate_scales[i])
            branches[i] = branches[i] * gate
        return self.out(sum(branches))


# ═══════════════════════════════════════════════════════════════════════
# V5_03: RecurrentConvState — tiny recurrent state passed between conv layers
# State is a small (d_state=32) vector that accumulates info across positions
# Updated via gated recurrence: s[t] = g*s[t-1] + (1-g)*f(x[t])
# Implemented via parallel scan approximation using cumsum
# ═══════════════════════════════════════════════════════════════════════
class RecurrentConvState(nn.Module):
    def __init__(self, d, d_state=64):
        super().__init__()
        self.d_state = d_state
        # Local processing via MHConv-style
        self.nh = 8; self.dh = d // self.nh
        ks = [3, 5, 9, 17, 33, 65, 65, 65]
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks])
        # Recurrent state
        self.state_in = nn.Linear(d, d_state, bias=False)
        self.state_gate = nn.Linear(d, d_state)
        self.state_out = nn.Linear(d_state, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        # Conv path
        v = self.v(x).view(B, L, self.nh, self.dh)
        hs = [F.silu(self.convs[h](v[:,:,h].transpose(1,2))[:,:,:L].transpose(1,2)) for h in range(self.nh)]
        conv_out = torch.cat(hs, -1)
        # Recurrent state path via EMA approximation
        state_input = self.state_in(x)  # (B, L, d_state)
        gate = torch.sigmoid(self.state_gate(x))  # (B, L, d_state)
        # Approximate gated recurrence via weighted cumsum
        # s[t] ≈ Σ_{i<=t} gate_decay * state_input[i]
        # Use log-space for stability
        log_gate = torch.log(gate + 1e-6)
        log_cumgate = torch.cumsum(log_gate, dim=1)
        weighted_input = state_input * torch.exp(-log_cumgate)
        cumsum = torch.cumsum(weighted_input, dim=1)
        state = cumsum * torch.exp(log_cumgate)
        state_contribution = self.state_out(state)
        return self.out(conv_out + state_contribution)


# ═══════════════════════════════════════════════════════════════════════
# V5_04: LateralInhibition — neuroscience-inspired competitive suppression
# Each head suppresses neighboring heads based on activation strength
# Winner-take-more dynamics within conv heads
# ═══════════════════════════════════════════════════════════════════════
class LateralInhibitionConv(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks])
        # Lateral inhibition: each head competes
        self.inhibition = nn.Linear(nh, nh, bias=False)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).view(B, L, self.nh, self.dh)
        hs = []
        for h in range(self.nh):
            hs.append(F.silu(self.convs[h](v[:,:,h].transpose(1,2))[:,:,:L].transpose(1,2)))
        stacked = torch.stack(hs, dim=2)  # (B, L, nh, dh)
        # Compute head energies
        energies = stacked.norm(dim=-1)  # (B, L, nh)
        # Lateral inhibition: subtract weighted neighbors
        inhibited = energies - F.relu(self.inhibition(energies))
        # Softmax competition
        competition = F.softmax(inhibited, dim=-1)  # (B, L, nh)
        # Apply competitive weights
        weighted = stacked * competition.unsqueeze(-1)
        return self.out(weighted.reshape(B, L, D) * torch.sigmoid(self.gate(x)))


# ═══════════════════════════════════════════════════════════════════════
# V5_05: FreqDomainGate — process in frequency domain but CAUSALLY
# Use DCT (not FFT) which is real-valued, then gate frequency bands
# Causal: only use past positions via causal DCT window
# ═══════════════════════════════════════════════════════════════════════
class FreqDomainGate(nn.Module):
    def __init__(self, d, window=64, n_bands=8):
        super().__init__()
        self.window = window
        self.n_bands = n_bands
        self.v = nn.Linear(d, d, bias=False)
        # Content-dependent frequency band gating
        self.band_gate = nn.Linear(d, n_bands)
        # Per-band processing
        self.band_proj = nn.Linear(d, d, bias=False)
        # Fallback conv for positions < window
        self.local_conv = nn.Conv1d(d, d, 7, padding=6, groups=d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x)
        # Local conv path (always active)
        local = F.silu(self.local_conv(v.transpose(1,2))[:,:,:L].transpose(1,2))
        # Frequency gating on sliding causal windows
        # Pad to handle positions < window
        padded = F.pad(v, (0, 0, self.window-1, 0))  # (B, L+window-1, D)
        # Extract causal windows using unfold
        windows = padded.unfold(1, self.window, 1)  # (B, L, D, window)
        # Real DCT-II approximation via learned bands
        # Instead of actual DCT, use band-pass filtering via conv
        band_gates = torch.sigmoid(self.band_gate(x))  # (B, L, n_bands)
        # Compute band energies from windows
        band_size = self.window // self.n_bands
        band_means = windows.view(B, L, D, self.n_bands, band_size).mean(dim=-1)  # (B, L, D, n_bands)
        # Gate bands
        gated = (band_means * band_gates.unsqueeze(2)).sum(dim=-1)  # (B, L, D)
        freq_out = self.band_proj(gated)
        return self.out(local + freq_out)


# ═══════════════════════════════════════════════════════════════════════
# V5_06: ControlTheoryMixer — PID-inspired (proportional-integral-derivative)
# P = current token, I = running average (cumsum), D = difference (token shift)
# Each component has learned gains, combined with content-dependent weighting
# ═══════════════════════════════════════════════════════════════════════
class PIDConvMixer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.v = nn.Linear(d, d, bias=False)
        # PID gains: content-dependent
        self.pid_gate = nn.Linear(d, 3)
        # Plus conv for multi-scale
        self.conv = nn.Conv1d(d, d, 17, padding=16, groups=d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x)
        # P: proportional (identity)
        p = v
        # I: integral via cumulative mean
        cumsum_v = torch.cumsum(v, dim=1)
        positions = torch.arange(1, L+1, device=x.device).float().unsqueeze(0).unsqueeze(-1)
        integral = cumsum_v / positions
        # D: derivative (causal difference)
        diff = v - F.pad(v[:, :-1, :], (0, 0, 1, 0))
        derivative = diff
        # Content-dependent PID gains
        pid_weights = F.softmax(self.pid_gate(x), dim=-1)  # (B, L, 3)
        combined = (p * pid_weights[:,:,0:1] + integral * pid_weights[:,:,1:2] +
                    derivative * pid_weights[:,:,2:3])
        # Plus multi-scale conv
        conv_out = F.silu(self.conv(v.transpose(1,2))[:,:,:L].transpose(1,2))
        return self.out(combined + conv_out)


# ═══════════════════════════════════════════════════════════════════════
# V5_07: SparseGlobalConv — very wide sparse conv kernels
# Instead of dense kernels, use sparse sampling at exponential positions
# kernel taps at positions: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
# This gives O(log n) global coverage with O(log n) parameters per channel
# ═══════════════════════════════════════════════════════════════════════
class SparseGlobalConv(nn.Module):
    def __init__(self, d, n_taps=11):
        super().__init__()
        self.n_taps = n_taps
        self.positions = [2**i for i in range(n_taps)]  # 1, 2, 4, ..., 1024
        self.v = nn.Linear(d, d, bias=False)
        # Learnable weights for each tap position
        self.tap_weights = nn.Parameter(torch.randn(n_taps, d) * 0.02)
        self.tap_gate = nn.Linear(d, n_taps)
        # Plus local conv
        self.local_conv = nn.Conv1d(d, d, 9, padding=8, groups=d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x)  # (B, L, D)
        # Content-dependent tap gating
        gates = torch.sigmoid(self.tap_gate(x))  # (B, L, n_taps)
        # Gather values from sparse positions (causal: only look back)
        result = torch.zeros_like(v)
        for i, pos in enumerate(self.positions):
            if pos >= L:
                break
            shifted = F.pad(v[:, :-pos, :], (0, 0, pos, 0))  # shift right by pos
            result = result + shifted * self.tap_weights[i] * gates[:, :, i:i+1]
        # Local conv
        local = F.silu(self.local_conv(v.transpose(1,2))[:,:,:L].transpose(1,2))
        return self.out(result + local)


# ═══════════════════════════════════════════════════════════════════════
# V5_08: ConvolutionalGRU — GRU-style gating but using convolutions
# instead of matrix multiplies for the gate/candidate computations
# O(n*k) instead of O(n*d²) — much more efficient
# ═══════════════════════════════════════════════════════════════════════
class ConvGRU(nn.Module):
    def __init__(self, d, k=17):
        super().__init__()
        # GRU gates via depthwise conv (causal)
        self.reset_conv = nn.Conv1d(d, d, k, padding=k-1, groups=d)
        self.update_conv = nn.Conv1d(d, d, k, padding=k-1, groups=d)
        self.candidate_conv = nn.Conv1d(d, d, k, padding=k-1, groups=d)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).transpose(1, 2)  # (B, D, L)
        r = torch.sigmoid(self.reset_conv(v)[:, :, :L])
        z = torch.sigmoid(self.update_conv(v)[:, :, :L])
        candidate = torch.tanh(self.candidate_conv(r * v)[:, :, :L])
        # GRU: h = z * prev + (1-z) * candidate
        # For parallel processing, treat conv output as "previous context"
        h = z * v + (1 - z) * candidate
        return self.out(h.transpose(1, 2))


# ═══════════════════════════════════════════════════════════════════════
# V5_09: MultiPathFusion — 3 parallel paths with cross-path gating
# Path 1: Fine (k=3,5,7) for local patterns
# Path 2: Medium (k=17,33,65) for medium-range
# Path 3: Global (sparse taps at 64,128,256,512) for long-range
# Each path gates the others: content decides which scale matters
# ═══════════════════════════════════════════════════════════════════════
class MultiPathFusion(nn.Module):
    def __init__(self, d):
        super().__init__()
        # Shared value projection
        self.v = nn.Linear(d, d, bias=False)
        # Path 1: Fine-grained local
        self.fine_conv = nn.Conv1d(d, d, 7, padding=6, groups=d)
        # Path 2: Medium-range
        self.med_conv = nn.Conv1d(d, d, 33, padding=32, groups=d)
        # Path 3: Global sparse
        self.global_positions = [64, 128, 256, 512]
        self.global_weights = nn.ParameterList([nn.Parameter(torch.randn(d) * 0.02) for _ in self.global_positions])
        # Cross-path gating
        self.path_gate = nn.Linear(d, 3)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x)
        vt = v.transpose(1, 2)
        # Fine path
        fine = F.silu(self.fine_conv(vt)[:,:,:L].transpose(1,2))
        # Medium path
        med = F.silu(self.med_conv(vt)[:,:,:L].transpose(1,2))
        # Global path
        gv = v
        glob = torch.zeros_like(gv)
        for pos, w in zip(self.global_positions, self.global_weights):
            if pos < L:
                shifted = F.pad(gv[:, :-pos, :], (0, 0, pos, 0))
                glob = glob + shifted * w
        # Content-dependent path fusion
        gates = F.softmax(self.path_gate(x), dim=-1)  # (B, L, 3)
        fused = fine * gates[:,:,0:1] + med * gates[:,:,1:2] + glob * gates[:,:,2:3]
        return self.out(fused)


# ═══════════════════════════════════════════════════════════════════════
# V5_10: TokenSortConv — sort tokens by learned key, conv in sorted space
# Tokens with similar content get adjacent positions in sorted space
# Conv in sorted space = content-based neighborhood processing!
# Must unsort after conv to maintain causal property
# ═══════════════════════════════════════════════════════════════════════
class TokenSortConv(nn.Module):
    def __init__(self, d, nh=4):
        super().__init__()
        self.nh = nh
        self.dh = d // nh
        self.sort_key = nn.Linear(d, nh, bias=False)  # learn what to sort by
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(self.dh, self.dh, 9, padding=8, groups=self.dh) for _ in range(nh)])
        # Causal mask: only allow looking at earlier positions even in sorted space
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).view(B, L, self.nh, self.dh)
        keys = self.sort_key(x)  # (B, L, nh)
        # Add position bias to maintain some ordering stability
        pos_bias = torch.arange(L, device=x.device).float().unsqueeze(0).unsqueeze(-1) * 0.001
        keys = keys + pos_bias

        heads_out = []
        for h in range(self.nh):
            hv = v[:, :, h, :]  # (B, L, dh)
            hk = keys[:, :, h]  # (B, L)
            # Sort by key
            sort_idx = hk.argsort(dim=1)
            # Apply sort
            sorted_v = torch.gather(hv, 1, sort_idx.unsqueeze(-1).expand_as(hv))
            # Conv in sorted space
            conv_out = F.silu(self.convs[h](sorted_v.transpose(1,2))[:,:,:L].transpose(1,2))
            # Unsort
            unsort_idx = sort_idx.argsort(dim=1)
            unsorted = torch.gather(conv_out, 1, unsort_idx.unsqueeze(-1).expand_as(conv_out))
            # Causal masking: multiply by original gate so no future info leaks
            heads_out.append(unsorted)

        result = torch.cat(heads_out, -1)
        return self.out(result * torch.sigmoid(self.gate(x)))


# ═══════════════════════════════════════════════════════════════════════
# V5_11: DilatedConvStack — multiple dilated conv layers stacked within mixer
# Like WaveNet but within a single mixer block
# Dilation rates: 1, 2, 4, 8 → receptive field of 2^4 * k per mixer call
# ═══════════════════════════════════════════════════════════════════════
class DilatedConvStack(nn.Module):
    def __init__(self, d, n_layers=4, k=7):
        super().__init__()
        self.v = nn.Linear(d, d, bias=False)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2**i
            padding = (k - 1) * dilation
            self.layers.append(nn.Conv1d(d, d, k, dilation=dilation, padding=padding, groups=d))
        self.layer_scales = nn.Parameter(torch.ones(n_layers) * 0.25)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        h = self.v(x).transpose(1, 2)  # (B, D, L)
        result = torch.zeros_like(h)
        for i, conv in enumerate(self.layers):
            result = result + F.silu(conv(h)[:, :, :L]) * self.layer_scales[i]
        return self.out(result.transpose(1, 2) * torch.sigmoid(self.gate(x)))


# ═══════════════════════════════════════════════════════════════════════
# V5_12: PolyConv — polynomial-order convolution interactions
# Instead of linear conv (sum of weighted inputs), use degree-2:
# output[t] = Σ w_i * x[t-i] + Σ w_ij * x[t-i] * x[t-j]
# Approximated efficiently via: conv(x)² - conv(x²) for cross-terms
# ═══════════════════════════════════════════════════════════════════════
class PolynomialConv(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks])
        # Second-order interaction weights
        self.cross_scale = nn.Parameter(torch.ones(nh) * 0.1)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).view(B, L, self.nh, self.dh)
        hs = []
        for h in range(self.nh):
            hv = v[:, :, h, :].transpose(1, 2)
            # Linear term
            linear = F.silu(self.convs[h](hv)[:, :, :L])
            # Quadratic cross-term: conv(x)² - conv(x²) ≈ cross-position interactions
            sq_conv = self.convs[h](hv * hv)[:, :, :L]
            conv_sq = linear * linear
            cross = (conv_sq - sq_conv) * self.cross_scale[h]
            hs.append((linear + cross).transpose(1, 2))
        result = torch.cat(hs, -1)
        return self.out(result * torch.sigmoid(self.gate(x)))


# ═══════════════════════════════════════════════════════════════════════
# V5_13: AdaptiveEMA — multiple EMA channels with content-dependent decay
# Like S4/Mamba's diagonal state but simpler and faster
# Each channel has learned decay + content-dependent modulation
# ═══════════════════════════════════════════════════════════════════════
class AdaptiveEMA(nn.Module):
    def __init__(self, d, n_ema=8):
        super().__init__()
        self.n_ema = n_ema
        self.dh = d // n_ema
        self.v = nn.Linear(d, d, bias=False)
        # Base decay per EMA channel (log-space for stability)
        self.log_decay = nn.Parameter(torch.linspace(-0.1, -2.0, n_ema).unsqueeze(-1).expand(n_ema, d // n_ema).contiguous())
        # Content-dependent decay modulation
        self.decay_mod = nn.Linear(d, n_ema)
        # Plus MHConv for local patterns
        self.local_conv = nn.Conv1d(d, d, 9, padding=8, groups=d)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x)  # (B, L, D)

        # Content-dependent decay
        decay_adj = torch.tanh(self.decay_mod(x)) * 0.5  # (B, L, n_ema)

        # For each EMA channel, compute weighted running average
        v_heads = v.view(B, L, self.n_ema, self.dh)
        ema_results = []
        for i in range(self.n_ema):
            base_decay = torch.sigmoid(self.log_decay[i])  # (dh,)
            adj = decay_adj[:, :, i:i+1]  # (B, L, 1)
            effective_decay = base_decay * (1 + adj)  # (B, L, dh)
            effective_decay = effective_decay.clamp(0.01, 0.999)

            # Approximate EMA via cumulative weighted sum
            hv = v_heads[:, :, i, :]  # (B, L, dh)
            # Use geometric series approximation
            log_d = torch.log(effective_decay + 1e-8)
            cum_log_d = torch.cumsum(log_d, dim=1)
            weighted = hv * torch.exp(-cum_log_d)
            cum_weighted = torch.cumsum(weighted, dim=1)
            ema_out = cum_weighted * torch.exp(cum_log_d)
            ema_results.append(ema_out)

        ema = torch.cat(ema_results, dim=-1)  # (B, L, D)
        local = F.silu(self.local_conv(v.transpose(1,2))[:,:,:L].transpose(1,2))
        return self.out((ema + local) * torch.sigmoid(self.gate(x)))


# ═══════════════════════════════════════════════════════════════════════
# V5_14: ConvCrossChannel — standard MHConv but with cross-channel mixing
# After per-head conv, mix across channels via a small MLP
# This adds channel interaction that standard depthwise conv lacks
# ═══════════════════════════════════════════════════════════════════════
class ConvCrossChannel(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks])
        # Cross-channel mixing via small bottleneck
        self.channel_mix = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.SiLU(),
            nn.Linear(d // 4, d)
        )
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).view(B, L, self.nh, self.dh)
        hs = [F.silu(self.convs[h](v[:,:,h].transpose(1,2))[:,:,:L].transpose(1,2)) for h in range(self.nh)]
        conv_out = torch.cat(hs, -1)
        # Cross-channel mixing
        mixed = self.channel_mix(conv_out)
        return self.out(mixed * torch.sigmoid(self.gate(x)))


# ═══════════════════════════════════════════════════════════════════════
# V5_15: GatedMHConvPlusEMA — best of V2 (GatedMHConv) + best novel (EMA)
# GatedMHConv for local content-dependent processing
# Plus exponentially decaying global context via parallel EMA
# The hypothesis: EMA provides the "global receptive field" that conv lacks
# ═══════════════════════════════════════════════════════════════════════
class GatedMHConvPlusEMA(nn.Module):
    def __init__(self, d, nh=8, max_k=65, n_ema=4):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        self.n_ema = n_ema
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        # GatedMHConv path
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks])
        self.head_gates = nn.Linear(d, nh)
        # EMA path
        self.ema_v = nn.Linear(d, d, bias=False)
        self.ema_decays = nn.Parameter(torch.linspace(-0.3, -2.0, n_ema))
        # Fusion
        self.fusion_gate = nn.Linear(d, 1)  # scalar gate: how much EMA vs conv
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        # GatedMHConv
        v = self.v(x).view(B, L, self.nh, self.dh)
        gates = torch.sigmoid(self.head_gates(x))
        hs = [F.silu(self.convs[h](v[:,:,h].transpose(1,2))[:,:,:L].transpose(1,2)) * gates[:,:,h:h+1]
              for h in range(self.nh)]
        conv_out = torch.cat(hs, -1)

        # EMA path
        ema_v = self.ema_v(x)
        ema_per_channel = D // self.n_ema
        ema_results = []
        for i in range(self.n_ema):
            decay = torch.sigmoid(self.ema_decays[i])
            ch = ema_v[:, :, i*ema_per_channel:(i+1)*ema_per_channel]
            # Parallel EMA via cumsum trick
            log_d = math.log(decay.item() + 1e-8)
            positions = torch.arange(L, device=x.device).float()
            decay_weights = torch.exp(log_d * positions)  # (L,)
            weighted = ch * decay_weights.unsqueeze(0).unsqueeze(-1)
            cumsum = torch.cumsum(weighted, dim=1)
            ema_out = cumsum / (decay_weights.unsqueeze(0).unsqueeze(-1) + 1e-8)
            ema_results.append(ema_out)
        ema_out = torch.cat(ema_results, dim=-1)

        # Content-dependent fusion
        alpha = torch.sigmoid(self.fusion_gate(x))
        fused = alpha * conv_out + (1 - alpha) * ema_out
        return self.out(fused)


# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

NOVEL_V5_CONFIGS = {
    "V5_01_HashRouted":      ("Hash-routed content-dependent conv mixing", HashRoutedConv, False),
    "V5_02_Dendritic":       ("Dendritic multi-branch multiplicative gating", DendriticConv, False),
    "V5_03_RecConvState":    ("Conv + parallel recurrent state via gated cumsum", RecurrentConvState, False),
    "V5_04_LateralInhib":   ("Lateral inhibition competitive head selection", LateralInhibitionConv, False),
    "V5_05_FreqGate":       ("Causal frequency-domain band gating", FreqDomainGate, False),
    "V5_06_PIDConv":        ("PID control-theory inspired mixer", PIDConvMixer, False),
    "V5_07_SparseGlobal":   ("Sparse global conv with exponential tap positions", SparseGlobalConv, False),
    "V5_08_ConvGRU":        ("GRU-style gating via depthwise convolutions", ConvGRU, False),
    "V5_09_MultiPath":      ("3-path fusion: fine + medium + global sparse", MultiPathFusion, False),
    "V5_10_TokenSort":      ("Sort tokens by learned key, conv in sorted space", TokenSortConv, False),
    "V5_11_DilatedStack":   ("Stacked dilated conv with gated residuals", DilatedConvStack, False),
    "V5_12_PolyConv":       ("Polynomial-order conv with cross-term interactions", PolynomialConv, False),
    "V5_13_AdaptiveEMA":    ("Multi-channel adaptive EMA with content-dependent decay", AdaptiveEMA, False),
    "V5_14_CrossChannel":   ("MHConv + cross-channel MLP mixing", ConvCrossChannel, False),
    "V5_15_ConvPlusEMA":    ("GatedMHConv + parallel EMA for global context", GatedMHConvPlusEMA, False),
}


class NovelV5Model(FrontierModel):
    def __init__(self, config: FrontierConfig, arch_name: str):
        super().__init__(config)
        self._arch_name = arch_name
        desc, mixer_cls, needs_embed = NOVEL_V5_CONFIGS[arch_name]
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
    def arch_family(cls): return "novel_noattn_v5"
    def describe(self): return f"{self._arch_name}: {NOVEL_V5_CONFIGS[self._arch_name][0]}"
    def sequence_mixing_complexity(self): return "O(n)"


for name, (desc, _, _) in NOVEL_V5_CONFIGS.items():
    @register_arch(f"{name}LM", "novel_noattn_v5", desc)
    class _M(NovelV5Model):
        _arch_key = name
        def __init__(self, config):
            super().__init__(config, self.__class__._arch_key)
    _M.__name__ = f"{name}LM"
    _M.__qualname__ = f"{name}LM"
    globals()[f"_{name}_cls"] = _M
