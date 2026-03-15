"""
10 Novel Architectures That Don't Exist
=========================================
Each is a genuinely new sequence mixing mechanism, not from any paper.
Designed for speed — fixed 4-minute training budget means faster
architectures get more steps and thus more learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ═══════════════════════════════════════════════════════════════
# Shared components
# ═══════════════════════════════════════════════════════════════

class GatedFFN(nn.Module):
    def __init__(self, d, d_ff, bias=True):
        super().__init__()
        self.g = nn.Linear(d, d_ff, bias=bias)
        self.u = nn.Linear(d, d_ff, bias=bias)
        self.d = nn.Linear(d_ff, d, bias=bias)
    def forward(self, x):
        return self.d(F.silu(self.g(x)) * self.u(x))

class Block(nn.Module):
    """Generic pre-norm block: norm→mixer→residual, norm→ffn→residual."""
    def __init__(self, d, d_ff, mixer, rs=1.0):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = mixer
        self.n2 = nn.RMSNorm(d)
        self.ffn = GatedFFN(d, d_ff)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x

class LM(nn.Module):
    """Wraps any list of blocks into a full LM."""
    def __init__(self, vocab, d, blocks, tie=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.scale = math.sqrt(d)
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        if tie:
            self.head.weight = self.emb.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.emb(x) * self.scale
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))


# ═══════════════════════════════════════════════════════════════
# 1. MULTI-SCALE EMA MIXER
# ═══════════════════════════════════════════════════════════════
# Idea: Multiple exponential moving averages at different time scales,
# gated and combined. Extremely fast (just cumulative ops).

class MultiScaleEMA(nn.Module):
    """Learned multi-scale exponential moving averages with gating."""
    def __init__(self, d, n_scales=4):
        super().__init__()
        self.n_scales = n_scales
        self.d_per = d // n_scales
        # Learned decay rates (initialized to different scales)
        init_decays = torch.linspace(-1.0, -5.0, n_scales)  # log-space
        self.log_decay = nn.Parameter(init_decays.unsqueeze(-1).expand(n_scales, self.d_per).clone())
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, L, D = x.shape
        # Split into scale groups
        chunks = x.reshape(B, L, self.n_scales, self.d_per)
        decay = torch.sigmoid(self.log_decay)  # (n_scales, d_per), values in (0,1)

        # EMA via sequential scan (fast for short seqs, could use parallel scan)
        outputs = torch.zeros_like(chunks)
        state = torch.zeros(B, self.n_scales, self.d_per, device=x.device, dtype=x.dtype)
        for t in range(L):
            state = decay.unsqueeze(0) * state + (1 - decay.unsqueeze(0)) * chunks[:, t]
            outputs[:, t] = state

        out = outputs.reshape(B, L, D)
        gate = torch.sigmoid(self.gate(x))
        return self.out(out * gate)


# ═══════════════════════════════════════════════════════════════
# 2. BUTTERFLY MIXER
# ═══════════════════════════════════════════════════════════════
# Idea: FFT-butterfly-like structured sparse mixing. log(n) stages
# of pairwise swaps at increasing distances. O(n log n) with tiny constant.

class ButterflyMixer(nn.Module):
    """Log-depth structured sparse token mixing inspired by FFT butterfly."""
    def __init__(self, d, max_len=2048):
        super().__init__()
        self.d = d
        n_stages = int(math.log2(max_len))
        # Per-stage learned mixing weights
        self.stage_weights = nn.ParameterList([
            nn.Parameter(torch.randn(d) * 0.02) for _ in range(n_stages)
        ])
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, L, D = x.shape
        h = x
        for stage, w in enumerate(self.stage_weights):
            stride = 1 << stage
            if stride >= L:
                break
            alpha = torch.sigmoid(w)  # (D,)
            # Create shifted version
            shifted = torch.roll(h, shifts=stride, dims=1)
            # Zero out the wrapped-around positions (causal)
            mask = torch.ones(L, device=x.device)
            mask[:stride] = 0
            shifted = shifted * mask.unsqueeze(0).unsqueeze(-1)
            h = alpha * h + (1 - alpha) * shifted

        gate = torch.sigmoid(self.gate(x))
        return self.out(h * gate)


# ═══════════════════════════════════════════════════════════════
# 3. DIFFUSION MIXER
# ═══════════════════════════════════════════════════════════════
# Idea: Information diffuses between adjacent tokens like heat diffusion.
# Multiple sub-steps per layer = wider effective receptive field.
# Think of it as a learned PDE on the sequence.

class DiffusionMixer(nn.Module):
    """Local diffusion: tokens exchange info with neighbors over K sub-steps."""
    def __init__(self, d, n_substeps=4):
        super().__init__()
        self.n_substeps = n_substeps
        # Per-substep diffusion rates and gates
        self.diff_rates = nn.ParameterList([
            nn.Parameter(torch.zeros(d)) for _ in range(n_substeps)
        ])
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, L, D = x.shape
        h = x
        for step in range(self.n_substeps):
            rate = torch.sigmoid(self.diff_rates[step])  # (D,)
            # Laplacian: h[t] - 0.5*(h[t-1] + h[t+1])
            left = F.pad(h[:, :-1], (0, 0, 1, 0))   # shift right (causal: use past)
            # Causal diffusion: only diffuse from past to present
            h = h + rate * (left - h)
        return self.out(h)


# ═══════════════════════════════════════════════════════════════
# 4. COMPETITIVE GATES MIXER
# ═══════════════════════════════════════════════════════════════
# Idea: 4 simple mixers (identity, shift-1, shift-2, local-avg) compete
# via a learned per-token gate. No attention, no recurrence.

class CompetitiveGates(nn.Module):
    """Multiple cheap mixing strategies compete via per-token gating."""
    def __init__(self, d, n_strategies=4):
        super().__init__()
        self.n = n_strategies
        self.gate = nn.Linear(d, n_strategies)
        self.projs = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_strategies)])
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, L, D = x.shape
        # Strategy 0: identity
        s0 = self.projs[0](x)
        # Strategy 1: shift by 1 (causal)
        s1 = self.projs[1](F.pad(x[:, :-1], (0, 0, 1, 0)))
        # Strategy 2: shift by 2
        s2 = self.projs[2](F.pad(x[:, :-2], (0, 0, 2, 0)))
        # Strategy 3: running average of last 4
        kernel = torch.ones(1, 1, 4, device=x.device) / 4
        avg = F.conv1d(
            x.transpose(1, 2),  # (B, D, L)
            kernel.expand(D, 1, 4),
            padding=3, groups=D
        )[:, :, :L].transpose(1, 2)
        s3 = self.projs[3](avg)

        strategies = torch.stack([s0, s1, s2, s3], dim=-1)  # (B, L, D, 4)
        weights = F.softmax(self.gate(x), dim=-1)  # (B, L, 4)
        mixed = (strategies * weights.unsqueeze(2)).sum(-1)  # (B, L, D)
        return self.out(mixed)


# ═══════════════════════════════════════════════════════════════
# 5. PING-PONG NET
# ═══════════════════════════════════════════════════════════════
# Idea: Two parallel half-width streams that exchange information
# every layer. Each stream has its own mixing, but they cross-talk.

class PingPongMixer(nn.Module):
    """Two parallel streams with cross-talk."""
    def __init__(self, d):
        super().__init__()
        half = d // 2
        self.half = half
        # Stream A: causal conv
        self.conv_a = nn.Conv1d(half, half, 3, padding=2, groups=half)
        # Stream B: shift + linear
        self.proj_b = nn.Linear(half, half)
        # Cross-talk
        self.cross_ab = nn.Linear(half, half, bias=False)
        self.cross_ba = nn.Linear(half, half, bias=False)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, L, D = x.shape
        a, b = x[..., :self.half], x[..., self.half:]

        # Stream A: causal conv
        a_conv = self.conv_a(a.transpose(1, 2))[:, :, :L].transpose(1, 2)
        # Stream B: shift + project
        b_shift = self.proj_b(F.pad(b[:, :-1], (0, 0, 1, 0)))

        # Cross-talk
        a_new = F.silu(a_conv + self.cross_ba(b_shift))
        b_new = F.silu(b_shift + self.cross_ab(a_conv))

        out = torch.cat([a_new, b_new], dim=-1)
        return self.out(out * torch.sigmoid(self.gate(x)))


# ═══════════════════════════════════════════════════════════════
# 6. HASH BUCKET MIXER
# ═══════════════════════════════════════════════════════════════
# Idea: Hash tokens into buckets based on their content, average
# within buckets. This is learned locality-sensitive hashing for
# content-based grouping without explicit attention.

class HashBucketMixer(nn.Module):
    """Content-based hashing into buckets + within-bucket averaging."""
    def __init__(self, d, n_buckets=32, n_hashes=2):
        super().__init__()
        self.n_buckets = n_buckets
        self.n_hashes = n_hashes
        # Learnable hash projections
        self.hash_projs = nn.ModuleList([
            nn.Linear(d, n_buckets, bias=False) for _ in range(n_hashes)
        ])
        self.value_proj = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, L, D = x.shape
        v = self.value_proj(x)

        # Causal mask: position indices
        positions = torch.arange(L, device=x.device)

        accum = torch.zeros_like(v)
        for hash_proj in self.hash_projs:
            # Soft hash assignment
            logits = hash_proj(x)  # (B, L, n_buckets)
            assignments = F.softmax(logits, dim=-1)  # (B, L, n_buckets)

            # Causal bucket aggregation: cumulative sum trick
            # For each bucket, accumulate values from past tokens assigned to it
            weighted_v = torch.einsum('bln,bld->blnd', assignments, v)  # (B, L, n_buckets, D)
            cum_v = torch.cumsum(weighted_v, dim=1)  # causal cumsum
            cum_w = torch.cumsum(assignments, dim=1)  # (B, L, n_buckets)

            # Read from buckets using current token's assignment
            read = torch.einsum('bln,blnd->bld', assignments, cum_v)
            norm = torch.einsum('bln,bln->bl', assignments, cum_w).unsqueeze(-1).clamp(min=1e-6)
            accum = accum + read / norm

        accum = accum / self.n_hashes
        gate = torch.sigmoid(self.gate(x))
        return self.out(accum * gate)


# ═══════════════════════════════════════════════════════════════
# 7. RECURSIVE COMPRESS MIXER
# ═══════════════════════════════════════════════════════════════
# Idea: Recursively compress the sequence by 2x at each level
# (like a binary tree), then broadcast back. Captures multi-scale
# context in O(n log n).

class RecursiveCompress(nn.Module):
    """Binary-tree compression + broadcast for multi-scale mixing."""
    def __init__(self, d, max_depth=4):
        super().__init__()
        self.max_depth = max_depth
        self.compress = nn.ModuleList([nn.Linear(d * 2, d) for _ in range(max_depth)])
        self.broadcast = nn.ModuleList([nn.Linear(d, d) for _ in range(max_depth)])
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, L, D = x.shape
        # Compress phase: merge pairs of tokens
        levels = [x]
        h = x
        for depth in range(self.max_depth):
            if h.shape[1] < 2:
                break
            L_curr = h.shape[1]
            # Pad to even length if needed
            if L_curr % 2 == 1:
                h = F.pad(h, (0, 0, 0, 1))
                L_curr += 1
            # Merge pairs
            pairs = h.reshape(B, L_curr // 2, 2 * D)
            h = F.silu(self.compress[depth](pairs))
            levels.append(h)

        # Broadcast phase: send compressed info back down
        for depth in range(len(levels) - 2, -1, -1):
            parent = levels[depth + 1]
            child = levels[depth]
            # Upsample parent to match child length
            parent_up = parent.repeat_interleave(2, dim=1)[:, :child.shape[1]]
            levels[depth] = child + self.broadcast[depth](parent_up)

        return self.out(levels[0][:, :L])


# ═══════════════════════════════════════════════════════════════
# 8. MOMENTUM MIXER
# ═══════════════════════════════════════════════════════════════
# Idea: Each channel has a "velocity" that accumulates gradients-of-tokens.
# Position = integral of velocity = integral of integral of input.
# Double integration captures trends, not just values.

class MomentumMixer(nn.Module):
    """Double-integration mixer: tokens build up velocity and position."""
    def __init__(self, d):
        super().__init__()
        self.d = d
        # Learned friction / damping
        self.friction = nn.Parameter(torch.zeros(d))
        self.input_scale = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, L, D = x.shape
        friction = torch.sigmoid(self.friction)  # damping in (0,1)
        impulse = self.input_scale(x)

        velocity = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        position = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(L):
            velocity = friction * velocity + impulse[:, t]
            position = friction * position + velocity
            outputs.append(position)

        out = torch.stack(outputs, dim=1)
        gate = torch.sigmoid(self.gate(x))
        return self.out(out * gate)


# ═══════════════════════════════════════════════════════════════
# 9. ROTATE-AND-REDUCE MIXER
# ═══════════════════════════════════════════════════════════════
# Idea: Learned permutation (soft) of channels, followed by local
# reduction (conv). By composing rotations across layers, global
# mixing emerges from purely local operations.

class RotateReduce(nn.Module):
    """Learned channel rotation + local causal reduction."""
    def __init__(self, d, kernel=5):
        super().__init__()
        # "Rotation" = learned orthogonal-ish mixing of channels
        self.rotate = nn.Linear(d, d, bias=False)
        # Local causal reduction
        self.reduce = nn.Conv1d(d, d, kernel, padding=kernel-1, groups=d)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, L, D = x.shape
        h = self.rotate(x)
        h = self.reduce(h.transpose(1, 2))[:, :, :L].transpose(1, 2)
        h = F.silu(h)
        return self.out(h * torch.sigmoid(self.gate(x)))


# ═══════════════════════════════════════════════════════════════
# 10. WAVELET MIXER
# ═══════════════════════════════════════════════════════════════
# Idea: Haar wavelet decomposition of the sequence, process at each
# scale with learned transforms, then reconstruct. Like a learned
# multi-resolution analysis.

class WaveletMixer(nn.Module):
    """Haar wavelet decompose → per-scale transform → reconstruct."""
    def __init__(self, d, n_levels=3):
        super().__init__()
        self.n_levels = n_levels
        self.detail_transforms = nn.ModuleList([nn.Linear(d, d) for _ in range(n_levels)])
        self.approx_transform = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, L, D = x.shape
        details = []
        h = x

        # Forward wavelet: split into approx + detail at each level
        for level in range(self.n_levels):
            curr_L = h.shape[1]
            if curr_L < 2:
                break
            if curr_L % 2 == 1:
                h = F.pad(h, (0, 0, 0, 1))
                curr_L += 1
            even = h[:, 0::2]  # (B, L/2, D)
            odd = h[:, 1::2]
            approx = (even + odd) / 2
            detail = (even - odd) / 2
            details.append((detail, curr_L, level))
            h = approx

        # Transform coarsest level
        h = self.approx_transform(h)

        # Inverse wavelet: reconstruct from coarse to fine
        for detail, orig_L, level in reversed(details):
            detail = self.detail_transforms[level](detail)
            # Reconstruct
            even = h + detail
            odd = h - detail
            # Interleave
            recon = torch.zeros(B, even.shape[1] * 2, D, device=x.device, dtype=x.dtype)
            recon[:, 0::2] = even
            recon[:, 1::2] = odd
            h = recon[:, :orig_L]

        h = h[:, :L]  # trim to original length
        gate = torch.sigmoid(self.gate(x))
        return self.out(h * gate)


# ═══════════════════════════════════════════════════════════════
# Model builders — each returns a full LM
# ═══════════════════════════════════════════════════════════════

def _count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

VOCAB = 49152
D = 512
D_FF = 2048
RS = 0.5  # residual scale (winner from ablation system)

def build_model(name: str) -> nn.Module:
    """Build a named novel architecture."""
    builders = {
        "multiscale_ema": build_multiscale_ema,
        "butterfly": build_butterfly,
        "diffusion": build_diffusion,
        "competitive_gates": build_competitive_gates,
        "pingpong": build_pingpong,
        "hash_bucket": build_hash_bucket,
        "recursive_compress": build_recursive_compress,
        "momentum": build_momentum,
        "rotate_reduce": build_rotate_reduce,
        "wavelet": build_wavelet,
    }
    if name not in builders:
        raise ValueError(f"Unknown arch: {name}. Available: {list(builders.keys())}")
    return builders[name]()


def build_multiscale_ema():
    blocks = [Block(D, D_FF, MultiScaleEMA(D, n_scales=4), RS) for _ in range(22)]
    return LM(VOCAB, D, blocks)

def build_butterfly():
    blocks = [Block(D, D_FF, ButterflyMixer(D), RS) for _ in range(22)]
    return LM(VOCAB, D, blocks)

def build_diffusion():
    blocks = [Block(D, D_FF, DiffusionMixer(D, n_substeps=4), RS) for _ in range(22)]
    return LM(VOCAB, D, blocks)

def build_competitive_gates():
    blocks = [Block(D, D_FF, CompetitiveGates(D), RS) for _ in range(18)]
    return LM(VOCAB, D, blocks)

def build_pingpong():
    blocks = [Block(D, D_FF, PingPongMixer(D), RS) for _ in range(20)]
    return LM(VOCAB, D, blocks)

def build_hash_bucket():
    blocks = [Block(D, D_FF, HashBucketMixer(D, n_buckets=32, n_hashes=2), RS) for _ in range(18)]
    return LM(VOCAB, D, blocks)

def build_recursive_compress():
    blocks = [Block(D, D_FF, RecursiveCompress(D, max_depth=4), RS) for _ in range(16)]
    return LM(VOCAB, D, blocks)

def build_momentum():
    blocks = [Block(D, D_FF, MomentumMixer(D), RS) for _ in range(22)]
    return LM(VOCAB, D, blocks)

def build_rotate_reduce():
    blocks = [Block(D, D_FF, RotateReduce(D, kernel=5), RS) for _ in range(20)]
    return LM(VOCAB, D, blocks)

def build_wavelet():
    blocks = [Block(D, D_FF, WaveletMixer(D, n_levels=3), RS) for _ in range(18)]
    return LM(VOCAB, D, blocks)


ALL_ARCHS = [
    "multiscale_ema",
    "butterfly",
    "diffusion",
    "competitive_gates",
    "pingpong",
    "hash_bucket",
    "recursive_compress",
    "momentum",
    "rotate_reduce",
    "wavelet",
]
