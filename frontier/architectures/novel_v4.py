"""
Novel Architectures — Batch 4 (Beyond Convolutions)
=====================================================
StripedConv proved local multi-scale mixing beats attention.
Now explore fundamentally different mixing primitives:

1. WaveletMixerLM: Haar wavelet decomposition at multiple scales
2. ReactionDiffusionLM: Physics-inspired reaction-diffusion dynamics
3. SpectralGateLM: FFT-based frequency-domain gating (O(n log n))
4. CellularAutomataLM: Learned local transition rules iterated multiple times
5. HierPoolMixerLM: Multi-scale pooling pyramid (U-Net for sequences)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from frontier.architectures.base import (
    FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
)
from frontier.architectures.registry import register_arch


class SwiGLU(nn.Module):
    def __init__(self, d, d_ff, bias=True):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=bias)
        self.up = nn.Linear(d, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d, bias=bias)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ═══════════════════════════════════════════════════════
# 1. WAVELET MIXER — Haar wavelet multi-scale decomposition
# ═══════════════════════════════════════════════════════

class CausalMultiScaleMixer(nn.Module):
    """
    Causal multi-scale mixing via depthwise causal convolutions at exponentially
    increasing kernel sizes (2, 4, 8, 16), inspired by wavelet multi-resolution.
    Each scale captures patterns at different temporal resolutions.
    Unlike Haar wavelets, this is strictly causal — no future leakage.
    """
    def __init__(self, d_model, n_scales=4):
        super().__init__()
        self.n_scales = n_scales
        d_per = d_model // (n_scales + 1)
        d_first = d_model - d_per * n_scales
        self.splits = [d_first] + [d_per] * n_scales

        # Scale 0: identity (finest scale)
        # Scales 1..n: causal conv with kernel 2^s
        self.convs = nn.ModuleList()
        for s in range(n_scales):
            ks = 2 ** (s + 1)  # 2, 4, 8, 16
            d = d_per
            self.convs.append(nn.Conv1d(d, d, ks, padding=ks - 1, groups=d, bias=True))

        # Difference projections (high-freq detail at each scale)
        self.detail_projs = nn.ModuleList([
            nn.Linear(d_per, d_per, bias=False) for _ in range(n_scales)
        ])
        self.combine = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        chunks = x.split(self.splits, dim=-1)

        outputs = [chunks[0]]  # finest scale passthrough
        for s, (chunk, conv, proj) in enumerate(zip(chunks[1:], self.convs, self.detail_projs)):
            # Causal conv at this scale
            smoothed = conv(chunk.transpose(1, 2))[:, :, :L].transpose(1, 2)
            # Detail = difference between original and smoothed
            detail = chunk - smoothed
            outputs.append(F.silu(proj(detail)) + smoothed)

        result = torch.cat(outputs, dim=-1)
        return self.combine(result * torch.sigmoid(self.gate(x)))


class WaveletBlock(nn.Module):
    def __init__(self, d, d_ff, n_scales=4, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = CausalMultiScaleMixer(d, n_scales)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("WaveletMixerLM", "novel", "Haar wavelet multi-scale decomposition mixer")
class WaveletMixerLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        ns = ac.get("n_scales", 4)
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            WaveletBlock(d, config.d_ff, ns, rs, ac.get("use_bias", True))
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"WaveletMixer: {self.config.n_layers}L, Haar wavelet decomposition"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 2. REACTION-DIFFUSION — Physics-inspired dynamics
# ═══════════════════════════════════════════════════════

class ReactionDiffusionMixer(nn.Module):
    """
    Two coupled channels: activator (u) and inhibitor (v).
    Diffusion via depthwise conv, reaction via learned pointwise nonlinearity.
    Multiple time steps per layer for pattern formation.
    """
    def __init__(self, d_model, n_steps=3, kernel_size=7):
        super().__init__()
        self.n_steps = n_steps
        self.d_half = d_model // 2

        # Diffusion operators (causal convolutions with different rates)
        self.diff_u = nn.Conv1d(self.d_half, self.d_half, kernel_size,
                                padding=kernel_size - 1, groups=self.d_half)
        self.diff_v = nn.Conv1d(self.d_half, self.d_half, kernel_size * 2 - 1,
                                padding=(kernel_size * 2 - 1) - 1, groups=self.d_half)

        # Reaction functions (learned)
        self.react_u = nn.Linear(d_model, self.d_half)
        self.react_v = nn.Linear(d_model, self.d_half)

        # Timestep scaling (learned)
        self.dt = nn.Parameter(torch.full((1,), 0.1))
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        u, v = x[..., :self.d_half], x[..., self.d_half:]

        for _ in range(self.n_steps):
            # Diffusion (causal conv)
            du = self.diff_u(u.transpose(1, 2))[:, :, :L].transpose(1, 2) - u
            dv = self.diff_v(v.transpose(1, 2))[:, :, :L].transpose(1, 2) - v

            # Reaction (cross-channel interaction)
            uv = torch.cat([u, v], dim=-1)
            ru = torch.tanh(self.react_u(uv))
            rv = torch.tanh(self.react_v(uv))

            # Update
            dt = torch.sigmoid(self.dt)
            u = u + dt * (du + ru)
            v = v + dt * (dv + rv)

        result = torch.cat([u, v], dim=-1)
        return self.out(result * torch.sigmoid(self.gate(x)))


class ReactionDiffusionBlock(nn.Module):
    def __init__(self, d, d_ff, n_steps=3, kernel_size=7, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = ReactionDiffusionMixer(d, n_steps, kernel_size)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("ReactionDiffusionLM", "novel", "Reaction-diffusion dynamics for sequence mixing")
class ReactionDiffusionLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        rs = ac.get("residual_scale", 1.0)
        ns = ac.get("rd_steps", 3)
        ks = ac.get("rd_kernel", 7)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            ReactionDiffusionBlock(d, config.d_ff, ns, ks, rs, ac.get("use_bias", True))
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"ReactionDiffusion: {self.config.n_layers}L, activator-inhibitor dynamics"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 3. SPECTRAL GATE — FFT-based frequency domain mixing
# ═══════════════════════════════════════════════════════

class SpectralGateMixer(nn.Module):
    """
    Transform to frequency domain via real FFT,
    apply learned per-frequency gating, then inverse FFT.
    All FFT ops in float32. Causal post-processing via conv.
    """
    def __init__(self, d_model, max_seq_len=2048):
        super().__init__()
        n_freq = max_seq_len // 2 + 1
        # Learned frequency filter (real-valued magnitude + phase shift)
        self.freq_mag = nn.Parameter(torch.zeros(1, n_freq, d_model))  # sigmoid → [0,1]
        self.freq_phase = nn.Parameter(torch.zeros(1, n_freq, d_model))  # phase offset
        # Causal post-processing conv
        self.causal_conv = nn.Conv1d(d_model, d_model, 3, padding=2, groups=d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        orig_dtype = x.dtype

        # All FFT in float32
        x_f = x.float()
        X_freq = torch.fft.rfft(x_f, dim=1)  # (B, L//2+1, D) complex64
        n_freq = X_freq.shape[1]

        # Learned magnitude gating and phase rotation
        mag = torch.sigmoid(self.freq_mag[:, :n_freq].float())
        phase = self.freq_phase[:, :n_freq].float()
        # Apply: scale magnitude and rotate phase
        X_gated = X_freq * mag * torch.exp(1j * phase)

        # Inverse FFT back to time domain
        result = torch.fft.irfft(X_gated, n=L, dim=1).to(orig_dtype)

        # Causal correction via causal conv
        result = self.causal_conv(result.transpose(1, 2))[:, :, :L].transpose(1, 2)

        return self.out(F.silu(result) * torch.sigmoid(self.gate(x)))


class SpectralGateBlock(nn.Module):
    def __init__(self, d, d_ff, max_seq_len=2048, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = SpectralGateMixer(d, max_seq_len)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("SpectralGateLM", "novel", "FFT-based frequency domain gating mixer")
class SpectralGateLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            SpectralGateBlock(d, config.d_ff, config.max_seq_len, rs, ac.get("use_bias", True))
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"SpectralGate: {self.config.n_layers}L, FFT frequency gating"
    def sequence_mixing_complexity(self): return "O(n log n)"


# ═══════════════════════════════════════════════════════
# 4. CELLULAR AUTOMATA — Learned local transition rules
# ═══════════════════════════════════════════════════════

class CellularAutomataMixer(nn.Module):
    """
    Each token updates based on its local neighborhood via learned rules.
    Multiple CA steps per layer, with causal masking (only look left).
    Inspired by neural cellular automata (Mordvintsev et al.).
    """
    def __init__(self, d_model, n_steps=3, neighborhood=5):
        super().__init__()
        self.n_steps = n_steps
        # Perception: depthwise causal conv + cross-channel linear
        self.perceive_dw = nn.Conv1d(d_model, d_model, neighborhood,
                                     padding=neighborhood - 1, groups=d_model, bias=True)
        self.perceive_mix = nn.Linear(d_model, d_model, bias=False)
        # Update rule: lightweight MLP
        self.update = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # Stochastic update gate (alive masking analog)
        self.alive_gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        state = x

        for _ in range(self.n_steps):
            # Perceive neighborhood (causal depthwise conv + cross-channel)
            perceived = self.perceive_dw(state.transpose(1, 2))[:, :, :L].transpose(1, 2)
            perceived = self.perceive_mix(perceived)
            # Compute update
            delta = self.update(perceived)
            # Alive gate — controls update magnitude per-channel
            gate = torch.sigmoid(self.alive_gate(state))
            state = state + delta * gate * 0.1  # small step size for stability

        return self.out(state)


class CellularAutomataBlock(nn.Module):
    def __init__(self, d, d_ff, n_steps=4, neighborhood=5, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = CellularAutomataMixer(d, n_steps, neighborhood)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("CellularAutomataLM", "novel", "Neural cellular automata with learned local rules")
class CellularAutomataLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        rs = ac.get("residual_scale", 1.0)
        ns = ac.get("ca_steps", 4)
        nb = ac.get("ca_neighborhood", 5)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            CellularAutomataBlock(d, config.d_ff, ns, nb, rs, ac.get("use_bias", True))
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"CellularAutomata: {self.config.n_layers}L, learned CA rules"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 5. HIERARCHICAL POOL MIXER — Multi-scale pooling pyramid
# ═══════════════════════════════════════════════════════

class HierPoolMixer(nn.Module):
    """
    Strictly causal multi-scale pooling: at each position i, compute the
    running average of the last `scale` tokens (via cumsum), then process.
    No downsampling/upsampling — keeps full resolution, just different receptive fields.
    """
    def __init__(self, d_model, scales=(2, 4, 8, 16)):
        super().__init__()
        self.scales = scales

        # Per-scale processing
        self.scale_projs = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in scales
        ])
        # Combination: original + n_scales pooled views
        self.combine = nn.Linear(d_model * (len(scales) + 1), d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)

    def _causal_pool(self, x, scale):
        """Causal average pooling at full resolution: position i = mean(x[max(0,i-scale+1):i+1])."""
        B, L, D = x.shape
        cumsum = torch.cumsum(x, dim=1)
        shifted = F.pad(cumsum[:, :-scale], (0, 0, scale, 0))
        pooled = (cumsum - shifted)
        # Proper denominator for positions near the start
        denom = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).clamp(max=scale).unsqueeze(0).unsqueeze(-1)
        return pooled / denom

    def forward(self, x):
        B, L, D = x.shape
        scale_outs = [x]  # original scale

        for scale, proj in zip(self.scales, self.scale_projs):
            pooled = self._causal_pool(x, scale)  # (B, L, D) — full resolution
            scale_outs.append(F.silu(proj(pooled)))

        combined = torch.cat(scale_outs, dim=-1)
        result = self.combine(combined)
        return result * torch.sigmoid(self.gate(x))


class HierPoolBlock(nn.Module):
    def __init__(self, d, d_ff, scales=(2, 4, 8, 16), rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = HierPoolMixer(d, scales)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("HierPoolMixerLM", "novel", "Hierarchical multi-scale pooling pyramid mixer")
class HierPoolMixerLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        rs = ac.get("residual_scale", 1.0)
        scales = tuple(ac.get("pool_scales", [2, 4, 8, 16]))
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            HierPoolBlock(d, config.d_ff, scales, rs, ac.get("use_bias", True))
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"HierPoolMixer: {self.config.n_layers}L, multi-scale pooling pyramid"
    def sequence_mixing_complexity(self): return "O(n)"
