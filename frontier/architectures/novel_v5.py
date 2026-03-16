"""
Novel Architectures — Batch 5 (Synthesis + Radical Ideas)
==========================================================
Findings so far:
- Multi-scale local mixing (StripedConv, CausalMultiScale, HierPool) consistently beats transformer
- Reaction-diffusion dynamics work surprisingly well
- Best results: StripedConvDeep 4.07, HierPool 4.08, WaveletMixer 4.11

Batch 5 strategy:
1. ConvPoolHybridLM: Combine StripedConv + HierPool (best of both)
2. DiffusionConvLM: Reaction-diffusion with StripedConv diffusion operator
3. SpectralGateV2LM: Fixed FFT mixer (float32 cast for complex ops)
4. GravityMixerLM: Tokens attract/repel based on learned "mass" — gravity sim
5. MultiHeadConvLM: Attention-like multi-head structure but with conv instead of softmax
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


def _make_model(cls_name, config, blocks_fn):
    """Shared boilerplate for all batch 5 models."""
    class Model(FrontierModel):
        def __init__(self, config):
            super().__init__(config)
            d = config.d_model
            self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
            self.blocks = blocks_fn(config)
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
        def describe(self): return f"{cls_name}: {self.config.n_layers}L"
        def sequence_mixing_complexity(self): return "O(n)"
    Model.__name__ = cls_name
    Model.__qualname__ = cls_name
    return Model


# ═══════════════════════════════════════════════════════
# 1. CONV-POOL HYBRID — Best of StripedConv + HierPool
# ═══════════════════════════════════════════════════════

class ConvPoolMixer(nn.Module):
    """
    Split channels: half goes through StripedConv (local multi-width conv),
    half goes through causal multi-scale pooling. Then combine.
    """
    def __init__(self, d_model, kernel_sizes=(3, 7, 15, 31), pool_scales=(2, 4, 8, 16)):
        super().__init__()
        self.d_conv = d_model // 2
        self.d_pool = d_model - self.d_conv

        # Conv branch (StripedConv-style)
        n_strips = len(kernel_sizes)
        d_per = self.d_conv // n_strips
        d_first = self.d_conv - d_per * (n_strips - 1)
        self.conv_splits = [d_first] + [d_per] * (n_strips - 1)
        self.convs = nn.ModuleList()
        for i, ks in enumerate(kernel_sizes):
            d = d_first if i == 0 else d_per
            self.convs.append(nn.Conv1d(d, d, ks, padding=ks - 1, groups=d, bias=True))

        # Pool branch
        self.pool_scales = pool_scales
        self.pool_projs = nn.ModuleList([
            nn.Linear(self.d_pool, self.d_pool, bias=False) for _ in pool_scales
        ])

        self.combine = nn.Linear(d_model + self.d_pool * len(pool_scales), d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)

    def _causal_pool(self, x, scale):
        B, L, D = x.shape
        cumsum = torch.cumsum(x, dim=1)
        shifted = F.pad(cumsum[:, :-scale], (0, 0, scale, 0))
        pooled = cumsum - shifted
        denom = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).clamp(max=scale).unsqueeze(0).unsqueeze(-1)
        return pooled / denom

    def forward(self, x):
        B, L, D = x.shape
        x_conv, x_pool = x[..., :self.d_conv], x[..., self.d_conv:]

        # Conv branch
        chunks = x_conv.split(self.conv_splits, dim=-1)
        conv_outs = []
        for chunk, conv in zip(chunks, self.convs):
            h = conv(chunk.transpose(1, 2))[:, :, :L].transpose(1, 2)
            conv_outs.append(F.silu(h))
        conv_out = torch.cat(conv_outs, dim=-1)

        # Pool branch
        pool_outs = []
        for scale, proj in zip(self.pool_scales, self.pool_projs):
            pooled = self._causal_pool(x_pool, scale)
            pool_outs.append(F.silu(proj(pooled)))

        combined = torch.cat([conv_out, x_pool] + pool_outs, dim=-1)
        return self.combine(combined) * torch.sigmoid(self.gate(x))


class ConvPoolBlock(nn.Module):
    def __init__(self, d, d_ff, ks=(3, 7, 15, 31), ps=(2, 4, 8, 16), rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = ConvPoolMixer(d, ks, ps)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("ConvPoolHybridLM", "novel", "StripedConv + causal pooling hybrid")
class ConvPoolHybridLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        ks = tuple(ac.get("kernel_sizes", [3, 7, 15, 31]))
        ps = tuple(ac.get("pool_scales", [2, 4, 8, 16]))
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            ConvPoolBlock(d, config.d_ff, ks, ps, rs, ac.get("use_bias", True))
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
    def describe(self): return f"ConvPoolHybrid: {self.config.n_layers}L, conv+pool channels"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 2. DIFFUSION-CONV — Reaction-diffusion with conv diffusion
# ═══════════════════════════════════════════════════════

class DiffusionConvMixer(nn.Module):
    """
    Like ReactionDiffusion but the diffusion operator is multi-width causal conv
    (StripedConv-style) instead of a single conv. Richer diffusion dynamics.
    """
    def __init__(self, d_model, n_steps=3, kernel_sizes=(3, 7, 15)):
        super().__init__()
        self.n_steps = n_steps
        self.d_half = d_model // 2

        # Multi-width diffusion for activator
        self.diff_convs = nn.ModuleList([
            nn.Conv1d(self.d_half, self.d_half, ks, padding=ks - 1, groups=self.d_half)
            for ks in kernel_sizes
        ])
        self.diff_combine = nn.Linear(self.d_half * len(kernel_sizes), self.d_half, bias=False)

        # Single conv diffusion for inhibitor (slower, wider)
        self.diff_v = nn.Conv1d(self.d_half, self.d_half, 15, padding=14, groups=self.d_half)

        # Reaction
        self.react_u = nn.Linear(d_model, self.d_half)
        self.react_v = nn.Linear(d_model, self.d_half)
        self.dt = nn.Parameter(torch.full((1,), 0.1))
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        u, v = x[..., :self.d_half], x[..., self.d_half:]

        for _ in range(self.n_steps):
            # Multi-width diffusion for u
            diff_outs = []
            for conv in self.diff_convs:
                diff_outs.append(conv(u.transpose(1, 2))[:, :, :L].transpose(1, 2))
            du = self.diff_combine(torch.cat(diff_outs, dim=-1)) - u

            # Single wide diffusion for v
            dv = self.diff_v(v.transpose(1, 2))[:, :, :L].transpose(1, 2) - v

            # Reaction
            uv = torch.cat([u, v], dim=-1)
            ru = torch.tanh(self.react_u(uv))
            rv = torch.tanh(self.react_v(uv))

            dt = torch.sigmoid(self.dt)
            u = u + dt * (du + ru)
            v = v + dt * (dv + rv)

        result = torch.cat([u, v], dim=-1)
        return self.out(result * torch.sigmoid(self.gate(x)))


class DiffusionConvBlock(nn.Module):
    def __init__(self, d, d_ff, n_steps=3, ks=(3, 7, 15), rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = DiffusionConvMixer(d, n_steps, ks)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("DiffusionConvLM", "novel", "Reaction-diffusion with multi-width conv diffusion")
class DiffusionConvLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        rs = ac.get("residual_scale", 1.0)
        ns = ac.get("rd_steps", 3)
        ks = tuple(ac.get("diff_kernels", [3, 7, 15]))
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            DiffusionConvBlock(d, config.d_ff, ns, ks, rs, ac.get("use_bias", True))
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
    def describe(self): return f"DiffusionConv: {self.config.n_layers}L, multi-width diffusion"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 3. SPECTRAL GATE V2 — FFT mixer with float32 safety
# ═══════════════════════════════════════════════════════

class SpectralGateV2Mixer(nn.Module):
    """FFT frequency gating, all ops in float32 for safety."""
    def __init__(self, d_model, max_seq_len=2048):
        super().__init__()
        n_freq = max_seq_len // 2 + 1
        self.freq_mag = nn.Parameter(torch.zeros(1, n_freq, d_model))
        self.freq_phase = nn.Parameter(torch.zeros(1, n_freq, d_model))
        self.causal_conv = nn.Conv1d(d_model, d_model, 5, padding=4, groups=d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        orig_dtype = x.dtype
        x_f = x.float()
        X_freq = torch.fft.rfft(x_f, dim=1)
        n_freq = X_freq.shape[1]
        mag = torch.sigmoid(self.freq_mag[:, :n_freq].float())
        phase = self.freq_phase[:, :n_freq].float()
        X_gated = X_freq * mag * torch.exp(1j * phase)
        result = torch.fft.irfft(X_gated, n=L, dim=1).to(orig_dtype)
        result = self.causal_conv(result.transpose(1, 2))[:, :, :L].transpose(1, 2)
        return self.out(F.silu(result) * torch.sigmoid(self.gate(x)))


class SpectralGateV2Block(nn.Module):
    def __init__(self, d, d_ff, max_seq_len=2048, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = SpectralGateV2Mixer(d, max_seq_len)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("SpectralGateV2LM", "novel", "FFT frequency gating v2 (float32 safe)")
class SpectralGateV2LM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            SpectralGateV2Block(d, config.d_ff, config.max_seq_len, rs, ac.get("use_bias", True))
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
    def describe(self): return f"SpectralGateV2: {self.config.n_layers}L, FFT gating"
    def sequence_mixing_complexity(self): return "O(n log n)"


# ═══════════════════════════════════════════════════════
# 4. GRAVITY MIXER — Token interaction via inverse-distance "gravity"
# ═══════════════════════════════════════════════════════

class GravityMixer(nn.Module):
    """
    Each token has a learned "mass" (per-channel). Tokens interact via
    causal gravity: token j influences token i (j <= i) proportional to
    mass_j / distance(i,j). Uses cumsum trick for O(n) computation.
    """
    def __init__(self, d_model, n_fields=4):
        super().__init__()
        self.n_fields = n_fields
        d_per = d_model // n_fields
        self.d_per = d_per
        self.d_first = d_model - d_per * (n_fields - 1)

        # Mass projection per field (learned from input)
        self.mass_proj = nn.Linear(d_model, d_model)
        # Decay rates per field (learned, controls distance falloff)
        self.decay = nn.Parameter(torch.linspace(0.8, 0.99, n_fields))
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        mass = torch.sigmoid(self.mass_proj(x))  # (B, L, D) "mass" per token per channel

        # Weighted input: mass * x
        weighted = mass * x

        # For each field with different decay rate, compute causal gravity sum
        splits = [self.d_first] + [self.d_per] * (self.n_fields - 1)
        chunks = weighted.split(splits, dim=-1)

        field_outs = []
        for i, chunk in enumerate(chunks):
            decay = torch.sigmoid(self.decay[i])
            # Exponential distance weighting via cumsum trick:
            # field_out[t] = sum_{s<=t} decay^(t-s) * chunk[s]
            # = decay * field_out[t-1] + chunk[t]
            # Compute via log-space cumsum for numerical stability
            log_decay = torch.log(decay.clamp(min=1e-6))
            # Create position-weighted inputs
            positions = torch.arange(L, device=x.device, dtype=x.dtype)
            # chunk[s] * decay^(-s), then cumsum, then * decay^t
            scale_down = torch.exp(-log_decay * positions).unsqueeze(0).unsqueeze(-1)
            scale_up = torch.exp(log_decay * positions).unsqueeze(0).unsqueeze(-1)
            weighted_chunk = chunk * scale_down
            cum = torch.cumsum(weighted_chunk, dim=1)
            field_outs.append(cum * scale_up)

        result = torch.cat(field_outs, dim=-1)
        return self.out(F.silu(result) * torch.sigmoid(self.gate(x)))


class GravityBlock(nn.Module):
    def __init__(self, d, d_ff, n_fields=4, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = GravityMixer(d, n_fields)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("GravityMixerLM", "novel", "Gravity-inspired token interaction with exponential decay")
class GravityMixerLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        rs = ac.get("residual_scale", 1.0)
        nf = ac.get("n_fields", 4)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            GravityBlock(d, config.d_ff, nf, rs, ac.get("use_bias", True))
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
    def describe(self): return f"GravityMixer: {self.config.n_layers}L, causal gravity fields"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 5. MULTI-HEAD CONV — Attention structure with conv mixing
# ═══════════════════════════════════════════════════════

class MultiHeadConvMixer(nn.Module):
    """
    Like multi-head attention but replace Q@K^T with causal convolution.
    Each head has a different kernel size (like different "attention patterns").
    V projection still exists. Output combines all heads.
    """
    def __init__(self, d_model, n_heads=8, kernel_sizes=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if kernel_sizes is None:
            # Exponentially spaced kernels across heads
            kernel_sizes = [2**(i+1) + 1 for i in range(n_heads)]  # 3, 5, 9, 17, 33, 65, 129, 257
            # Cap at reasonable size
            kernel_sizes = [min(ks, 65) for ks in kernel_sizes]

        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Per-head causal conv (depthwise on d_head channels)
        self.head_convs = nn.ModuleList([
            nn.Conv1d(self.d_head, self.d_head, ks, padding=ks - 1, groups=self.d_head)
            for ks in kernel_sizes[:n_heads]
        ])

        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        v = self.v_proj(x)
        # Split into heads
        v_heads = v.view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)  # (B, H, L, d_head)

        head_outs = []
        for h in range(self.n_heads):
            vh = v_heads[:, h]  # (B, L, d_head)
            # Causal conv per head
            out = self.head_convs[h](vh.transpose(1, 2))[:, :, :L].transpose(1, 2)
            head_outs.append(F.silu(out))

        # Concatenate heads
        combined = torch.cat(head_outs, dim=-1)  # (B, L, D)
        return self.out(combined)


class MultiHeadConvBlock(nn.Module):
    def __init__(self, d, d_ff, n_heads=8, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = MultiHeadConvMixer(d, n_heads)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("MultiHeadConvLM", "novel", "Multi-head attention structure with per-head conv mixing")
class MultiHeadConvLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        rs = ac.get("residual_scale", 1.0)
        nh = ac.get("n_heads", 8)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            MultiHeadConvBlock(d, config.d_ff, nh, rs, ac.get("use_bias", True))
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
    def describe(self): return f"MultiHeadConv: {self.config.n_layers}L, conv-based multi-head"
    def sequence_mixing_complexity(self): return "O(n)"
