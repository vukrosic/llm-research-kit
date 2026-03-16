"""
Novel Architectures — Batch 6 (Exploit MultiHeadConv + New Ideas)
=================================================================
Best so far: MultiHeadConv 4.0748 (26% faster than transformer)
             StripedConvDeep 4.0711

Batch 6:
1. MultiHeadConvDeepLM: Deeper MultiHeadConv (20L, narrower FFN)
2. MultiHeadConvPoolLM: MultiHeadConv + causal pooling in alternating layers
3. ConvCascadeLM: Cascaded convs — output of short conv feeds into longer conv
4. DilatedConvMixerLM: Dilated causal convolutions (like WaveNet)
5. EMAMixerLM: Multiple exponential moving averages with different decay rates
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
# 1. MULTI-HEAD CONV DEEP — 20 layers, narrower FFN
# ═══════════════════════════════════════════════════════

class MultiHeadConvMixerV2(nn.Module):
    """Multi-head conv with gating and value projection."""
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # Exponentially spaced kernels
        kernel_sizes = [min(2**(i+1) + 1, 65) for i in range(n_heads)]

        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.head_convs = nn.ModuleList([
            nn.Conv1d(self.d_head, self.d_head, ks, padding=ks - 1, groups=self.d_head)
            for ks in kernel_sizes
        ])
        self.gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        v = self.v_proj(x)
        v_heads = v.view(B, L, self.n_heads, self.d_head)

        head_outs = []
        for h in range(self.n_heads):
            vh = v_heads[:, :, h]
            out = self.head_convs[h](vh.transpose(1, 2))[:, :, :L].transpose(1, 2)
            head_outs.append(F.silu(out))

        combined = torch.cat(head_outs, dim=-1)
        return self.out(combined * torch.sigmoid(self.gate(x)))


class MHConvBlock(nn.Module):
    def __init__(self, d, d_ff, n_heads=8, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = MultiHeadConvMixerV2(d, n_heads)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("MultiHeadConvDeepLM", "novel", "Deep MultiHeadConv: 20L with narrower FFN")
class MultiHeadConvDeepLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            MHConvBlock(d, config.d_ff, nh, rs, ac.get("use_bias", True))
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
    def describe(self): return f"MultiHeadConvDeep: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 2. MULTI-HEAD CONV + POOL — Alternating layers
# ═══════════════════════════════════════════════════════

class CausalPoolMixer(nn.Module):
    """Causal multi-scale pooling mixer (from batch 4, standalone)."""
    def __init__(self, d_model, scales=(2, 4, 8, 16)):
        super().__init__()
        self.scales = scales
        self.projs = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in scales])
        self.combine = nn.Linear(d_model * (len(scales) + 1), d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        outs = [x]
        cumsum = torch.cumsum(x, dim=1)
        denom_cache = {}
        for scale, proj in zip(self.scales, self.projs):
            shifted = F.pad(cumsum[:, :-scale], (0, 0, scale, 0))
            pooled = cumsum - shifted
            if scale not in denom_cache:
                denom_cache[scale] = torch.arange(1, L+1, device=x.device, dtype=x.dtype).clamp(max=scale).unsqueeze(0).unsqueeze(-1)
            outs.append(F.silu(proj(pooled / denom_cache[scale])))
        return self.combine(torch.cat(outs, dim=-1)) * torch.sigmoid(self.gate(x))


class MHConvPoolBlock(nn.Module):
    """Alternates between MultiHeadConv and CausalPool based on layer index."""
    def __init__(self, d, d_ff, layer_idx, n_heads=8, scales=(2, 4, 8, 16), rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        if layer_idx % 3 == 2:  # Every 3rd layer is pool
            self.mix = CausalPoolMixer(d, scales)
        else:
            self.mix = MultiHeadConvMixerV2(d, n_heads)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("MultiHeadConvPoolLM", "novel", "MultiHeadConv with causal pool every 3rd layer")
class MultiHeadConvPoolLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        ps = tuple(ac.get("pool_scales", [2, 4, 8, 16]))
        rs = ac.get("residual_scale", 1.0)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            MHConvPoolBlock(d, config.d_ff, i, nh, ps, rs, ac.get("use_bias", True))
            for i in range(config.n_layers)
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
    def describe(self): return f"MHConvPool: {self.config.n_layers}L, conv+pool hybrid"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 3. CONV CASCADE — Cascaded convolutions (short → long)
# ═══════════════════════════════════════════════════════

class ConvCascadeMixer(nn.Module):
    """
    Cascaded causal convolutions: output of short conv feeds into longer conv.
    This creates a growing receptive field within a single layer,
    similar to how WaveNet stacks dilated convolutions.
    """
    def __init__(self, d_model, n_stages=4, base_kernel=3):
        super().__init__()
        self.stages = nn.ModuleList()
        for i in range(n_stages):
            ks = base_kernel + i * 4  # 3, 7, 11, 15
            self.stages.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, ks, padding=ks - 1, groups=d_model),
                nn.SiLU(),
            ))
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        h = x.transpose(1, 2)  # (B, D, L)
        for stage in self.stages:
            h = stage(h)[:, :, :L]
        h = h.transpose(1, 2)
        return self.out(h * torch.sigmoid(self.gate(x)))


class ConvCascadeBlock(nn.Module):
    def __init__(self, d, d_ff, n_stages=4, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = ConvCascadeMixer(d, n_stages)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("ConvCascadeLM", "novel", "Cascaded causal convolutions with growing receptive field")
class ConvCascadeLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        rs = ac.get("residual_scale", 1.0)
        ns = ac.get("n_stages", 4)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            ConvCascadeBlock(d, config.d_ff, ns, rs, ac.get("use_bias", True))
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
    def describe(self): return f"ConvCascade: {self.config.n_layers}L, cascaded growing receptive field"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 4. DILATED CONV MIXER — WaveNet-style dilated convolutions
# ═══════════════════════════════════════════════════════

class DilatedConvMixer(nn.Module):
    """
    Parallel dilated causal convolutions with different dilation rates.
    Each channel group gets a different dilation, covering exponentially
    increasing receptive fields. Like WaveNet but parallel.
    """
    def __init__(self, d_model, n_dilations=4, base_kernel=3):
        super().__init__()
        self.n_dilations = n_dilations
        d_per = d_model // n_dilations
        d_first = d_model - d_per * (n_dilations - 1)
        self.splits = [d_first] + [d_per] * (n_dilations - 1)

        self.convs = nn.ModuleList()
        for i in range(n_dilations):
            d = d_first if i == 0 else d_per
            dilation = 2 ** i  # 1, 2, 4, 8
            padding = (base_kernel - 1) * dilation
            self.convs.append(
                nn.Conv1d(d, d, base_kernel, padding=padding, dilation=dilation, groups=d)
            )

        self.channel_mix = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        chunks = x.split(self.splits, dim=-1)
        conv_outs = []
        for chunk, conv in zip(chunks, self.convs):
            h = conv(chunk.transpose(1, 2))[:, :, :L].transpose(1, 2)
            conv_outs.append(h)
        out = torch.cat(conv_outs, dim=-1)
        out = F.silu(self.channel_mix(out))
        return self.out(out * torch.sigmoid(self.gate(x)))


class DilatedConvBlock(nn.Module):
    def __init__(self, d, d_ff, n_dilations=4, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = DilatedConvMixer(d, n_dilations)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("DilatedConvMixerLM", "novel", "WaveNet-style parallel dilated causal convolutions")
class DilatedConvMixerLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        rs = ac.get("residual_scale", 1.0)
        nd = ac.get("n_dilations", 4)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            DilatedConvBlock(d, config.d_ff, nd, rs, ac.get("use_bias", True))
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
    def describe(self): return f"DilatedConv: {self.config.n_layers}L, WaveNet-style dilations"
    def sequence_mixing_complexity(self): return "O(n)"


# ═══════════════════════════════════════════════════════
# 5. EMA MIXER — Multiple exponential moving averages
# ═══════════════════════════════════════════════════════

class EMAMixer(nn.Module):
    """
    Multiple parallel EMA channels with learned decay rates.
    Each channel group is smoothed with a different decay rate via cumsum trick.
    Stable: uses log-space computation, no exp(large_number).
    """
    def __init__(self, d_model, n_emas=8):
        super().__init__()
        self.n_emas = n_emas
        self.d_per = d_model // n_emas
        self.d_first = d_model - self.d_per * (n_emas - 1)
        # Learned decay rates (pre-sigmoid, so actual decay in [0,1])
        # Initialize to spread from fast (0.5) to slow (0.99)
        init_vals = torch.linspace(-0.0, 4.0, n_emas)  # sigmoid(0)=0.5, sigmoid(4)≈0.98
        self.decay_logits = nn.Parameter(init_vals)
        self.in_proj = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        h = self.in_proj(x)

        splits = [self.d_first] + [self.d_per] * (self.n_emas - 1)
        chunks = h.split(splits, dim=-1)

        ema_outs = []
        for i, chunk in enumerate(chunks):
            decay = torch.sigmoid(self.decay_logits[i])
            # EMA via cumsum: y[t] = decay * y[t-1] + (1-decay) * x[t]
            # = sum_{s<=t} (1-decay) * decay^(t-s) * x[s]
            # Compute in log-space to avoid overflow
            log_d = torch.log(decay + 1e-8)
            positions = torch.arange(L, device=x.device, dtype=chunk.dtype)
            # Scale input by decay^(-s), cumsum, then scale by decay^t
            inv_decay_weights = torch.exp(-log_d * positions).unsqueeze(0).unsqueeze(-1)
            decay_weights = torch.exp(log_d * positions).unsqueeze(0).unsqueeze(-1)
            weighted = chunk * (1 - decay) * inv_decay_weights
            cum = torch.cumsum(weighted, dim=1)
            ema_outs.append(cum * decay_weights)

        result = torch.cat(ema_outs, dim=-1)
        return self.out(F.silu(result) * torch.sigmoid(self.gate(x)))


class EMABlock(nn.Module):
    def __init__(self, d, d_ff, n_emas=8, rs=1.0, bias=True):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = EMAMixer(d, n_emas)
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


@register_arch("EMAMixerLM", "novel", "Multiple EMA channels with learned decay rates")
class EMAMixerLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        rs = ac.get("residual_scale", 1.0)
        ne = ac.get("n_emas", 8)
        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            EMABlock(d, config.d_ff, ne, rs, ac.get("use_bias", True))
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
    def describe(self): return f"EMAMixer: {self.config.n_layers}L, multi-rate EMA"
    def sequence_mixing_complexity(self): return "O(n)"
