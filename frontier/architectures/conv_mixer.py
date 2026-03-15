"""
Convolution-Based Sequence Mixing
===================================
Replace attention with long convolutions or convolution hierarchies.

Key idea: Learned convolution kernels can capture sequential patterns
without the quadratic cost of attention. Modern approaches use
parameterized long convolutions that can capture very long dependencies.

Variants:
- Hyena: Hierarchy of long convolutions with gating
- ConvMixer: Depthwise convolutions + pointwise mixing
- MultiResConv: Multiple convolution scales in parallel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from frontier.architectures.base import (
    FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
)
from frontier.architectures.registry import register_arch


class CausalConv1d(nn.Module):
    """Causal 1D convolution (no future leakage)."""

    def __init__(self, d_model: int, kernel_size: int, groups: int = 1):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=groups, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.padding, 0))  # left-pad for causal
        x = self.conv(x)
        return x.transpose(1, 2)


class HyenaOperator(nn.Module):
    """
    Simplified Hyena operator: a hierarchy of gated long convolutions.

    Instead of a single attention matrix, Hyena uses:
    1. Short linear projections to create multiple "views"
    2. Long convolution filters (implicitly parameterized)
    3. Element-wise gating between views

    This creates an "implicit attention" with O(n log n) complexity
    (via FFT-based convolution).
    """

    def __init__(
        self,
        d_model: int,
        order: int = 2,  # number of Hyena recurrences
        kernel_size: int = 128,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.order = order

        # Input projections: create (order + 1) views
        self.in_proj = nn.Linear(d_model, d_model * (order + 1), bias=use_bias)

        # Implicit long convolution filters (one per order)
        self.filters = nn.ModuleList([
            ImplicitLongConv(d_model, kernel_size)
            for _ in range(order)
        ])

        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # Project to multiple views
        projections = self.in_proj(x).reshape(B, L, self.order + 1, D)
        views = [projections[:, :, i] for i in range(self.order + 1)]

        # Iterative gated convolution
        # v = views[0]
        # For each order i: v = views[i+1] * conv_i(v)
        v = views[0]
        for i in range(self.order):
            v = self.filters[i](v)
            v = v * views[i + 1]  # element-wise gating

        return self.dropout(self.out_proj(v))


class ImplicitLongConv(nn.Module):
    """
    Implicitly parameterized long convolution.

    Instead of storing a full kernel, we parameterize it as the output
    of a small MLP applied to position indices. This allows arbitrarily
    long kernels with fixed parameter count.

    For efficiency, we use FFT-based convolution.
    """

    def __init__(self, d_model: int, max_kernel_size: int = 128):
        super().__init__()
        self.d_model = d_model
        self.max_kernel_size = max_kernel_size

        # Kernel generator: position -> kernel weight
        self.kernel_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, d_model),
        )

        # Exponential decay for positions (ensures kernel dies off)
        self.decay = nn.Parameter(torch.ones(d_model) * -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        K = min(L, self.max_kernel_size)

        # Generate kernel
        positions = torch.arange(K, device=x.device, dtype=x.dtype).unsqueeze(-1) / K
        kernel = self.kernel_net(positions)  # (K, D)

        # Apply exponential decay
        decay = torch.exp(self.decay).unsqueeze(0)  # (1, D)
        decay_weights = torch.exp(-decay * torch.arange(K, device=x.device, dtype=x.dtype).unsqueeze(-1))
        kernel = kernel * decay_weights  # (K, D)

        # FFT-based causal convolution
        # Pad both x and kernel to FFT length
        fft_len = 1
        while fft_len < L + K - 1:
            fft_len *= 2

        # x: (B, L, D) -> per-channel 1D convolution
        x_padded = F.pad(x.transpose(1, 2), (0, fft_len - L))  # (B, D, fft_len)
        k_padded = F.pad(kernel.T, (0, fft_len - K))  # (D, fft_len)

        X = torch.fft.rfft(x_padded, dim=-1)
        K_fft = torch.fft.rfft(k_padded, dim=-1)

        Y = X * K_fft.unsqueeze(0)
        y = torch.fft.irfft(Y, n=fft_len, dim=-1)[..., :L]  # (B, D, L)

        return y.transpose(1, 2)


class MultiResConv(nn.Module):
    """
    Multi-resolution convolution: parallel convolutions at different scales.

    Captures both local patterns (small kernels) and longer-range
    dependencies (large kernels) simultaneously.
    """

    def __init__(
        self,
        d_model: int,
        kernel_sizes: tuple = (3, 7, 15, 31),
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__()
        n_scales = len(kernel_sizes)
        d_per_scale = d_model // n_scales

        self.convs = nn.ModuleList([
            CausalConv1d(d_per_scale, ks, groups=d_per_scale)
            for ks in kernel_sizes
        ])

        # Split and merge projections
        self.split_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.merge_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.gate = nn.Linear(d_model, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.split_proj(x)
        chunks = h.chunk(len(self.convs), dim=-1)
        convolved = [conv(chunk) for conv, chunk in zip(self.convs, chunks)]
        merged = torch.cat(convolved, dim=-1)
        gate = torch.sigmoid(self.gate(x))
        return self.dropout(self.merge_proj(merged * gate))


# ──────────────────────────────────────────────
# Blocks
# ──────────────────────────────────────────────

class HyenaBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, order: int = 2,
                 kernel_size: int = 128, dropout: float = 0.0,
                 residual_scale: float = 1.0, use_bias: bool = True):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.hyena = HyenaOperator(d_model, order, kernel_size, dropout, use_bias)
        self.norm2 = nn.RMSNorm(d_model)
        self.residual_scale = residual_scale

        self.gate_proj = nn.Linear(d_model, d_ff, bias=use_bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=use_bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=use_bias)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_scale * self.hyena(self.norm1(x))
        h = self.norm2(x)
        ffn_out = self.ffn_dropout(self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h)))
        x = x + self.residual_scale * ffn_out
        return x


class MultiResBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int,
                 kernel_sizes: tuple = (3, 7, 15, 31),
                 dropout: float = 0.0, residual_scale: float = 1.0,
                 use_bias: bool = True):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.conv = MultiResConv(d_model, kernel_sizes, dropout, use_bias)
        self.norm2 = nn.RMSNorm(d_model)
        self.residual_scale = residual_scale

        self.gate_proj = nn.Linear(d_model, d_ff, bias=use_bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=use_bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=use_bias)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_scale * self.conv(self.norm1(x))
        h = self.norm2(x)
        ffn_out = self.ffn_dropout(self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h)))
        x = x + self.residual_scale * ffn_out
        return x


# ──────────────────────────────────────────────
# Full Language Models
# ──────────────────────────────────────────────

@register_arch("HyenaLM", "convolution", "Hyena hierarchy long convolution language model")
class HyenaLM(FrontierModel):

    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config

        order = ac.get("order", 2)
        kernel_size = ac.get("kernel_size", 128)
        residual_scale = ac.get("residual_scale", 1.0)
        use_bias = ac.get("use_bias", True)

        self.embed = EmbeddingWithScale(config.vocab_size, config.d_model, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            HyenaBlock(
                config.d_model, config.d_ff, order, kernel_size,
                config.dropout, residual_scale, use_bias,
            )
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(config.d_model)
        self.head = LMHead(
            config.d_model, config.vocab_size,
            embedding_weight=self.embed.embedding.weight if config.tie_weights else None
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))

    @classmethod
    def arch_family(cls) -> str:
        return "convolution"

    def describe(self) -> str:
        ac = self.config.arch_config
        return f"Hyena: {self.config.n_layers}L x {self.config.d_model}d, order={ac.get('order', 2)}, k={ac.get('kernel_size', 128)}"

    def sequence_mixing_complexity(self) -> str:
        return "O(n log n)"


@register_arch("MultiResLM", "convolution", "Multi-resolution convolution language model")
class MultiResLM(FrontierModel):

    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config

        kernel_sizes = tuple(ac.get("kernel_sizes", [3, 7, 15, 31]))
        residual_scale = ac.get("residual_scale", 1.0)
        use_bias = ac.get("use_bias", True)

        self.embed = EmbeddingWithScale(config.vocab_size, config.d_model, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            MultiResBlock(
                config.d_model, config.d_ff, kernel_sizes,
                config.dropout, residual_scale, use_bias,
            )
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(config.d_model)
        self.head = LMHead(
            config.d_model, config.vocab_size,
            embedding_weight=self.embed.embedding.weight if config.tie_weights else None
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))

    @classmethod
    def arch_family(cls) -> str:
        return "convolution"

    def describe(self) -> str:
        return f"MultiRes Conv: {self.config.n_layers}L x {self.config.d_model}d"

    def sequence_mixing_complexity(self) -> str:
        return "O(n*k)"  # k = max kernel size
