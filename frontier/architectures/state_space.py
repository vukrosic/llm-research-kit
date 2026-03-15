"""
State Space Model Architectures
================================
Implements SSM-based language models inspired by Mamba, S4, and related work.

Key idea: Replace attention with structured state space layers that provide
O(n) sequence mixing with recurrent inference capability.

Architecture families:
- Mamba: Selective SSM with input-dependent parameters
- S4D: Diagonal state space with HiPPO initialization
- Gated SSM: SSM with various gating mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass

from frontier.architectures.base import (
    FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
)
from frontier.architectures.registry import register_arch


# ──────────────────────────────────────────────
# Core SSM Primitives
# ──────────────────────────────────────────────

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (Mamba-style).

    The key insight: make SSM parameters (B, C, Δ) input-dependent,
    allowing the model to selectively propagate or forget information
    along the sequence.

    For training, we use the parallel scan formulation.
    For inference, this naturally supports O(1) per-token recurrence.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand_factor: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor
        self.d_conv = d_conv

        # Input projection: x -> (z, x_ssm) where z is the gate
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=use_bias)

        # Short convolution before SSM (local context)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )

        # SSM parameters: input-dependent B, C, and discretization step Δ
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Learnable log(A) — initialized to S4D-real (log-spaced)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D "skip connection" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Delta (discretization step) projection bias initialization
        dt_init_std = 1.0 / math.sqrt(self.d_inner)
        # We project from x to get dt, but initialize the bias for good dt range
        inv_dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        self.dt_bias = nn.Parameter(inv_dt)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Project input to inner dimension with gate
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Short convolution for local context
        x_ssm = x_ssm.transpose(1, 2)  # (B, d_inner, L)
        x_ssm = self.conv1d(x_ssm)[:, :, :seq_len]  # causal conv
        x_ssm = x_ssm.transpose(1, 2)  # (B, L, d_inner)
        x_ssm = F.silu(x_ssm)

        # Compute input-dependent SSM parameters
        x_dbl = self.x_proj(x_ssm)  # (B, L, 2*d_state + 1)
        B = x_dbl[..., :self.d_state]  # (B, L, d_state)
        C = x_dbl[..., self.d_state:2*self.d_state]  # (B, L, d_state)
        dt = F.softplus(x_dbl[..., -1] + self.dt_bias.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1).mean(-1))  # (B, L)

        # Discretize: A_bar = exp(A * dt), B_bar = B * dt
        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative for stability
        dt_expanded = dt.unsqueeze(-1).unsqueeze(-1)  # (B, L, 1, 1)
        A_bar = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt_expanded)  # (B, L, d_inner, d_state)
        B_bar = B.unsqueeze(2) * dt_expanded  # (B, L, 1, d_state) broadcast

        # Parallel scan (sequential for correctness, can be optimized with associative scan)
        y = self._parallel_scan(x_ssm, A_bar, B_bar, C)

        # Skip connection + gate
        y = y + x_ssm * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)  # gated output

        return self.dropout(self.out_proj(y))

    def _parallel_scan(self, x, A_bar, B_bar, C):
        """
        Simple sequential scan. In production, use a CUDA kernel or
        associative scan for O(n log n) parallelism.

        For research at 2048 seq_len, sequential scan is fast enough.
        """
        batch, seq_len, d_inner = x.shape
        d_state = self.d_state

        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            # h = A_bar * h + B_bar * x
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            # y = C * h
            y_t = (C[:, t].unsqueeze(1) * h).sum(-1)  # (B, d_inner)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, L, d_inner)


class SSMBlock(nn.Module):
    """A single SSM block: norm -> SSM -> residual, norm -> FFN -> residual."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand_factor: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        ffn_type: str = "gated_silu",
        use_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            dropout=dropout,
            use_bias=use_bias,
        )
        self.norm2 = nn.RMSNorm(d_model)
        self.residual_scale = residual_scale

        # FFN
        if ffn_type == "gated_silu":
            self.ffn = GatedFFN(d_model, d_ff, dropout=dropout, use_bias=use_bias)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=use_bias),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model, bias=use_bias),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_scale * self.ssm(self.norm1(x))
        x = x + self.residual_scale * self.ffn(self.norm2(x))
        return x


class GatedFFN(nn.Module):
    """SiLU-gated FFN (SwiGLU variant)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = True):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=use_bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=use_bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


# ──────────────────────────────────────────────
# Full Language Models
# ──────────────────────────────────────────────

@register_arch("MambaLM", "state_space", "Mamba-style selective SSM language model")
class MambaLM(FrontierModel):
    """
    Full Mamba-style language model.

    Architecture: Embedding -> N x SSMBlock -> Norm -> LM Head
    Sequence mixing: Selective SSM (O(n) training, O(1) per-token inference)
    """

    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config

        d_model = config.d_model
        d_state = ac.get("d_state", 64)
        d_conv = ac.get("d_conv", 4)
        expand_factor = ac.get("expand_factor", 2)
        d_ff = config.d_ff
        n_layers = config.n_layers
        residual_scale = ac.get("residual_scale", 1.0)
        ffn_type = ac.get("ffn_type", "gated_silu")
        use_bias = ac.get("use_bias", True)

        self.embed = EmbeddingWithScale(config.vocab_size, d_model, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            SSMBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                d_ff=d_ff,
                dropout=config.dropout,
                residual_scale=residual_scale,
                ffn_type=ffn_type,
                use_bias=use_bias,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self.head = LMHead(
            d_model, config.vocab_size,
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
        h = self.norm(h)
        return self.head(h)

    @classmethod
    def arch_family(cls) -> str:
        return "state_space"

    def describe(self) -> str:
        ac = self.config.arch_config
        return (
            f"Mamba SSM: {self.config.n_layers}L x {self.config.d_model}d, "
            f"state={ac.get('d_state', 64)}, conv={ac.get('d_conv', 4)}, "
            f"expand={ac.get('expand_factor', 2)}"
        )

    def supports_recurrent_inference(self) -> bool:
        return True

    def sequence_mixing_complexity(self) -> str:
        return "O(n)"


@register_arch("S4DLM", "state_space", "S4D diagonal state space language model")
class S4DLM(FrontierModel):
    """
    S4D-style language model with diagonal state space layers.

    Unlike Mamba, S4D uses fixed (not input-dependent) A, B parameters
    with HiPPO-inspired initialization. Simpler but potentially less
    flexible.
    """

    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config

        d_model = config.d_model
        d_state = ac.get("d_state", 64)
        n_layers = config.n_layers
        d_ff = config.d_ff
        use_bias = ac.get("use_bias", True)

        self.embed = EmbeddingWithScale(config.vocab_size, d_model, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            S4DBlock(d_model, d_state, d_ff, config.dropout, use_bias=use_bias)
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self.head = LMHead(
            d_model, config.vocab_size,
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
        h = self.norm(h)
        return self.head(h)

    @classmethod
    def arch_family(cls) -> str:
        return "state_space"

    def describe(self) -> str:
        return f"S4D: {self.config.n_layers}L x {self.config.d_model}d, state={self.config.arch_config.get('d_state', 64)}"

    def supports_recurrent_inference(self) -> bool:
        return True

    def sequence_mixing_complexity(self) -> str:
        return "O(n)"


class S4DLayer(nn.Module):
    """Diagonal S4 layer with fixed A (HiPPO-initialized) and learnable B, C, D."""

    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # HiPPO-inspired diagonal A: -1/2 + ni for n=0..d_state-1
        # Real part initialized to log-spaced values
        A_real = -0.5 * torch.ones(d_model, d_state)
        A_imag = math.pi * torch.arange(d_state).float().unsqueeze(0).expand(d_model, -1)
        self.A_real = nn.Parameter(A_real)
        self.A_imag = nn.Parameter(A_imag)

        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.D = nn.Parameter(torch.ones(d_model))

        self.dt = nn.Parameter(torch.ones(d_model) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        dt = F.softplus(self.dt)

        # Discretize diagonal A
        A = torch.complex(self.A_real, self.A_imag)
        A_bar = torch.exp(A * dt.unsqueeze(-1))  # (d_model, d_state)
        B_bar = self.B * dt.unsqueeze(-1)

        # Sequential scan (real part only for output)
        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device, dtype=torch.cfloat)
        outputs = []

        for t in range(seq_len):
            h = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * x[:, t].unsqueeze(-1).to(torch.cfloat)
            y_t = (self.C.unsqueeze(0).to(torch.cfloat) * h).sum(-1).real
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y + x * self.D.unsqueeze(0).unsqueeze(0)


class S4DBlock(nn.Module):
    """S4D block: norm -> S4D -> residual, norm -> FFN -> residual."""

    def __init__(self, d_model: int, d_state: int, d_ff: int, dropout: float = 0.0, use_bias: bool = True):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.s4d = S4DLayer(d_model, d_state)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = GatedFFN(d_model, d_ff, dropout=dropout, use_bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.s4d(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
