"""
Extended Ablation Layers
========================
Provides building blocks for ~40 architecture ablation experiments.
All modules are designed to be drop-in replaceable via config flags.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchtune.modules import RotaryPositionalEmbeddings


# ──────────────────────────────────────────────
# Normalization helpers
# ──────────────────────────────────────────────

def build_norm(norm_type: str, d_model: int) -> nn.Module:
    """Factory for normalization layers."""
    if norm_type == "rmsnorm":
        return nn.RMSNorm(d_model)
    elif norm_type == "layernorm":
        return nn.LayerNorm(d_model)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


# ──────────────────────────────────────────────
# Feed-Forward Networks
# ──────────────────────────────────────────────

class FeedForwardAblation(nn.Module):
    """Standard 2-layer FFN with configurable activation."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0,
                 activation_type: str = "squared_relu", use_bias: bool = False):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=use_bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        self.activation_type = activation_type

    def forward(self, x):
        x = self.up_proj(x)
        if self.activation_type == "squared_relu":
            x = torch.square(F.relu(x))
        elif self.activation_type == "gelu":
            x = F.gelu(x)
        elif self.activation_type == "silu":
            x = F.silu(x)
        elif self.activation_type == "relu":
            x = F.relu(x)
        elif self.activation_type == "tanh":
            x = torch.tanh(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation_type}")
        return self.down_proj(self.dropout(x))


class SwiGLUFFN(nn.Module):
    """SwiGLU: gate × SiLU(gate_proj) — matches LLaMA style.
    Uses d_ff as the hidden size of each branch (total params similar to 2×d_ff vanilla)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        # keep param count ~equal: SwiGLU needs 3 matrices; vanilla FFN needs 2.
        # We scale the hidden dim by 2/3 so param count matches.
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class GatedLinearFFN(nn.Module):
    """GLU-style: sigmoid gate × linear branch."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(torch.sigmoid(self.gate_proj(x)) * self.up_proj(x)))


class BilinearFFN(nn.Module):
    """Bilinear FFN: gate × linear — no nonlinearity."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(self.gate_proj(x) * self.up_proj(x)))


class GatedSquaredReluFFN(nn.Module):
    """Gated Squared ReLU: sigmoid gate × squared_relu branch."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(
            self.dropout(torch.sigmoid(self.gate_proj(x)) * torch.square(F.relu(self.up_proj(x))))
        )


# ──────────────────────────────────────────────
# SwiGLU Variations (20-experiment swarm)
# ──────────────────────────────────────────────

class SwiGLUNarrowFFN(nn.Module):
    """SwiGLU with hidden_dim = d_ff × ½ (very narrow gated FFN)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 1 / 2)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class SwiGLUThreeQuarterFFN(nn.Module):
    """SwiGLU with hidden_dim = d_ff × ¾ (slightly wider than 2/3)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 3 / 4)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class SwiGLUFullWidthFFN(nn.Module):
    """SwiGLU with hidden_dim = d_ff × 1 (un-compressed; more params than baseline)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = d_ff  # 1× — 50% more params than 2/3 variant
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class SwiGLUWideFFN(nn.Module):
    """SwiGLU with hidden_dim = d_ff × 8/3 (LLaMA-style ratio when d_ff=4×d_model)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        # 8/3 × d_ff is the common LLaMA ratio derived from 4×d_model × 2/3 kept param-matched
        # but here d_ff is already 4×d_model so we use 8/3 to get the real LLaMA hidden size
        hidden = int(d_model * 8 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class GeGLUFFN(nn.Module):
    """GeGLU: GELU gate × linear branch (Noam Shazeer 2020)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.gelu(self.gate_proj(x)) * self.up_proj(x)))


class ReGLUFFN(nn.Module):
    """ReGLU: ReLU gate × linear branch."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.relu(self.gate_proj(x)) * self.up_proj(x)))


class SwiGLUDualGateFFN(nn.Module):
    """SwiGLU with two independent gating paths — both are SiLU-gated.
    Output = down(SiLU(g1) × SiLU(g2) × up). Encourages selectivity."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 1 / 2)
        self.gate1     = nn.Linear(d_model, hidden, bias=use_bias)
        self.gate2     = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(
            self.dropout(F.silu(self.gate1(x)) * F.silu(self.gate2(x)) * self.up_proj(x))
        )


class SwiGLUResidualFFN(nn.Module):
    """SwiGLU with an extra skip connection *inside* the FFN."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj     = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj   = nn.Linear(hidden, d_model, bias=use_bias)
        self.linear_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x):
        gated = self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))
        return gated + self.linear_proj(x)


class SwiGLUSharedGateFFN(nn.Module):
    """SwiGLU where gate and up projections are a single fused Linear."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.shared_proj = nn.Linear(d_model, hidden * 2, bias=use_bias)
        self.down_proj   = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x):
        proj = self.shared_proj(x)
        gate, up = proj.chunk(2, dim=-1)
        return self.down_proj(self.dropout(F.silu(gate) * up))


class SwiGLUDeepFFN(nn.Module):
    """Double-stacked SwiGLU with internal residuals."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 1 / 3)
        self.g1 = nn.Linear(d_model, hidden, bias=use_bias)
        self.u1 = nn.Linear(d_model, hidden, bias=use_bias)
        self.d1 = nn.Linear(hidden,  d_model, bias=use_bias)
        self.g2 = nn.Linear(d_model, hidden, bias=use_bias)
        self.u2 = nn.Linear(d_model, hidden, bias=use_bias)
        self.d2 = nn.Linear(hidden,  d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.d1(self.dropout(F.silu(self.g1(x)) * self.u1(x)))
        x = x + self.d2(self.dropout(F.silu(self.g2(x)) * self.u2(x)))
        return x


class SwiGLUSwiGLUFFN(nn.Module):
    """Sequential SwiGLU: stage-1 → stage-2, no inner residual."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        mid    = hidden // 2
        self.g1 = nn.Linear(d_model, hidden, bias=use_bias)
        self.u1 = nn.Linear(d_model, hidden, bias=use_bias)
        self.d1 = nn.Linear(hidden, mid, bias=use_bias)
        self.g2 = nn.Linear(mid, mid, bias=use_bias)
        self.u2 = nn.Linear(mid, mid, bias=use_bias)
        self.d2 = nn.Linear(mid, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.d1(self.dropout(F.silu(self.g1(x)) * self.u1(x)))
        return self.d2(self.dropout(F.silu(self.g2(h)) * self.u2(h)))


class SwiGLUBiasFFN(nn.Module):
    """SwiGLU with bias on all projections."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=True)
        self.up_proj   = nn.Linear(d_model, hidden, bias=True)
        self.down_proj = nn.Linear(hidden, d_model, bias=True)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ──────────────────────────────────────────────
# SciSpace-inspired GLU Variations (20 experiments)
# ──────────────────────────────────────────────

class SciSinGLUFFN(nn.Module):
    """SinGLU: sin() gate × linear branch. Periodic gating."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(torch.sin(self.gate_proj(x)) * self.up_proj(x)))


class SciTanhGLUFFN(nn.Module):
    """TanhGLU: tanh() gate × linear branch."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(torch.tanh(self.gate_proj(x)) * self.up_proj(x)))


class SciSigmoidGLUFFN(nn.Module):
    """Classic GLU: sigmoid() gate × linear branch (original Dauphin 2017)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(torch.sigmoid(self.gate_proj(x)) * self.up_proj(x)))


class SciSoftplusGLUFFN(nn.Module):
    """SoftplusGLU: softplus() gate × linear branch. Smooth, positive gate."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.softplus(self.gate_proj(x)) * self.up_proj(x)))


class SciELUGLUFFN(nn.Module):
    """ELUGLU: ELU() gate × linear branch."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.elu(self.gate_proj(x)) * self.up_proj(x)))


class SciCELUGLUFFN(nn.Module):
    """CELUGLU: CELU() gate × linear branch."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.celu(self.gate_proj(x)) * self.up_proj(x)))


class SciHardswishGLUFFN(nn.Module):
    """HardswishGLU: hardswish() gate × linear branch. Cheap SiLU approx."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.hardswish(self.gate_proj(x)) * self.up_proj(x)))


class SciLaplaceGLUFFN(nn.Module):
    """LaplaceGLU: exp(-|x|) gate × linear branch. Double-exponential decay."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(torch.exp(-torch.abs(self.gate_proj(x))) * self.up_proj(x)))


class SciSinCosGLUFFN(nn.Module):
    """SinCosGLU: sin(g) + cos(g) gate × linear branch. Dual periodic."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        g = self.gate_proj(x)
        return self.down_proj(self.dropout((torch.sin(g) + torch.cos(g)) * self.up_proj(x)))


class SciSingleProjGLUFFN(nn.Module):
    """Single-projection GLU: one proj split into gate + value (minimal params)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.proj = nn.Linear(d_model, hidden * 2, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        proj = self.proj(x)
        gate, val = proj.chunk(2, dim=-1)
        return self.down_proj(self.dropout(F.silu(gate) * val))


class SciTripleProjGLUFFN(nn.Module):
    """Triple-projection GLU: gate × up × aux (three independent projections)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 1 / 2)  # reduced hidden to keep params manageable
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.aux_proj  = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(
            self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x) * torch.sigmoid(self.aux_proj(x)))
        )


class SciPreGateGLUFFN(nn.Module):
    """Pre-gate GLU: gate is applied BEFORE the up-projection."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, d_model, bias=use_bias)  # gate in d_model space
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        gated_input = F.silu(self.gate_proj(x)) * x
        return self.down_proj(self.dropout(self.up_proj(gated_input)))


class SciPostGateGLUFFN(nn.Module):
    """Post-gate GLU: gate is applied AFTER the down-projection."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.gate_proj = nn.Linear(d_model, d_model, bias=use_bias)  # gate in d_model space
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        h = self.down_proj(self.dropout(F.silu(self.up_proj(x))))
        return h * torch.sigmoid(self.gate_proj(x))


class SciSwiGLUTopKFFN(nn.Module):
    """SwiGLU with top-k sparsity on the gate (only top-k values pass)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)
        self.k = max(1, hidden // 4)  # keep top 25%

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        # Zero out all but top-k gate values per token
        topk_vals, topk_idx = torch.topk(gate, self.k, dim=-1)
        mask = torch.zeros_like(gate)
        mask.scatter_(-1, topk_idx, 1.0)
        gate = gate * mask
        return self.down_proj(self.dropout(gate * self.up_proj(x)))


class SciSwiGLULeakyFFN(nn.Module):
    """SwiGLU with LeakyReLU gate instead of SiLU."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.leaky_relu(self.gate_proj(x), 0.01) * self.up_proj(x)))


class SciAsymGLUFFN(nn.Module):
    """Asymmetric GLU: gate uses SiLU, value uses GELU (different activations)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * F.gelu(self.up_proj(x))))


class SciSwiGLUPrenormDownFFN(nn.Module):
    """SwiGLU with LayerNorm applied to hidden repr before down-proj."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.hidden_norm = nn.LayerNorm(hidden)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.dropout(self.hidden_norm(h)))


class SciSwiGLUScaleGateFFN(nn.Module):
    """SwiGLU with a learnable per-dim scale factor on the gate output."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.gate_scale = nn.Parameter(torch.ones(hidden))
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x)) * self.gate_scale
        return self.down_proj(self.dropout(gate * self.up_proj(x)))


class SciCompositeGLUFFN(nn.Module):
    """Composite GLU: 0.5*SiLU(g) + 0.5*GELU(g) as the gate. Blended activation."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        g = self.gate_proj(x)
        gate = 0.5 * F.silu(g) + 0.5 * F.gelu(g)
        return self.down_proj(self.dropout(gate * self.up_proj(x)))


class SciSwiGLUMoELiteFFN(nn.Module):
    """MoE-lite SwiGLU: two separate SwiGLU experts, soft-routed via sigmoid."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 1 / 3)  # each expert is small
        # Expert 1
        self.g1 = nn.Linear(d_model, hidden, bias=use_bias)
        self.u1 = nn.Linear(d_model, hidden, bias=use_bias)
        self.d1 = nn.Linear(hidden, d_model, bias=use_bias)
        # Expert 2
        self.g2 = nn.Linear(d_model, hidden, bias=use_bias)
        self.u2 = nn.Linear(d_model, hidden, bias=use_bias)
        self.d2 = nn.Linear(hidden, d_model, bias=use_bias)
        # Router
        self.router = nn.Linear(d_model, 2, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        weights = torch.softmax(self.router(x), dim=-1)  # (B, T, 2)
        e1 = self.d1(self.dropout(F.silu(self.g1(x)) * self.u1(x)))
        e2 = self.d2(self.dropout(F.silu(self.g2(x)) * self.u2(x)))
        return weights[..., 0:1] * e1 + weights[..., 1:2] * e2


def build_ffn(ffn_type: str, d_model: int, d_ff: int, dropout: float,
              activation_type: str = "squared_relu", use_bias: bool = False) -> nn.Module:
    """Factory for FFN modules."""
    if ffn_type == "standard":
        return FeedForwardAblation(d_model, d_ff, dropout, activation_type, use_bias)
    elif ffn_type == "swiglu":
        return SwiGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "glu":
        return GatedLinearFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "bilinear":
        return BilinearFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "gated_sq_relu":
        return GatedSquaredReluFFN(d_model, d_ff, dropout, use_bias)
    # ── SwiGLU variations ──────────────────────────────────────────────────
    elif ffn_type == "swiglu_narrow":       return SwiGLUNarrowFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "swiglu_3q":           return SwiGLUThreeQuarterFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "swiglu_full":         return SwiGLUFullWidthFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "swiglu_wide":         return SwiGLUWideFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "geglu":               return GeGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "reglu":               return ReGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "swiglu_dual_gate":    return SwiGLUDualGateFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "swiglu_residual":     return SwiGLUResidualFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "swiglu_shared_gate":  return SwiGLUSharedGateFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "swiglu_deep":         return SwiGLUDeepFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "swiglu_swiglu":       return SwiGLUSwiGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "swiglu_bias":         return SwiGLUBiasFFN(d_model, d_ff, dropout, use_bias)
    # ── SciSpace-inspired variations ───────────────────────────────────────
    elif ffn_type == "scispace_singlu":       return SciSinGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_tanhglu":      return SciTanhGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_sigmoidglu":   return SciSigmoidGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_softplusglu":  return SciSoftplusGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_eluglu":       return SciELUGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_celuglu":      return SciCELUGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_hardswishglu": return SciHardswishGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_laplaceglu":   return SciLaplaceGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_sincosglu":    return SciSinCosGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_singleprojglu":return SciSingleProjGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_tripleprojglu":return SciTripleProjGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_pregatelu":    return SciPreGateGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_postgatelu":   return SciPostGateGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_topkglu":      return SciSwiGLUTopKFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_leakyglu":     return SciSwiGLULeakyFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_asymglu":      return SciAsymGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_prenormdown":  return SciSwiGLUPrenormDownFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_scalegate":    return SciSwiGLUScaleGateFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_composite":    return SciCompositeGLUFFN(d_model, d_ff, dropout, use_bias)
    elif ffn_type == "scispace_moelite":      return SciSwiGLUMoELiteFFN(d_model, d_ff, dropout, use_bias)
    else:
        raise ValueError(f"Unknown ffn_type: {ffn_type}")


# ──────────────────────────────────────────────
# RoPE
# ──────────────────────────────────────────────

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(dim=dim, max_seq_len=max_seq_len, base=base)

    def forward(self, x_BTHD: torch.Tensor):
        return self.rope(x_BTHD)


# ──────────────────────────────────────────────
# Attention
# ──────────────────────────────────────────────

class MultiHeadAttentionAblation(nn.Module):
    """
    Configurable Multi-Head Attention supporting:
      - GQA (n_kv_heads < n_heads)
      - MQA (n_kv_heads == 1)
      - Full MHA (n_kv_heads == n_heads)
      - Optional QK-norm
      - Optional RoPE (use_rope=False → no positional bias)
      - Optional attention projection bias
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        n_kv_heads: int | None = None,
        use_qk_norm: bool = True,
        rope_base: float = 10000.0,
        use_rope: bool = True,
        use_bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.d_k = d_model // n_heads
        self.use_qk_norm = use_qk_norm
        self.use_rope = use_rope

        q_size  = d_model
        kv_size = self.n_kv_heads * self.d_k
        o_size  = d_model

        self.q_size   = q_size
        self.kv_size  = kv_size
        self.qkv_size = q_size + 2 * kv_size

        self.qkvo_proj = nn.Parameter(torch.empty(q_size + 2 * kv_size + o_size, d_model))
        with torch.no_grad():
            torch.nn.init.normal_(self.qkvo_proj, mean=0.0, std=0.02)

        if use_bias:
            self.qkvo_bias = nn.Parameter(torch.zeros(q_size + 2 * kv_size + o_size))
        else:
            self.register_parameter("qkvo_bias", None)

        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.d_k)
            self.k_norm = nn.RMSNorm(self.d_k)

        if self.use_rope:
            self.rotary = Rotary(self.d_k, max_seq_len, base=rope_base)

        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        b = self.qkvo_bias[:self.qkv_size] if self.qkvo_bias is not None else None
        qkv = F.linear(x, self.qkvo_proj[:self.qkv_size], b)
        Q, K, V = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        V = V.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)

        if self.use_qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        if self.use_rope:
            Q = self.rotary(Q)
            K = self.rotary(K)

        if self.n_kv_heads != self.n_heads:
            K = torch.repeat_interleave(K, self.num_key_value_groups, dim=2)
            V = torch.repeat_interleave(V, self.num_key_value_groups, dim=2)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        ob = self.qkvo_bias[self.qkv_size:] if self.qkvo_bias is not None else None
        return F.linear(attn_output, self.qkvo_proj[self.qkv_size:], ob)


# ──────────────────────────────────────────────
# Transformer Blocks
# ──────────────────────────────────────────────

class TransformerBlockAblation(nn.Module):
    """
    Pre-LN transformer block (default, same as original).
    Supports all norm / FFN / attention ablation flags.
    norm_position: "pre" (default pre-LN) | "post" (post-LN) | "sandwich" (pre+post)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
        n_kv_heads: int | None = None,
        use_qk_norm: bool = True,
        activation_type: str = "squared_relu",
        rope_base: float = 10000.0,
        norm_type: str = "rmsnorm",       # "rmsnorm" | "layernorm" | "none"
        norm_position: str = "pre",       # "pre" | "post" | "sandwich"
        ffn_type: str = "standard",       # see build_ffn()
        use_rope: bool = True,
        use_bias: bool = False,
        residual_scale: float = 1.0,      # for deep-norm style scaling
    ):
        super().__init__()

        self.norm_position  = norm_position
        self.residual_scale = residual_scale

        self.attention = MultiHeadAttentionAblation(
            d_model, n_heads, max_seq_len, dropout, n_kv_heads,
            use_qk_norm, rope_base, use_rope, use_bias,
        )
        self.feed_forward = build_ffn(ffn_type, d_model, d_ff, dropout, activation_type, use_bias)

        self.norm1 = build_norm(norm_type, d_model)
        self.norm2 = build_norm(norm_type, d_model)
        # sandwich: extra post-norms
        if norm_position == "sandwich":
            self.post_norm1 = build_norm(norm_type, d_model)
            self.post_norm2 = build_norm(norm_type, d_model)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        if self.norm_position == "pre":
            attn_out = self.attention(self.norm1(x))
            x = x + self.dropout_layer(attn_out) * self.residual_scale
            ff_out = self.feed_forward(self.norm2(x))
            x = x + self.dropout_layer(ff_out) * self.residual_scale

        elif self.norm_position == "post":
            attn_out = self.attention(x)
            x = self.norm1(x + self.dropout_layer(attn_out) * self.residual_scale)
            ff_out = self.feed_forward(x)
            x = self.norm2(x + self.dropout_layer(ff_out) * self.residual_scale)

        elif self.norm_position == "sandwich":
            # Pre-norm → attn → post-norm → residual
            attn_out = self.post_norm1(self.attention(self.norm1(x)))
            x = x + self.dropout_layer(attn_out) * self.residual_scale
            ff_out = self.post_norm2(self.feed_forward(self.norm2(x)))
            x = x + self.dropout_layer(ff_out) * self.residual_scale

        return x


class ParallelTransformerBlock(nn.Module):
    """
    Parallel (PaLM-style) block: attention and FFN run in parallel on the
    pre-normed input and their outputs are summed into the residual.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
        n_kv_heads: int | None = None,
        use_qk_norm: bool = True,
        activation_type: str = "squared_relu",
        rope_base: float = 10000.0,
        norm_type: str = "rmsnorm",
        ffn_type: str = "standard",
        use_rope: bool = True,
        use_bias: bool = False,
    ):
        super().__init__()
        self.norm = build_norm(norm_type, d_model)
        self.attention = MultiHeadAttentionAblation(
            d_model, n_heads, max_seq_len, dropout, n_kv_heads,
            use_qk_norm, rope_base, use_rope, use_bias,
        )
        self.feed_forward = build_ffn(ffn_type, d_model, d_ff, dropout, activation_type, use_bias)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        normed = self.norm(x)
        return x + self.dropout_layer(self.attention(normed) + self.feed_forward(normed))
