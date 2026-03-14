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


# ──────────────────────────────────────────────
# Gen9: Novel Bilinear-Gate FFN variants
# Current bilinear = gate(x) * up(x) (no activation on gate — pure bilinear).
# These variants apply different activations to the gate path. Never tried before.
# ──────────────────────────────────────────────

class BilinearGatedFFN(nn.Module):
    """Bilinear FFN with configurable gate activation.
    Base bilinear is: gate_proj(x) * up_proj(x) — no activation.
    This class applies a nonlinearity to the gate path: act(gate_proj(x)) * up_proj(x).
    10 novel gate activations tested in Gen9: elu, softplus, cos, abs, sqr, cubic,
    gaussian, star (StarReLU-like), mish, sq_silu.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0,
                 use_bias: bool = False, gate_type: str = 'elu'):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, hidden, bias=use_bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout   = nn.Dropout(dropout)
        self.gate_type = gate_type

    def forward(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        t = self.gate_type
        if t == 'elu':
            gate = F.elu(g)
        elif t == 'softplus':
            gate = F.softplus(g)
        elif t == 'cos':
            gate = torch.cos(g)
        elif t == 'abs':
            gate = g.abs()
        elif t == 'sqr':
            gate = g.square()
        elif t == 'cubic':
            gate = g.pow(3)
        elif t == 'gaussian':
            gate = torch.exp(-g.square().clamp(max=20))
        elif t == 'star':
            gate = g * torch.sigmoid(g)  # StarReLU: x*σ(x)
        elif t == 'mish':
            gate = g * torch.tanh(F.softplus(g))  # Mish: x*tanh(softplus(x))
        elif t == 'sq_silu':
            gate = F.silu(g).square()  # Squared SiLU
        else:
            raise ValueError(f"Unknown gate_type: {t}")
        return self.down_proj(self.dropout(gate * u))


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
    # ── Gen9: Novel bilinear gate activations (never tried before) ─────────
    elif ffn_type == "bilinear_elu":       return BilinearGatedFFN(d_model, d_ff, dropout, use_bias, gate_type='elu')
    elif ffn_type == "bilinear_softplus":  return BilinearGatedFFN(d_model, d_ff, dropout, use_bias, gate_type='softplus')
    elif ffn_type == "bilinear_cos":       return BilinearGatedFFN(d_model, d_ff, dropout, use_bias, gate_type='cos')
    elif ffn_type == "bilinear_abs":       return BilinearGatedFFN(d_model, d_ff, dropout, use_bias, gate_type='abs')
    elif ffn_type == "bilinear_sqr":       return BilinearGatedFFN(d_model, d_ff, dropout, use_bias, gate_type='sqr')
    elif ffn_type == "bilinear_cubic":     return BilinearGatedFFN(d_model, d_ff, dropout, use_bias, gate_type='cubic')
    elif ffn_type == "bilinear_gaussian":  return BilinearGatedFFN(d_model, d_ff, dropout, use_bias, gate_type='gaussian')
    elif ffn_type == "bilinear_star":      return BilinearGatedFFN(d_model, d_ff, dropout, use_bias, gate_type='star')
    elif ffn_type == "bilinear_mish":      return BilinearGatedFFN(d_model, d_ff, dropout, use_bias, gate_type='mish')
    elif ffn_type == "bilinear_sq_silu":   return BilinearGatedFFN(d_model, d_ff, dropout, use_bias, gate_type='sq_silu')
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
        qk_norm_type: str = "rmsnorm",
        use_q_norm: bool = True,
        use_k_norm: bool = True,
        attn_scale: float = 1.0,
        attn_window_size: int | None = None,
        attn_softcap: float | None = None,
        attn_activation: str = "softmax",
        use_shared_qkv: bool = False,
        hilo_fraction: float | None = None,
        kv_pool_factor: int | None = None,
        poly_order: int | None = None,
        value_norm: bool = False,
        # Gen9 novel attention mechanisms
        cosine_attn: bool = False,     # L2-normalize Q and K (cosine similarity attention)
        q_rope_only: bool = False,     # apply RoPE to Q only, not K (asymmetric positional)
        alibi: bool = False,           # ALiBi: linear position biases on logits, replaces RoPE
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.d_k = d_model // n_heads
        self.use_qk_norm = use_qk_norm
        self.use_rope = use_rope
        self.qk_norm_type = qk_norm_type
        self.use_q_norm = use_q_norm
        self.use_k_norm = use_k_norm
        self.attn_scale = attn_scale
        self.attn_window_size = attn_window_size
        self.attn_softcap = attn_softcap
        self.attn_activation = attn_activation
        self.use_shared_qkv = use_shared_qkv
        self.hilo_fraction = hilo_fraction
        self.kv_pool_factor = kv_pool_factor
        self.poly_order = poly_order
        self.value_norm = value_norm
        self.cosine_attn = cosine_attn
        self.q_rope_only = q_rope_only
        self.alibi = alibi
        if alibi:
            # Precompute ALiBi slopes: m_h = 2^(-8h/n_heads) for h=1..n_heads
            n = self.n_heads
            slopes = torch.pow(2, -(8.0 / n) * torch.arange(1, n + 1, dtype=torch.float32))
            self.register_buffer('alibi_slopes', slopes)

        if self.value_norm:
            self.v_norm = nn.RMSNorm(self.d_k)

        q_size  = d_model
        kv_size = self.n_kv_heads * self.d_k
        o_size  = d_model

        self.q_size   = q_size
        self.kv_size  = kv_size
        if self.use_shared_qkv:
            # Use a single projection for Q, K, and V
            self.qkv_size = q_size
            self.qkvo_proj = nn.Parameter(torch.empty(q_size + o_size, d_model))
        else:
            self.qkv_size = q_size + 2 * kv_size
            self.qkvo_proj = nn.Parameter(torch.empty(q_size + 2 * kv_size + o_size, d_model))
        
        with torch.no_grad():
            torch.nn.init.normal_(self.qkvo_proj, mean=0.0, std=0.02)

        if use_bias:
            bias_size = self.qkv_size + o_size
            self.qkvo_bias = nn.Parameter(torch.zeros(bias_size))
        else:
            self.register_parameter("qkvo_bias", None)

        if self.use_qk_norm:
            if self.qk_norm_type == "rmsnorm":
                if self.use_q_norm: self.q_norm = nn.RMSNorm(self.d_k)
                if self.use_k_norm: self.k_norm = nn.RMSNorm(self.d_k)
            else:
                if self.use_q_norm: self.q_norm = nn.LayerNorm(self.d_k)
                if self.use_k_norm: self.k_norm = nn.LayerNorm(self.d_k)

        if self.use_rope:
            self.rotary = Rotary(self.d_k, max_seq_len, base=rope_base)

        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        b = self.qkvo_bias[:self.qkv_size] if self.qkvo_bias is not None else None
        qkv_full = F.linear(x, self.qkvo_proj[:self.qkv_size], b)
        
        if self.use_shared_qkv:
            # tied weights for Q, K, V - all use full n_heads
            Q = K = V = qkv_full
            Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
            K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k)  # Use n_heads, not n_kv_heads
            V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k)  # Use n_heads, not n_kv_heads
        else:
            Q, K, V = qkv_full.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
            K = K.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
            V = V.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)

        if self.value_norm:
            V = self.v_norm(V)

        if self.kv_pool_factor is not None and self.kv_pool_factor > 1 and self.hilo_fraction is None:
            # Pool K and V across time dimension (Global pooling for all heads if no HiLo)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)
            K = F.avg_pool2d(K, kernel_size=(self.kv_pool_factor, 1), stride=(self.kv_pool_factor, 1))
            V = F.avg_pool2d(V, kernel_size=(self.kv_pool_factor, 1), stride=(self.kv_pool_factor, 1))
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)

        if self.use_qk_norm:
            if self.use_q_norm: Q = self.q_norm(Q)
            if self.use_k_norm: K = self.k_norm(K)

        # === cosine_attn: L2-normalize Q and K for cosine similarity attention ===
        if self.cosine_attn:
            Q = F.normalize(Q, dim=-1)
            K = F.normalize(K, dim=-1)

        if self.use_rope:
            Q = self.rotary(Q)
            if not self.q_rope_only:
                K = self.rotary(K)

        if self.n_kv_heads != self.n_heads:
            K = torch.repeat_interleave(K, self.num_key_value_groups, dim=2)
            V = torch.repeat_interleave(V, self.num_key_value_groups, dim=2)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        if self.hilo_fraction is not None:
            # HiLo: Split heads into local (high-frequency) and global (low-frequency)
            self.n_local = int(self.n_heads * self.hilo_fraction)
            # We will use windowing for high heads and pooling for low heads in the manual block

        # Check if we need manual attention
        needs_manual = (
            self.attn_activation != "softmax" or
            self.attn_softcap is not None or
            self.attn_window_size is not None or
            self.attn_scale != 1.0 or
            self.poly_order is not None or
            self.kv_pool_factor is not None or
            self.hilo_fraction is not None or
            self.alibi or
            self.cosine_attn
        )

        if not needs_manual:
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            if self.hilo_fraction is not None:
                # Split into high and low freq heads
                n_local = self.n_local
                Q_hi, K_hi, V_hi = Q[:, :n_local], K[:, :n_local], V[:, :n_local]
                Q_lo, K_lo, V_lo = Q[:, n_local:], K[:, n_local:], V[:, n_local:]

                # 1. High-frequency (Local) attention: use windowed mask
                scale = (1.0 / math.sqrt(self.d_k)) * self.attn_scale
                scores_hi = torch.matmul(Q_hi, K_hi.transpose(-2, -1)) * scale
                
                seq_len_q = scores_hi.size(-2)
                seq_len_k = scores_hi.size(-1)
                mask_hi = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=scores_hi.device).tril()
                
                # Default HiLo window is 64 if not specified
                win_size = self.attn_window_size if self.attn_window_size is not None else 64
                window_mask = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=scores_hi.device).tril(diagonal=-win_size)
                mask_hi = mask_hi & ~window_mask
                
                scores_hi = scores_hi.masked_fill(~mask_hi, float('-inf'))
                attn_weights_hi = F.softmax(scores_hi, dim=-1)
                hi_out = torch.matmul(attn_weights_hi, V_hi)

                # 2. Low-frequency (Global) attention: use pooled KV
                pool = self.kv_pool_factor if self.kv_pool_factor is not None else 4
                # We need to pool K_lo and V_lo across time (dim -2)
                # K_lo shape: (B, H_lo, T, D)
                K_lo_p = F.avg_pool2d(K_lo, kernel_size=(pool, 1), stride=(pool, 1))
                V_lo_p = F.avg_pool2d(V_lo, kernel_size=(pool, 1), stride=(pool, 1))
                
                scores_lo = torch.matmul(Q_lo, K_lo_p.transpose(-2, -1)) * scale
                # No window mask for global heads, just causal (but since KV is pooled, causal is complex)
                # For simplicity, we use the fact that it's "global" and often used in encoders,
                # but here we'll just use the standard causal mask if available or none.
                # Actually, standard HiLo usually doesn't pool for causal, but we'll follow the user's "global" intent.
                attn_weights_lo = F.softmax(scores_lo, dim=-1)
                lo_out = torch.matmul(attn_weights_lo, V_lo_p)
                
                attn_output = torch.cat([hi_out, lo_out], dim=1)
            else:
                # cosine_attn uses scale=1.0 (vectors already normalized), else standard scale
                scale = 1.0 if self.cosine_attn else (1.0 / math.sqrt(self.d_k)) * self.attn_scale
                scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

                # === ALiBi: add linear position biases per head ===
                if self.alibi:
                    seq_len_q = scores.size(-2)
                    seq_len_k = scores.size(-1)
                    pos = torch.arange(seq_len_k, device=scores.device, dtype=scores.dtype)
                    pos_q = torch.arange(seq_len_q, device=scores.device, dtype=scores.dtype)
                    # relative position bias: j - i (negative for causal, attending to past)
                    rel_pos = pos.unsqueeze(0) - pos_q.unsqueeze(1)  # (T_q, T_k)
                    rel_pos = rel_pos.unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, T_k)
                    slopes = self.alibi_slopes.view(1, self.n_heads, 1, 1)
                    scores = scores + slopes * rel_pos

                if self.attn_softcap is not None:
                    scores = self.attn_softcap * torch.tanh(scores / self.attn_softcap)

                # Causal mask + optional window mask
                seq_len_q = scores.size(-2)
                seq_len_k = scores.size(-1)
                mask = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=scores.device).tril()

                if self.attn_window_size is not None:
                    window_mask = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=scores.device).tril(diagonal=-self.attn_window_size)
                    mask = mask & ~window_mask

                scores = scores.masked_fill(~mask, float('-inf'))

                if self.poly_order is not None:
                    scores = torch.pow(F.relu(scores), self.poly_order)

                if self.attn_activation == "softmax":
                    attn_weights = F.softmax(scores, dim=-1)
                elif self.attn_activation == "relu":
                    attn_weights = F.relu(scores)
                    attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
                elif self.attn_activation == "squared_relu":
                    attn_weights = torch.square(F.relu(scores))
                    attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
                elif self.attn_activation == "gelu":
                    attn_weights = F.gelu(scores)
                    attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
                else:
                    attn_weights = F.softmax(scores, dim=-1)

                if self.dropout > 0.0 and self.training:
                    attn_weights = F.dropout(attn_weights, p=self.dropout)

                attn_output = torch.matmul(attn_weights, V)

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
        qk_norm_type: str = "rmsnorm",
        use_q_norm: bool = True,
        use_k_norm: bool = True,
        attn_scale: float = 1.0,
        attn_window_size: int | None = None,
        attn_softcap: float | None = None,
        attn_activation: str = "softmax",
        use_shared_qkv: bool = False,
        hilo_fraction: float | None = None,
        kv_pool_factor: int | None = None,
        poly_order: int | None = None,
        value_norm: bool = False,
        layer_scale_init: float | None = None,
        stochastic_depth_rate: float = 0.0,
        # Gen9 novel mechanisms
        cosine_attn: bool = False,
        q_rope_only: bool = False,
        alibi: bool = False,
        gated_residual: bool = False,
        gate_init: float = 0.0,          # sigmoid(gate_init) = starting gate value
        gate_per_channel: bool = False,   # per-feature gate vector vs scalar
    ):
        super().__init__()

        self.norm_position  = norm_position
        self.residual_scale = residual_scale
        self.stochastic_depth_rate = stochastic_depth_rate
        self.gated_residual = gated_residual

        self.attention = MultiHeadAttentionAblation(
            d_model, n_heads, max_seq_len, dropout, n_kv_heads,
            use_qk_norm, rope_base, use_rope, use_bias,
            qk_norm_type=qk_norm_type, use_q_norm=use_q_norm, use_k_norm=use_k_norm,
            attn_scale=attn_scale, attn_window_size=attn_window_size,
            attn_softcap=attn_softcap, attn_activation=attn_activation,
            use_shared_qkv=use_shared_qkv, hilo_fraction=hilo_fraction,
            kv_pool_factor=kv_pool_factor, poly_order=poly_order,
            value_norm=value_norm,
            cosine_attn=cosine_attn, q_rope_only=q_rope_only, alibi=alibi,
        )
        self.feed_forward = build_ffn(ffn_type, d_model, d_ff, dropout, activation_type, use_bias)

        self.norm1 = build_norm(norm_type, d_model)
        self.norm2 = build_norm(norm_type, d_model)
        # sandwich: extra post-norms
        if norm_position == "sandwich":
            self.post_norm1 = build_norm(norm_type, d_model)
            self.post_norm2 = build_norm(norm_type, d_model)

        self.dropout_layer = nn.Dropout(dropout)

        # LayerScale: learnable per-channel scaling of sublayer outputs
        if layer_scale_init is not None:
            self.layer_scale1 = nn.Parameter(torch.ones(d_model) * layer_scale_init)
            self.layer_scale2 = nn.Parameter(torch.ones(d_model) * layer_scale_init)
        else:
            self.layer_scale1 = None
            self.layer_scale2 = None

        # Gated residual: learned sigmoid gate controlling residual strength per block
        # gate_init: sigmoid(gate_init) is the starting gate value (0.0 → 0.5, 2.0 → 0.88)
        # gate_per_channel: learn d_model gate values instead of 1 scalar per sublayer
        if gated_residual:
            gate_shape = (d_model,) if gate_per_channel else (1,)
            self.gate_attn = nn.Parameter(torch.full(gate_shape, float(gate_init)))
            self.gate_ffn  = nn.Parameter(torch.full(gate_shape, float(gate_init)))

    def _apply_layer_scale(self, x, ls):
        if ls is not None:
            return x * ls
        return x

    def forward(self, x):
        # Stochastic depth: skip entire block during training
        if self.stochastic_depth_rate > 0.0 and self.training:
            if torch.rand(1).item() < self.stochastic_depth_rate:
                return x

        # Gated residual multipliers (sigmoid gate; 0.5 at init, learned per block)
        if self.gated_residual:
            ga = torch.sigmoid(self.gate_attn)
            gf = torch.sigmoid(self.gate_ffn)
        else:
            ga = gf = 1.0

        if self.norm_position == "pre":
            attn_out = self._apply_layer_scale(self.attention(self.norm1(x)), self.layer_scale1)
            x = x + ga * self.dropout_layer(attn_out) * self.residual_scale
            ff_out = self._apply_layer_scale(self.feed_forward(self.norm2(x)), self.layer_scale2)
            x = x + gf * self.dropout_layer(ff_out) * self.residual_scale

        elif self.norm_position == "post":
            attn_out = self._apply_layer_scale(self.attention(x), self.layer_scale1)
            x = self.norm1(x + ga * self.dropout_layer(attn_out) * self.residual_scale)
            ff_out = self._apply_layer_scale(self.feed_forward(x), self.layer_scale2)
            x = self.norm2(x + gf * self.dropout_layer(ff_out) * self.residual_scale)

        elif self.norm_position == "sandwich":
            attn_out = self.post_norm1(self.attention(self.norm1(x)))
            attn_out = self._apply_layer_scale(attn_out, self.layer_scale1)
            x = x + ga * self.dropout_layer(attn_out) * self.residual_scale
            ff_out = self.post_norm2(self.feed_forward(self.norm2(x)))
            ff_out = self._apply_layer_scale(ff_out, self.layer_scale2)
            x = x + gf * self.dropout_layer(ff_out) * self.residual_scale

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
        qk_norm_type: str = "rmsnorm",
        use_q_norm: bool = True,
        use_k_norm: bool = True,
        attn_scale: float = 1.0,
        attn_window_size: int | None = None,
        attn_softcap: float | None = None,
        attn_activation: str = "softmax",
        use_shared_qkv: bool = False,
        hilo_fraction: float | None = None,
        kv_pool_factor: int | None = None,
        poly_order: int | None = None,
        value_norm: bool = False,
        layer_scale_init: float | None = None,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()
        self.stochastic_depth_rate = stochastic_depth_rate
        self.norm = build_norm(norm_type, d_model)
        self.attention = MultiHeadAttentionAblation(
            d_model, n_heads, max_seq_len, dropout, n_kv_heads,
            use_qk_norm, rope_base, use_rope, use_bias,
            qk_norm_type=qk_norm_type, use_q_norm=use_q_norm, use_k_norm=use_k_norm,
            attn_scale=attn_scale, attn_window_size=attn_window_size,
            attn_softcap=attn_softcap, attn_activation=attn_activation,
            use_shared_qkv=use_shared_qkv, hilo_fraction=hilo_fraction,
            kv_pool_factor=kv_pool_factor, poly_order=poly_order,
            value_norm=value_norm,
        )
        self.feed_forward = build_ffn(ffn_type, d_model, d_ff, dropout, activation_type, use_bias)
        self.dropout_layer = nn.Dropout(dropout)

        if layer_scale_init is not None:
            self.layer_scale = nn.Parameter(torch.ones(d_model) * layer_scale_init)
        else:
            self.layer_scale = None

    def forward(self, x):
        if self.stochastic_depth_rate > 0.0 and self.training:
            if torch.rand(1).item() < self.stochastic_depth_rate:
                return x
        normed = self.norm(x)
        out = self.attention(normed) + self.feed_forward(normed)
        if self.layer_scale is not None:
            out = out * self.layer_scale
        return x + self.dropout_layer(out)
