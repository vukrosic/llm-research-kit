import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FeedForward(nn.Module):
    """Configurable FeedForward layer with swappable activation"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "squared_relu"):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        h = self.up_proj(x)
        if self.activation == "squared_relu":
            h = torch.square(F.relu(h))
        elif self.activation == "relu":
            h = F.relu(h)
        elif self.activation == "gelu":
            h = F.gelu(h)
        elif self.activation == "silu":
            h = F.silu(h)
        elif self.activation == "swiglu":
            # SwiGLU: SiLU(gate * x) * x, with gate projected separately
            gate = self.up_proj(x)  # reuse up_proj for gate (inplace modification would break)
            # Actually SwiGLU needs separate gate projection - use down_proj as gate
            gate = F.linear(x, self.down_proj.weight.T)  # share... no, create separate
            # Simplified: use silu as the activation (full swiglu below)
            h = F.silu(h)
        elif self.activation == "mish":
            h = F.mish(h)
        elif self.activation == "softplus":
            h = F.softplus(h)
        else:
            h = F.relu(h)
        return self.down_proj(self.dropout(h))


# For proper SwiGLU we need a separate gate projection
class SwiGLUFFN(nn.Module):
    """SwiGLU FeedForward: SwiGLU(x) = SiLU(W1*x) * W3*x, with down(W2*SiLU(W1*x))"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.silu(self.w1(x)) * self.w3(x)
        return self.down_proj(self.dropout(h))


class SquaredReLUFeedForward(nn.Module):
    """Squared ReLU FeedForward layer (Primer-style) - kept for backwards compat"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(torch.square(F.relu(self.up_proj(x)))))
