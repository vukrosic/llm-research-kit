"""
RWKV Architecture
==================
Linear-complexity RNN that can be trained in parallel.

Key ideas:
- Time-mixing: exponential decay weighted attention (WKV operator)
- Channel-mixing: token shift + gated FFN
- No attention matrix — purely recurrent, O(n) training via cumsum tricks

RWKV achieves transformer-level quality with RNN-level inference cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from frontier.architectures.base import (
    FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
)
from frontier.architectures.registry import register_arch


class TokenShift(nn.Module):
    """
    RWKV's token shift: mix current token with previous token.
    x_shifted = lerp(x_{t-1}, x_t, mix_weight)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.mix = nn.Parameter(torch.ones(d_model) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shift x by 1 position (pad with zeros at start)
        x_prev = F.pad(x[:, :-1], (0, 0, 1, 0))
        mix = torch.sigmoid(self.mix)
        return x * mix + x_prev * (1 - mix)


class WKVOperator(nn.Module):
    """
    The WKV (Weighted Key-Value) operator — RWKV's core sequence mixer.

    For each position t:
        wkv_t = Σ_{s<t} exp(-(t-s)*w + k_s) * v_s + exp(u + k_t) * v_t
                ─────────────────────────────────────────────────────────
                Σ_{s<t} exp(-(t-s)*w + k_s)       + exp(u + k_t)

    w: per-channel time decay (learned)
    u: per-channel bonus for current token (learned)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Time decay: initialized to small positive values
        self.w = nn.Parameter(torch.ones(d_model) * -5.0)  # log-space, exp(w) is actual decay
        # Bonus for current position
        self.u = nn.Parameter(torch.ones(d_model) * 0.5)

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            k: (batch, seq_len, d_model)
            v: (batch, seq_len, d_model)
        Returns:
            wkv: (batch, seq_len, d_model)
        """
        B, L, D = k.shape
        w = -torch.exp(self.w)  # negative decay rate
        u = self.u

        # Sequential computation (can be parallelized with custom CUDA kernel)
        outputs = []
        # Running state: (a, b) where a = numerator accumulator, b = denominator accumulator
        a = torch.zeros(B, D, device=k.device, dtype=k.dtype)
        b = torch.zeros(B, D, device=k.device, dtype=k.dtype)
        # Maximum k seen so far (for numerical stability)
        p = torch.full((B, D), -1e38, device=k.device, dtype=k.dtype)

        for t in range(L):
            kt = k[:, t]  # (B, D)
            vt = v[:, t]  # (B, D)

            # Current token contribution (with bonus u)
            wt = u + kt
            # Past contribution decayed
            q = torch.maximum(p + w, kt)

            e1 = torch.exp(p + w - q)
            e2 = torch.exp(kt - q)

            # Update numerator and denominator
            a_new = e1 * a + e2 * vt
            b_new = e1 * b + e2

            # Output with current-token bonus
            eq = torch.maximum(p + w, wt)
            e1q = torch.exp(p + w - eq)
            e2q = torch.exp(wt - eq)
            out_t = (e1q * a + e2q * vt) / (e1q * b + e2q).clamp(min=1e-6)

            outputs.append(out_t)

            a = a_new
            b = b_new
            p = q

        return torch.stack(outputs, dim=1)


class TimeMixing(nn.Module):
    """RWKV time-mixing block: token shift -> R,K,V projections -> WKV -> output."""

    def __init__(self, d_model: int, use_bias: bool = True):
        super().__init__()
        self.shift_r = TokenShift(d_model)
        self.shift_k = TokenShift(d_model)
        self.shift_v = TokenShift(d_model)

        self.receptance = nn.Linear(d_model, d_model, bias=use_bias)
        self.key = nn.Linear(d_model, d_model, bias=use_bias)
        self.value = nn.Linear(d_model, d_model, bias=use_bias)
        self.output = nn.Linear(d_model, d_model, bias=use_bias)

        self.wkv = WKVOperator(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.receptance(self.shift_r(x)))
        k = self.key(self.shift_k(x))
        v = self.value(self.shift_v(x))

        wkv = self.wkv(k, v)
        return self.output(r * wkv)


class ChannelMixing(nn.Module):
    """RWKV channel-mixing block: token shift -> gated FFN."""

    def __init__(self, d_model: int, d_ff: int, use_bias: bool = True):
        super().__init__()
        self.shift_r = TokenShift(d_model)
        self.shift_k = TokenShift(d_model)

        self.receptance = nn.Linear(d_model, d_model, bias=use_bias)
        self.key = nn.Linear(d_model, d_ff, bias=use_bias)
        self.value = nn.Linear(d_ff, d_model, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.receptance(self.shift_r(x)))
        k = torch.square(F.relu(self.key(self.shift_k(x))))  # squared ReLU
        return r * self.value(k)


class RWKVBlock(nn.Module):
    """Single RWKV block: norm -> time_mix -> residual, norm -> channel_mix -> residual."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        use_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.time_mix = TimeMixing(d_model, use_bias)
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_mix = ChannelMixing(d_model, d_ff, use_bias)
        self.residual_scale = residual_scale
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_scale * self.dropout(self.time_mix(self.norm1(x)))
        x = x + self.residual_scale * self.dropout(self.channel_mix(self.norm2(x)))
        return x


@register_arch("RWKVLM", "rwkv", "RWKV-style linear RNN language model")
class RWKVLM(FrontierModel):

    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config

        residual_scale = ac.get("residual_scale", 1.0)
        use_bias = ac.get("use_bias", True)

        self.embed = EmbeddingWithScale(config.vocab_size, config.d_model, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            RWKVBlock(
                d_model=config.d_model,
                d_ff=config.d_ff,
                dropout=config.dropout,
                residual_scale=residual_scale,
                use_bias=use_bias,
            )
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
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
        return "rwkv"

    def describe(self) -> str:
        return f"RWKV: {self.config.n_layers}L x {self.config.d_model}d"

    def supports_recurrent_inference(self) -> bool:
        return True

    def sequence_mixing_complexity(self) -> str:
        return "O(n)"
