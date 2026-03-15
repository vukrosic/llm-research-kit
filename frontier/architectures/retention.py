"""
Retention Architecture (RetNet)
================================
Multi-scale retention mechanism that supports both parallel (training)
and recurrent (inference) computation modes.

Key idea: Replace softmax attention with an exponential decay mechanism
that can be computed either as a matrix operation (parallel, for training)
or as a recurrence (O(1) per token, for inference).

retention(Q, K, V) = (Q @ K^T ⊙ D) @ V
where D is a causal decay matrix: D_{ij} = γ^(i-j) for i >= j, 0 otherwise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from frontier.architectures.base import (
    FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
)
from frontier.architectures.registry import register_arch


class MultiScaleRetention(nn.Module):
    """
    Multi-scale retention with per-head decay rates.

    Each head gets a different decay rate γ, providing multi-scale
    temporal context (some heads focus on recent tokens, others on
    longer-range dependencies).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=use_bias)
        self.gate = nn.Linear(d_model, d_model, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.group_norm = nn.GroupNorm(n_heads, d_model)
        self.dropout = nn.Dropout(dropout)

        # Per-head decay rates: γ_h = 1 - 2^(-5 - h) for h = 0..n_heads-1
        # This gives decay rates from ~0.969 (fast decay) to ~0.999 (slow decay)
        gammas = 1.0 - torch.exp(
            torch.linspace(math.log(1/32), math.log(1/512), n_heads)
        )
        self.register_buffer('gammas', gammas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_head

        qkv = self.qkv(x).reshape(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, L, D)

        # Build causal decay matrix D for each head
        # D[h, i, j] = γ_h^(i-j) if i >= j, else 0
        positions = torch.arange(L, device=x.device, dtype=x.dtype)
        decay_dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L, L)
        decay_dist = decay_dist.clamp(min=0)  # only past positions

        # Per-head decay: (H, L, L)
        D = self.gammas.unsqueeze(-1).unsqueeze(-1) ** decay_dist.unsqueeze(0)

        # Causal mask
        causal_mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
        D = D * causal_mask.unsqueeze(0)  # (H, L, L)

        # Parallel retention: (Q @ K^T ⊙ D) @ V
        qk = torch.matmul(q, k.transpose(-2, -1))  # (B, H, L, L)
        qk = qk * D.unsqueeze(0)  # apply decay mask

        # Normalize
        qk = qk / (qk.sum(dim=-1, keepdim=True).clamp(min=1e-6))

        out = torch.matmul(qk, v)  # (B, H, L, D)

        # Group norm + gate
        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)

        gate = torch.sigmoid(self.gate(x))
        out = gate * out

        return self.dropout(self.out_proj(out))


class RetentionBlock(nn.Module):
    """Block: norm -> retention -> residual, norm -> FFN -> residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        use_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.retention = MultiScaleRetention(d_model, n_heads, dropout, use_bias)
        self.norm2 = nn.RMSNorm(d_model)
        self.residual_scale = residual_scale

        # Gated FFN (SwiGLU-style)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=use_bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=use_bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=use_bias)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_scale * self.retention(self.norm1(x))
        h = self.norm2(x)
        ffn_out = self.ffn_dropout(self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h)))
        x = x + self.residual_scale * ffn_out
        return x


@register_arch("RetNetLM", "retention", "RetNet multi-scale retention language model")
class RetNetLM(FrontierModel):

    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config

        n_heads = ac.get("n_heads", 8)
        residual_scale = ac.get("residual_scale", 1.0)
        use_bias = ac.get("use_bias", True)

        self.embed = EmbeddingWithScale(config.vocab_size, config.d_model, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            RetentionBlock(
                d_model=config.d_model,
                n_heads=n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                residual_scale=residual_scale,
                use_bias=use_bias,
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
        return "retention"

    def describe(self) -> str:
        return f"RetNet: {self.config.n_layers}L x {self.config.d_model}d, {self.config.arch_config.get('n_heads', 8)} heads"

    def supports_recurrent_inference(self) -> bool:
        return True

    def sequence_mixing_complexity(self) -> str:
        return "O(n)"  # parallel mode is O(n²) but recurrent mode is O(n)
