"""
Linear Attention Architectures
===============================
Replace softmax attention with kernel-based or other linear-complexity
attention mechanisms.

Key insight: softmax attention computes Q @ K^T (n x n matrix) then multiplies
by V. Linear attention reverses the order: Q @ (K^T @ V), which is O(n*d²)
instead of O(n²*d).

Architecture variants:
- Hedgehog: Learned feature map for linear attention
- CosFormer: Cosine-based linear attention with re-weighting
- GLA: Gated Linear Attention with data-dependent decay
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from frontier.architectures.base import (
    FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
)
from frontier.architectures.registry import register_arch


# ──────────────────────────────────────────────
# Linear Attention Mechanisms
# ──────────────────────────────────────────────

class LinearAttention(nn.Module):
    """
    Basic linear attention with ELU feature map.

    Instead of softmax(QK^T)V, computes φ(Q)(φ(K)^T V) where φ = 1 + elu.
    This allows O(n) complexity by maintaining a running KV state.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        feature_map: str = "elu",  # "elu", "relu", "cosine", "hedgehog"
        use_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.feature_map_type = feature_map

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

        if feature_map == "hedgehog":
            # Learned feature map: trainable MLP that maps head_dim -> head_dim
            self.q_map = nn.Sequential(
                nn.Linear(self.d_head, self.d_head, bias=False),
                nn.ReLU(),
            )
            self.k_map = nn.Sequential(
                nn.Linear(self.d_head, self.d_head, bias=False),
                nn.ReLU(),
            )

    def _feature_map(self, x: torch.Tensor, is_query: bool = True) -> torch.Tensor:
        """Apply feature map φ to queries or keys."""
        if self.feature_map_type == "elu":
            return 1 + F.elu(x)
        elif self.feature_map_type == "relu":
            return F.relu(x)
        elif self.feature_map_type == "cosine":
            # CosFormer-style: use cos and sin of position-scaled features
            return F.relu(x)  # simplified; full cosformer adds positional re-weighting
        elif self.feature_map_type == "hedgehog":
            if is_query:
                return self.q_map(x)
            else:
                return self.k_map(x)
        else:
            return 1 + F.elu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_head

        qkv = self.qkv(x).reshape(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, L, D)

        # Apply feature maps
        q = self._feature_map(q, is_query=True)
        k = self._feature_map(k, is_query=False)

        # Causal linear attention via cumulative sum
        # For each position t: out_t = sum_{s<=t} φ(q_t)^T φ(k_s) v_s / sum_{s<=t} φ(q_t)^T φ(k_s)
        kv = torch.einsum('bhld,bhle->bhlde', k, v)  # (B, H, L, D, D)
        kv_cumsum = torch.cumsum(kv, dim=2)  # causal: only past keys
        k_cumsum = torch.cumsum(k, dim=2)  # for normalization

        # Numerator: q @ cumulative(k^T @ v)
        num = torch.einsum('bhld,bhlde->bhle', q, kv_cumsum)  # (B, H, L, D)
        # Denominator: q @ cumulative(k)
        den = torch.einsum('bhld,bhld->bhl', q, k_cumsum).unsqueeze(-1)  # (B, H, L, 1)
        den = den.clamp(min=1e-6)

        out = num / den  # (B, H, L, D)
        out = out.transpose(1, 2).reshape(B, L, self.d_model)

        return self.dropout(self.out_proj(out))


class GatedLinearAttention(nn.Module):
    """
    Gated Linear Attention (GLA).

    Adds a data-dependent decay gate to linear attention, allowing the model
    to control how much past information flows forward. This bridges
    linear attention and gated RNNs.

    gate_t = σ(W_g @ x_t)
    S_t = gate_t * S_{t-1} + k_t^T v_t
    o_t = q_t @ S_t
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
        self.gate_proj = nn.Linear(d_model, n_heads, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

        # Initialize gate bias to ~0.9 (mostly pass-through initially)
        nn.init.constant_(self.gate_proj.bias, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_head

        qkv = self.qkv(x).reshape(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, L, D)

        # Data-dependent decay gate per head
        gate = torch.sigmoid(self.gate_proj(x))  # (B, L, H)
        gate = gate.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B, H, L, 1, 1)

        # Sequential scan with gated state
        S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(L):
            kv_t = torch.einsum('bhd,bhe->bhde', k[:, :, t], v[:, :, t])
            S = gate[:, :, t] * S + kv_t
            o_t = torch.einsum('bhd,bhde->bhe', q[:, :, t], S)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=2)  # (B, H, L, D)
        out = out.transpose(1, 2).reshape(B, L, self.d_model)

        return self.dropout(self.out_proj(out))


# ──────────────────────────────────────────────
# Blocks
# ──────────────────────────────────────────────

class LinearAttnBlock(nn.Module):
    """Block: norm -> linear attention -> residual, norm -> FFN -> residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attn_type: str = "linear",  # "linear", "gla"
        feature_map: str = "elu",
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        use_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.residual_scale = residual_scale

        if attn_type == "gla":
            self.attn = GatedLinearAttention(d_model, n_heads, dropout, use_bias)
        else:
            self.attn = LinearAttention(d_model, n_heads, dropout, feature_map, use_bias)

        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=use_bias),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=use_bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_scale * self.attn(self.norm1(x))
        x = x + self.residual_scale * self.ffn(self.norm2(x))
        return x


# ──────────────────────────────────────────────
# Full Language Models
# ──────────────────────────────────────────────

@register_arch("LinearAttnLM", "linear_attention", "Linear attention LM with configurable feature maps")
class LinearAttnLM(FrontierModel):

    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config

        n_heads = ac.get("n_heads", 8)
        feature_map = ac.get("feature_map", "elu")
        residual_scale = ac.get("residual_scale", 1.0)
        use_bias = ac.get("use_bias", True)

        self.embed = EmbeddingWithScale(config.vocab_size, config.d_model, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            LinearAttnBlock(
                d_model=config.d_model,
                n_heads=n_heads,
                d_ff=config.d_ff,
                attn_type="linear",
                feature_map=feature_map,
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
        return "linear_attention"

    def describe(self) -> str:
        ac = self.config.arch_config
        return f"Linear Attention ({ac.get('feature_map', 'elu')}): {self.config.n_layers}L x {self.config.d_model}d"

    def supports_recurrent_inference(self) -> bool:
        return True

    def sequence_mixing_complexity(self) -> str:
        return "O(n)"


@register_arch("GLALM", "linear_attention", "Gated Linear Attention language model")
class GLALM(FrontierModel):

    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config

        n_heads = ac.get("n_heads", 8)
        residual_scale = ac.get("residual_scale", 1.0)
        use_bias = ac.get("use_bias", True)

        self.embed = EmbeddingWithScale(config.vocab_size, config.d_model, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            LinearAttnBlock(
                d_model=config.d_model,
                n_heads=n_heads,
                d_ff=config.d_ff,
                attn_type="gla",
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
        return "linear_attention"

    def describe(self) -> str:
        return f"GLA: {self.config.n_layers}L x {self.config.d_model}d, {self.config.arch_config.get('n_heads', 8)} heads"

    def supports_recurrent_inference(self) -> bool:
        return True

    def sequence_mixing_complexity(self) -> str:
        return "O(n)"
