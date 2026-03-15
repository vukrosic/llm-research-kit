"""
Hybrid Architectures
=====================
Mix layers from different architecture families within a single model.

Key insight: Different sequence mixing mechanisms have complementary strengths.
Attention excels at precise retrieval, SSMs excel at continuous state tracking,
and linear attention provides efficient long-range context. Combining them
may yield something better than any single approach.

Patterns:
- Alternating: SSM, Attn, SSM, Attn, ...
- Sandwich: Attn layers in middle, SSM layers at edges
- Ratio: N SSM layers per 1 Attn layer
- Progressive: Start with local (SSM), end with global (Attn)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from frontier.architectures.base import (
    FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
)
from frontier.architectures.registry import register_arch
from frontier.architectures.state_space import SSMBlock
from frontier.architectures.retention import RetentionBlock
from frontier.architectures.linear_attention import LinearAttnBlock


class SoftmaxAttentionBlock(nn.Module):
    """Standard transformer attention block for use in hybrid architectures."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        use_bias: bool = True,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, use_bias, max_seq_len)
        self.norm2 = nn.RMSNorm(d_model)
        self.residual_scale = residual_scale

        # Gated FFN
        self.gate_proj = nn.Linear(d_model, d_ff, bias=use_bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=use_bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=use_bias)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_scale * self.attn(self.norm1(x))
        h = self.norm2(x)
        ffn_out = self.ffn_dropout(self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h)))
        x = x + self.residual_scale * ffn_out
        return x


class CausalSelfAttention(nn.Module):
    """Minimal causal self-attention for hybrid models."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 use_bias: bool = True, max_seq_len: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

        # QK Norm (proven winner from ablation system)
        self.q_norm = nn.LayerNorm(self.d_head)
        self.k_norm = nn.LayerNorm(self.d_head)

        # RoPE
        from torchtune.modules import RotaryPositionalEmbeddings
        self.rope = RotaryPositionalEmbeddings(self.d_head, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_head

        qkv = self.qkv(x).reshape(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope(q)
        k = self.rope(k)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.dropout(self.out_proj(out))


def build_hybrid_layers(
    pattern: str,
    n_layers: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    dropout: float,
    residual_scale: float,
    use_bias: bool,
    max_seq_len: int,
    ssm_config: dict,
) -> nn.ModuleList:
    """
    Build a heterogeneous stack of layers according to a pattern.

    Patterns:
    - "alternating_ssm_attn": SSM, Attn, SSM, Attn, ...
    - "ssm_heavy": 3 SSM per 1 Attn
    - "attn_sandwich": Attn at start/end, SSM in middle
    - "progressive": SSM early (local), Attn late (global)
    - "retention_attn": Alternating retention and attention
    """
    layers = nn.ModuleList()

    def make_attn():
        return SoftmaxAttentionBlock(
            d_model, n_heads, d_ff, dropout, residual_scale, use_bias, max_seq_len
        )

    def make_ssm():
        return SSMBlock(
            d_model=d_model,
            d_state=ssm_config.get("d_state", 64),
            d_conv=ssm_config.get("d_conv", 4),
            expand_factor=ssm_config.get("expand_factor", 2),
            d_ff=d_ff,
            dropout=dropout,
            residual_scale=residual_scale,
            use_bias=use_bias,
        )

    def make_retention():
        return RetentionBlock(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            dropout=dropout, residual_scale=residual_scale, use_bias=use_bias,
        )

    def make_linear_attn():
        return LinearAttnBlock(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            attn_type="gla", dropout=dropout, residual_scale=residual_scale, use_bias=use_bias,
        )

    if pattern == "alternating_ssm_attn":
        for i in range(n_layers):
            layers.append(make_ssm() if i % 2 == 0 else make_attn())

    elif pattern == "ssm_heavy":
        for i in range(n_layers):
            layers.append(make_attn() if i % 4 == 0 else make_ssm())

    elif pattern == "attn_sandwich":
        n_attn = max(2, n_layers // 4)
        n_ssm = n_layers - n_attn
        for i in range(n_attn // 2):
            layers.append(make_attn())
        for i in range(n_ssm):
            layers.append(make_ssm())
        for i in range(n_attn - n_attn // 2):
            layers.append(make_attn())

    elif pattern == "progressive":
        # SSM for first 2/3, attention for last 1/3
        cutoff = (2 * n_layers) // 3
        for i in range(n_layers):
            layers.append(make_ssm() if i < cutoff else make_attn())

    elif pattern == "retention_attn":
        for i in range(n_layers):
            layers.append(make_retention() if i % 2 == 0 else make_attn())

    elif pattern == "gla_attn":
        for i in range(n_layers):
            layers.append(make_linear_attn() if i % 2 == 0 else make_attn())

    elif pattern == "all_families":
        # Cycle through all families: SSM, Attn, Retention, GLA
        makers = [make_ssm, make_attn, make_retention, make_linear_attn]
        for i in range(n_layers):
            layers.append(makers[i % len(makers)]())

    else:
        raise ValueError(f"Unknown hybrid pattern: {pattern}")

    return layers


@register_arch("HybridLM", "hybrid", "Hybrid architecture mixing different layer types")
class HybridLM(FrontierModel):

    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config

        pattern = ac.get("pattern", "alternating_ssm_attn")
        n_heads = ac.get("n_heads", 8)
        residual_scale = ac.get("residual_scale", 1.0)
        use_bias = ac.get("use_bias", True)
        ssm_config = ac.get("ssm_config", {"d_state": 64, "d_conv": 4, "expand_factor": 2})

        self.embed = EmbeddingWithScale(config.vocab_size, config.d_model, dropout=config.dropout)
        self.blocks = build_hybrid_layers(
            pattern=pattern,
            n_layers=config.n_layers,
            d_model=config.d_model,
            n_heads=n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            residual_scale=residual_scale,
            use_bias=use_bias,
            max_seq_len=config.max_seq_len,
            ssm_config=ssm_config,
        )
        self.norm = nn.RMSNorm(config.d_model)
        self.head = LMHead(
            config.d_model, config.vocab_size,
            embedding_weight=self.embed.embedding.weight if config.tie_weights else None
        )
        self._pattern = pattern
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
        return "hybrid"

    def describe(self) -> str:
        return f"Hybrid ({self._pattern}): {self.config.n_layers}L x {self.config.d_model}d"

    def supports_recurrent_inference(self) -> bool:
        return False  # depends on pattern; conservative default

    def sequence_mixing_complexity(self) -> str:
        return "O(n) to O(n^2)"  # depends on which layers
