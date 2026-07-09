import torch
import torch.nn as nn

from .components import SquaredReLUFeedForward
from .minimax_sparse_attention import DenseGQAAttention, MiniMaxSparseAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
        n_kv_heads: int | None = None,
        attention_impl: str = "dense",
        minimax_sparse_config=None,
    ):
        super().__init__()
        if attention_impl == "dense":
            self.attention = DenseGQAAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
        elif attention_impl == "minimax_sparse":
            if minimax_sparse_config is None:
                raise ValueError("minimax_sparse_config is required")
            self.attention = MiniMaxSparseAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                max_seq_len=max_seq_len,
                block_size=minimax_sparse_config.block_size,
                top_k=minimax_sparse_config.top_k,
                index_dim=minimax_sparse_config.index_dim,
                dropout=minimax_sparse_config.dropout,
            )
        else:
            raise ValueError(f"unsupported attention_impl: {attention_impl}")

        self.feed_forward = SquaredReLUFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

