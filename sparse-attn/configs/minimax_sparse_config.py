from dataclasses import dataclass, field


@dataclass
class MiniMaxSparseConfig:
    block_size: int = 16
    top_k: int = 8
    index_dim: int | None = None
    pooling: str = "max"
    router_source: str = "separate"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.index_dim is not None and self.index_dim <= 0:
            raise ValueError("index_dim must be positive")
        if self.pooling not in {"max", "mean", "logsumexp"}:
            raise ValueError("pooling must be one of 'max', 'mean', or 'logsumexp'")
        if self.router_source not in {"separate", "group_mean_q"}:
            raise ValueError("router_source must be 'separate' or 'group_mean_q'")


@dataclass
class LLMConfig:
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 4
    n_layers: int = 22
    d_ff: int = 2048
    max_seq_len: int = 2048
    vocab_size: int = 49152
    dropout: float = 0.0
    attention_impl: str = "dense"
    minimax_sparse: MiniMaxSparseConfig = field(default_factory=MiniMaxSparseConfig)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.attention_impl not in {"dense", "minimax_sparse"}:
            raise ValueError("attention_impl must be 'dense' or 'minimax_sparse'")
        self.minimax_sparse.__post_init__()
