from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class MiniMaxSparseConfig:
    block_size: int = 16
    top_k: int = 8
    index_dim: Optional[int] = None
    pooling: str = "max"
    router_source: str = "separate"
    dropout: float = 0.0

    def __post_init__(self):
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
    """Default legacy tuned large preset: 88,630,528 parameters."""

    # Model architecture (88M Params)
    d_model: int = 512       
    n_heads: int = 8         
    n_layers: int = 22
    d_ff: int = 2048         
    
    # GQA parameters
    n_kv_heads: int = 4      
    
    # Data params
    # ⚠️ WARNING: For simplicity, I recomend not changing max_seq_len
    # If you change max_seq_len, you MUST re-run data preparation!
    # The data preparation script chunks data at this exact length, and the RoPE
    # cache is initialized with this value. Mismatches will cause runtime errors.
    # Run: python data/prepare_mix_data.py --target_tokens 25_000_000
    # you may change the number of tokens
    max_seq_len: int = 2048  # check the warning above
    vocab_size: int = 49152  
    
    # Base Training Defaults
    device: str = "auto"  # auto, cuda, mps, or cpu
    compile_model: bool = True
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    train_tokens: int = 8000000
    
    # Learning Rate (Aggressive for pre-training)
    muon_lr: float = 0.024
    muon_momentum: float = 0.95
    adamw_lr: float = 0.006
    warmup_ratio: float = 0.0
    schedule_type: str = "constant"

    # Evaluation
    eval_every: Optional[int] = None
    eval_steps: int = 100
    eval_milestones: Optional[Tuple[int, ...]] = None
    
    # Regularization
    weight_decay: float = 0.2
    dropout: float = 0.0
    grad_clip: float = 1.0
    use_amp: bool = True
    attention_impl: str = "dense"
    minimax_sparse: MiniMaxSparseConfig = field(default_factory=MiniMaxSparseConfig)
    
    # Logging
    log_milestones: Tuple[int, ...] = (100, 500, 1000)

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        if self.attention_impl not in {"dense", "minimax_sparse"}:
            raise ValueError("attention_impl must be 'dense' or 'minimax_sparse'")
        self.minimax_sparse.__post_init__()


@dataclass
class FiveMillionConfig(LLMConfig):
    """Tiny pipeline preset: 6,652,800 parameters."""

    d_model: int = 128
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 512
    n_kv_heads: int = 1
    max_seq_len: int = 2048
    train_tokens: int = 134_000_000  # 20x params


@dataclass
class TwentyFiveMillionConfig(LLMConfig):
    """Small scaling-law preset: 25,366,272 parameters."""

    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1536
    n_kv_heads: int = 4
    max_seq_len: int = 2048
    train_tokens: int = 507_000_000  # 20x params


@dataclass
class FiftyMillionConfig(LLMConfig):
    """Mid scaling-law preset: 48,244,224 parameters."""

    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    n_kv_heads: int = 4
    max_seq_len: int = 2048
    train_tokens: int = 965_000_000  # 20x params


@dataclass
class HundredMillionConfig(LLMConfig):
    """Large scaling-law preset: 100,169,472 parameters."""

    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 26
    d_ff: int = 2048
    n_kv_heads: int = 4
    max_seq_len: int = 2048
    train_tokens: int = 2_000_000_000  # 20x params
