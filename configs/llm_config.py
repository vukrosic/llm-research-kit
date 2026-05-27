from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LLMConfig:
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
    
    # Logging
    log_milestones: Tuple[int, ...] = (100, 500, 1000)

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


@dataclass
class ResearchConfig(LLMConfig):
    """Legacy research preset kept for backward compatibility."""

    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 4
    d_ff: int = 1536
    n_kv_heads: int = 3
    max_seq_len: int = 1024
    train_tokens: int = 25_000_000
    activation_variant: str = "squared_relu"
    activation_slope: float = 0.5


@dataclass
class FastResearchConfig(LLMConfig):
    """A smaller weekly-paper preset for very quick smoke-to-paper runs."""

    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 1024
    n_kv_heads: int = 2
    max_seq_len: int = 512
    train_tokens: int = 1_000_000
    activation_variant: str = "squared_relu"
    activation_slope: float = 0.5


@dataclass
class FiveMillionConfig(LLMConfig):
    """~5M parameter preset for pipeline checks and first scaling-law points."""

    d_model: int = 96
    n_heads: int = 3
    n_layers: int = 4
    d_ff: int = 384
    n_kv_heads: int = 1
    max_seq_len: int = 2048
    train_tokens: int = 8_000_000
    activation_variant: str = "squared_relu"
    activation_slope: float = 0.5


@dataclass
class TwentyFiveMillionConfig(LLMConfig):
    """~25M parameter preset using 32-wide heads and a compact depth."""

    d_model: int = 384
    n_heads: int = 12
    n_layers: int = 4
    d_ff: int = 1536
    n_kv_heads: int = 6
    max_seq_len: int = 2048
    train_tokens: int = 25_000_000
    activation_variant: str = "squared_relu"
    activation_slope: float = 0.5


@dataclass
class FiftyMillionConfig(LLMConfig):
    """~50M parameter preset that keeps the 32-wide head pattern."""

    d_model: int = 512
    n_heads: int = 16
    n_layers: int = 9
    d_ff: int = 2048
    n_kv_heads: int = 8
    max_seq_len: int = 2048
    train_tokens: int = 50_000_000
    activation_variant: str = "squared_relu"
    activation_slope: float = 0.5


@dataclass
class HundredMillionConfig(LLMConfig):
    """~100M parameter preset for the main scaling-law target."""

    d_model: int = 512
    n_heads: int = 16
    n_layers: int = 26
    d_ff: int = 2048
    n_kv_heads: int = 8
    max_seq_len: int = 2048
    train_tokens: int = 100_000_000
    activation_variant: str = "squared_relu"
    activation_slope: float = 0.5
