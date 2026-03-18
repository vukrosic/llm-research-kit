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
class OneBConfig(LLMConfig):
    # ~1.03B parameters with tied embeddings in this implementation.
    d_model: int = 2048
    n_heads: int = 16
    n_layers: int = 21
    d_ff: int = 8192
    n_kv_heads: int = 4

    max_seq_len: int = 2048
    vocab_size: int = 49152

    compile_model: bool = True
    batch_size: int = 1
    gradient_accumulation_steps: int = 32
    train_tokens: int = 20_000_000_000

    muon_lr: float = 0.008
    muon_momentum: float = 0.95
    adamw_lr: float = 0.0015
    warmup_ratio: float = 0.01
    schedule_type: str = "cosine"

    eval_every: Optional[int] = 250
    eval_steps: int = 100
    eval_milestones: Optional[Tuple[int, ...]] = None

    weight_decay: float = 0.1
    dropout: float = 0.0
    grad_clip: float = 1.0
    use_amp: bool = True

    log_milestones: Tuple[int, ...] = (10, 50, 100, 250, 500, 1000)
