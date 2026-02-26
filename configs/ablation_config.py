from .llm_config import LLMConfig
from dataclasses import dataclass

@dataclass
class QKNormAblationConfig(LLMConfig):
    train_tokens: int = 100_000_000
    use_qk_norm: bool = True

@dataclass
class PerHeadScalingAblationConfig(LLMConfig):
    train_tokens: int = 100_000_000
    use_per_head_scaling: bool = True

@dataclass
class KOnlyNormAblationConfig(LLMConfig):
    train_tokens: int = 100_000_000
    qk_norm_k_only: bool = True

@dataclass
class SharedNormAblationConfig(LLMConfig):
    train_tokens: int = 100_000_000
    shared_qk_norm: bool = True

@dataclass
class QKBiasAblationConfig(LLMConfig):
    train_tokens: int = 100_000_000
    use_qk_bias: bool = True
