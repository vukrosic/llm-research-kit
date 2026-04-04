"""
H2 configs: combine best activation (relu) with best LR schedule (cosine).
Expected val_loss ~6.045 (relu gain 0.049 + cosine gain 0.015 = 0.064 off baseline 6.108).
"""
from dataclasses import dataclass
from configs.titan_x_config import TitanXConfig


@dataclass
class ReluCosineConfig(TitanXConfig):
    """relu activation + 5% warmup + cosine decay. Tests additivity of both wins."""
    ffn_activation: str = "relu"
    warmup_ratio: float = 0.05
    schedule_type: str = "cosine"


@dataclass
class ReluConstantConfig(TitanXConfig):
    """relu activation + constant LR. Isolates activation benefit at this token count."""
    ffn_activation: str = "relu"
    warmup_ratio: float = 0.0
    schedule_type: str = "constant"
