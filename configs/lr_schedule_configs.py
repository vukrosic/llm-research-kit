"""
LR schedule variants for hypothesis H1: LR schedule matters for this model.

Test 1 (warmup): Does a linear warmup reduce early wasted updates?
  - TitanXWarmupConfig: constant LR + 5% warmup

Test 2 (cosine): Does cosine decay improve final convergence?
  - TitanXCosineConfig: 5% warmup + cosine decay to 10% of peak LR

Baseline: TitanXConfig — constant LR, warmup_ratio=0.0
"""
from dataclasses import dataclass
from configs.titan_x_config import TitanXConfig


@dataclass
class TitanXWarmupConfig(TitanXConfig):
    """Constant LR with 5% linear warmup. Tests whether warmup reduces
    early instability compared to the baseline (warmup_ratio=0)."""
    warmup_ratio: float = 0.05
    schedule_type: str = "constant"


@dataclass
class TitanXCosineConfig(TitanXConfig):
    """5% warmup + cosine decay to 10% of peak LR. Tests whether
    scheduled LR reduction improves final convergence."""
    warmup_ratio: float = 0.05
    schedule_type: str = "cosine"
