"""
H4: LR magnitude sweep — is Muon LR=0.024 optimal for relu+cosine?
The default LR was presumably tuned for squared_relu.
Testing half (0.012), current (0.024), and 1.5x (0.036).
AdamW LR is scaled proportionally (keeps Muon/AdamW ratio constant).
"""
from dataclasses import dataclass
from configs.best_combos import ReluCosineConfig


@dataclass
class ReluCosine_LR_Half(ReluCosineConfig):
    """relu+cosine, Muon LR=0.012 (half default), AdamW LR=0.003"""
    muon_lr: float = 0.012
    adamw_lr: float = 0.003


@dataclass
class ReluCosine_LR_Default(ReluCosineConfig):
    """relu+cosine, Muon LR=0.024 (default) — used as reference"""
    pass  # inherits defaults from ReluCosineConfig


@dataclass
class ReluCosine_LR_150pct(ReluCosineConfig):
    """relu+cosine, Muon LR=0.036 (1.5x default), AdamW LR=0.009"""
    muon_lr: float = 0.036
    adamw_lr: float = 0.009
