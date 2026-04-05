"""
Best config from H1-H9 experiments (2026-04-05).
Full attention (n_kv=8) + squared_relu + cosine LR + muon_momentum=0.90
→ 4.7952 val_loss at 8M tokens (vs 4.9214 original = −0.126 total gain).

Improvement chain:
  Original baseline:         4.9214  (squared_relu + constant LR, n_kv=4, momentum=0.95)
  + cosine LR (H6):          4.8956  (−0.026)
  + full attention (H8):     4.8785  (−0.017)
  + momentum=0.90 (H9):      4.7952  (−0.083)  ← largest single gain
"""
from dataclasses import dataclass
from configs.momentum_configs import MomentumLowConfig


@dataclass
class BestConfig(MomentumLowConfig):
    """
    Full attention (n_kv=8) + squared_relu + cosine LR + muon_momentum=0.90
    Best config at 8M tokens on TITAN X Pascal.
    val_loss=4.7952 at 8M tokens.
    """
    pass
