"""
H9: Muon momentum sweep — COMPLETE (2026-04-05)
Results: momentum=0.90 wins by -0.083 over default 0.95.
Sharp asymmetry: higher momentum catastrophically hurts (0.99 → +0.34 loss).

H10: Fine-grained momentum sweep around 0.90
Tests whether 0.85 or 0.80 continue to improve, or 0.90 is optimal.
"""
from dataclasses import dataclass
from configs.gqa_configs import FullAttentionConfig


# H9 configs (completed)
@dataclass
class MomentumLowConfig(FullAttentionConfig):
    """muon_momentum=0.90 — best from H9, val_loss=4.7952"""
    muon_momentum: float = 0.90


@dataclass
class MomentumHighConfig(FullAttentionConfig):
    """muon_momentum=0.98 — worse than default"""
    muon_momentum: float = 0.98


@dataclass
class MomentumVeryHighConfig(FullAttentionConfig):
    """muon_momentum=0.99 — catastrophically worse"""
    muon_momentum: float = 0.99


# H10 configs: fine-grained sweep below 0.90
@dataclass
class Momentum085Config(FullAttentionConfig):
    """muon_momentum=0.85 — testing if lower continues to help"""
    muon_momentum: float = 0.85


@dataclass
class Momentum080Config(FullAttentionConfig):
    """muon_momentum=0.80 — testing extreme low momentum"""
    muon_momentum: float = 0.80


@dataclass
class Momentum092Config(FullAttentionConfig):
    """muon_momentum=0.92 — narrow test between 0.90 and 0.95"""
    muon_momentum: float = 0.92
