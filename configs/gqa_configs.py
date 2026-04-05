"""
H8: GQA ratio sweep with squared_relu+cosine at 8M tokens.
Tests whether the 8Q/4KV grouping is optimal.
All configs keep total parameter count close to 88M.

Current: n_heads=8, n_kv_heads=4 (2:1 ratio)
Test:
  - Full attention: n_kv_heads=8 (1:1 ratio, no KV sharing)
  - Aggressive GQA: n_kv_heads=2 (4:1 ratio, more KV sharing)
  - Single KV: n_kv_heads=1 (MQA, 8:1 ratio)
"""
from dataclasses import dataclass
from configs.lr_schedule_configs import TitanXCosineConfig


@dataclass
class FullAttentionConfig(TitanXCosineConfig):
    """8Q/8KV — full multi-head attention, no KV sharing."""
    n_kv_heads: int = 8


@dataclass
class AggressiveGQAConfig(TitanXCosineConfig):
    """8Q/2KV — more aggressive grouping than default 8Q/4KV."""
    n_kv_heads: int = 2


@dataclass
class MQAConfig(TitanXCosineConfig):
    """8Q/1KV — Multi-Query Attention (single KV head, maximum sharing)."""
    n_kv_heads: int = 1
