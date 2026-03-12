"""
Ablation study configurations — Swarm of 40 experiments.

All experiments use the same 88M parameter architecture size and the same
batch size / seq_len as the baseline for a fair token-budget comparison.

Legend of config fields added in this swarm:
  norm_type       : "rmsnorm" | "layernorm" | "none"
  norm_position   : "pre" | "post" | "sandwich"
  ffn_type        : "standard" | "swiglu" | "glu" | "bilinear" | "gated_sq_relu"
  activation_type : "squared_relu" | "gelu" | "silu" | "relu" | "tanh"
  use_rope        : bool — toggle RoPE positional encoding
  use_bias        : bool — add bias to attention + FFN projections
  parallel_block  : bool — PaLM-style parallel attn+FFN
  use_learned_pos : bool — learnable positional embeddings (instead of RoPE)
  tie_weights     : bool — tie lm_head.weight to token_embedding.weight
  init_scheme     : "default" | "depth_scaled" | "gpt2" | "small_embed"
  residual_scale  : float — multiply residual branch by this constant
  n_kv_heads      : int — GQA / MQA head count (overrides baseline 4)
  final_norm_type : "rmsnorm" | "layernorm" | "none" — override final norm
  d_ff            : int — override FFN hidden size (width ablations)
  n_layers        : int — override number of transformer layers (depth ablations)
  d_model         : int — override model dimension
  n_heads         : int — override attention head count
"""

from dataclasses import dataclass, field
from typing import Tuple
from configs.llm_config import LLMConfig


# ══════════════════════════════════════════════════════════════════════════
#  BASELINE (unchanged from previous ablation study)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class BaselineConfig(LLMConfig):
    experiment_name: str = "baseline"
    use_embed_scale: bool = True
    use_qk_norm: bool = True
    muon_ns_steps: int = 5
    activation_type: str = "squared_relu"
    rope_base: float = 10000.0
    # New fields (defaults matching original behaviour)
    norm_type: str = "rmsnorm"
    norm_position: str = "pre"
    ffn_type: str = "standard"
    use_rope: bool = True
    use_bias: bool = False
    parallel_block: bool = False
    use_learned_pos: bool = False
    tie_weights: bool = True
    init_scheme: str = "default"
    residual_scale: float = 1.0
    final_norm_type: str = "rmsnorm"


# ══════════════════════════════════════════════════════════════════════════
#  ORIGINAL ABLATIONS (kept for continuity)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class NoEmbedScaleConfig(BaselineConfig):
    experiment_name: str = "no_embed_scale"
    use_embed_scale: bool = False

@dataclass
class NoQKNormConfig(BaselineConfig):
    experiment_name: str = "no_qk_norm"
    use_qk_norm: bool = False

@dataclass
class PolarExpress2Config(BaselineConfig):
    experiment_name: str = "polar_express_2"
    muon_ns_steps: int = 2

@dataclass
class ActGELUConfig(BaselineConfig):
    experiment_name: str = "act_gelu"
    activation_type: str = "gelu"

@dataclass
class ActSiLUConfig(BaselineConfig):
    experiment_name: str = "act_silu"
    activation_type: str = "silu"

@dataclass
class RoPEBase500kConfig(BaselineConfig):
    experiment_name: str = "rope_base_500k"
    rope_base: float = 500000.0

@dataclass
class ScheduleCosineConfig(BaselineConfig):
    experiment_name: str = "schedule_cosine"
    schedule_type: str = "cosine"
    warmup_ratio: float = 0.05

@dataclass
class MuonNoMomentumConfig(BaselineConfig):
    experiment_name: str = "muon_no_momentum"
    muon_momentum: float = 0.0

@dataclass
class NoWeightDecayConfig(BaselineConfig):
    experiment_name: str = "no_weight_decay"
    weight_decay: float = 0.0

@dataclass
class HighAdamLRConfig(BaselineConfig):
    experiment_name: str = "high_adam_lr"
    adamw_lr: float = 0.012

@dataclass
class HighMuonLRConfig(BaselineConfig):
    experiment_name: str = "high_muon_lr"
    muon_lr: float = 0.048


# ══════════════════════════════════════════════════════════════════════════
#  SWARM — NORMALIZATION ABLATIONS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class PostNormConfig(BaselineConfig):
    """Post-LN (original Transformer / BERT style)."""
    experiment_name: str = "post_norm"
    norm_position: str = "post"

@dataclass
class SandwichNormConfig(BaselineConfig):
    """Sandwich norm: pre-norm + post-norm around each sub-layer."""
    experiment_name: str = "sandwich_norm"
    norm_position: str = "sandwich"

@dataclass
class LayerNormConfig(BaselineConfig):
    """LayerNorm (with learnable affine) instead of RMSNorm."""
    experiment_name: str = "layer_norm"
    norm_type: str = "layernorm"

@dataclass
class LayerNormPostConfig(BaselineConfig):
    """LayerNorm + Post-LN position (original Transformer)."""
    experiment_name: str = "layer_norm_post"
    norm_type: str = "layernorm"
    norm_position: str = "post"

@dataclass
class NoNormConfig(BaselineConfig):
    """No normalisation at all (identity norms)."""
    experiment_name: str = "no_norm"
    norm_type: str = "none"
    final_norm_type: str = "none"

@dataclass
class NoFinalNormConfig(BaselineConfig):
    """Keep pre-layer RMSNorm but remove the final pre-lm-head norm."""
    experiment_name: str = "no_final_norm"
    final_norm_type: str = "none"


# ══════════════════════════════════════════════════════════════════════════
#  SWARM — FFN ARCHITECTURE ABLATIONS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class SwiGLUConfig(BaselineConfig):
    """SwiGLU FFN (LLaMA / PaLM 2 style)."""
    experiment_name: str = "swiglu"
    ffn_type: str = "swiglu"

@dataclass
class GLUConfig(BaselineConfig):
    """Gated Linear Unit (sigmoid gate × linear branch)."""
    experiment_name: str = "glu_ffn"
    ffn_type: str = "glu"

@dataclass
class BilinearFFNConfig(BaselineConfig):
    """Bilinear FFN: no nonlinearity — pure element-wise product."""
    experiment_name: str = "bilinear_ffn"
    ffn_type: str = "bilinear"

@dataclass
class GatedSqReluConfig(BaselineConfig):
    """Gated Squared ReLU: sigmoid gate × squared_relu branch."""
    experiment_name: str = "gated_sq_relu"
    ffn_type: str = "gated_sq_relu"

@dataclass
class ActReLUConfig(BaselineConfig):
    """Plain ReLU activation (vanilla Transformer FFN)."""
    experiment_name: str = "act_relu"
    activation_type: str = "relu"

@dataclass
class FFNRatio2Config(BaselineConfig):
    """Narrow FFN: d_ff = 2× d_model (baseline is 4×)."""
    experiment_name: str = "ffn_ratio_2"
    d_ff: int = 1024   # 2 × 512

@dataclass
class FFNRatio6Config(BaselineConfig):
    """Wide FFN: d_ff = 6× d_model."""
    experiment_name: str = "ffn_ratio_6"
    d_ff: int = 3072   # 6 × 512


# ══════════════════════════════════════════════════════════════════════════
#  SWARM — ATTENTION ABLATIONS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class FullMHAConfig(BaselineConfig):
    """Full Multi-Head Attention — every head has its own KV (no GQA)."""
    experiment_name: str = "full_mha"
    n_kv_heads: int = 8   # == n_heads

@dataclass
class MQAConfig(BaselineConfig):
    """Multi-Query Attention — single shared KV head."""
    experiment_name: str = "mqa"
    n_kv_heads: int = 1

@dataclass
class NoRoPEConfig(BaselineConfig):
    """No positional encoding (attention is position-agnostic)."""
    experiment_name: str = "no_rope"
    use_rope: bool = False

@dataclass
class LearnedPosEmbedConfig(BaselineConfig):
    """Learned absolute positional embeddings instead of RoPE."""
    experiment_name: str = "learned_pos_embed"
    use_rope: bool = False
    use_learned_pos: bool = True

@dataclass
class RoPEBase1MConfig(BaselineConfig):
    """Very high RoPE base (1M) — long-context friendly."""
    experiment_name: str = "rope_base_1M"
    rope_base: float = 1_000_000.0

@dataclass
class AttnBiasConfig(BaselineConfig):
    """Add bias terms to all QKV and output projections."""
    experiment_name: str = "attn_bias"
    use_bias: bool = True


# ══════════════════════════════════════════════════════════════════════════
#  SWARM — BLOCK STRUCTURE ABLATIONS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ParallelBlockConfig(BaselineConfig):
    """PaLM-style parallel block: attn + FFN operate in parallel."""
    experiment_name: str = "parallel_block"
    parallel_block: bool = True

@dataclass
class ResidualScale05Config(BaselineConfig):
    """Scale residual branch by 0.5 (DeepNorm-lite style)."""
    experiment_name: str = "residual_scale_05"
    residual_scale: float = 0.5


# ══════════════════════════════════════════════════════════════════════════
#  SWARM — DEPTH / WIDTH ABLATIONS
#  Keep total params ≈ 88M by trading depth vs width.
#  88M ≈ 12 × (512² × 12)  for a rough guide.
#  All variants stay within ±15% of baseline parameter count.
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class DeeperNarrowerConfig(BaselineConfig):
    """More layers, smaller d_model — deeper but thinner network.
    32 layers × d_model=384 → ~81M params (≈baseline)."""
    experiment_name: str = "deeper_narrower"
    n_layers: int = 32
    d_model: int = 384
    n_heads: int = 8
    d_ff: int = 1536   # 4 × 384
    n_kv_heads: int = 4

@dataclass
class ShallowerWiderConfig(BaselineConfig):
    """Fewer layers, larger d_model — shallower but wider network.
    14 layers × d_model=576 → ~91M params (≈baseline)."""
    experiment_name: str = "shallower_wider"
    n_layers: int = 14
    d_model: int = 576
    n_heads: int = 8
    d_ff: int = 2304   # 4 × 576
    n_kv_heads: int = 4


# ══════════════════════════════════════════════════════════════════════════
#  SWARM — WEIGHT INITIALISATION ABLATIONS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class DepthScaledInitConfig(BaselineConfig):
    """Depth-scaled init: std ∝ 1/√n_layers (stabilises deep models)."""
    experiment_name: str = "depth_scaled_init"
    init_scheme: str = "depth_scaled"

@dataclass
class GPT2InitConfig(BaselineConfig):
    """GPT-2-style init: output projections scaled by 1/√(2 × n_layers)."""
    experiment_name: str = "gpt2_init"
    init_scheme: str = "gpt2"

@dataclass
class SmallEmbedInitConfig(BaselineConfig):
    """Small embedding init (std=0.002 instead of 0.02)."""
    experiment_name: str = "small_embed_init"
    init_scheme: str = "small_embed"


# ══════════════════════════════════════════════════════════════════════════
#  SWARM — WEIGHT TYING
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class NoWeightTyingConfig(BaselineConfig):
    """Separate embedding and LM head weights (no tying)."""
    experiment_name: str = "no_weight_tying"
    tie_weights: bool = False


# ══════════════════════════════════════════════════════════════════════════
#  SWARM — OPTIMIZER / SCHEDULE ABLATIONS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class MuonNS10Config(BaselineConfig):
    """Muon with 10 polar-express steps (more precise orthogonalisation)."""
    experiment_name: str = "muon_ns_10"
    muon_ns_steps: int = 10

@dataclass
class LowMuonLRConfig(BaselineConfig):
    """Half the baseline Muon LR (conservative weight update)."""
    experiment_name: str = "low_muon_lr"
    muon_lr: float = 0.012

@dataclass
class LowAdamLRConfig(BaselineConfig):
    """Half the baseline AdamW LR."""
    experiment_name: str = "low_adam_lr"
    adamw_lr: float = 0.003

@dataclass
class LinearScheduleConfig(BaselineConfig):
    """Linear LR decay instead of constant."""
    experiment_name: str = "linear_schedule"
    schedule_type: str = "linear"
    warmup_ratio: float = 0.02

@dataclass
class NoGradClipConfig(BaselineConfig):
    """Remove gradient clipping (grad_clip → very large value)."""
    experiment_name: str = "no_grad_clip"
    grad_clip: float = 1e9

@dataclass
class Dropout01Config(BaselineConfig):
    """Light dropout (p=0.1) added throughout."""
    experiment_name: str = "dropout_01"
    dropout: float = 0.1


# ══════════════════════════════════════════════════════════════════════════
#  SWARM — COMBINATION "BEST-OF" ABLATIONS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class SwiGLUPostNormConfig(BaselineConfig):
    """SwiGLU FFN + Post-LN (classical Transformer variant)."""
    experiment_name: str = "swiglu_post_norm"
    ffn_type: str = "swiglu"
    norm_position: str = "post"
    norm_type: str = "layernorm"

@dataclass
class SwiGLULayerNormConfig(BaselineConfig):
    """SwiGLU + LayerNorm (pre-LN) combo."""
    experiment_name: str = "swiglu_layernorm"
    ffn_type: str = "swiglu"
    norm_type: str = "layernorm"

@dataclass
class ParallelSwiGLUConfig(BaselineConfig):
    """Parallel block + SwiGLU (PaLM-2-lite)."""
    experiment_name: str = "parallel_swiglu"
    parallel_block: bool = True
    ffn_type: str = "swiglu"

@dataclass
class FullMHASwiGLUConfig(BaselineConfig):
    """Full MHA (no GQA) + SwiGLU."""
    experiment_name: str = "full_mha_swiglu"
    n_kv_heads: int = 8
    ffn_type: str = "swiglu"

@dataclass
class GPT2StyleConfig(BaselineConfig):
    """Approximate GPT-2 style: full MHA + GELU + LayerNorm post + GPT-2 init + bias."""
    experiment_name: str = "gpt2_style"
    n_kv_heads: int = 8        # full MHA
    activation_type: str = "gelu"
    norm_type: str = "layernorm"
    norm_position: str = "post"
    use_bias: bool = True
    init_scheme: str = "gpt2"
    use_embed_scale: bool = False


# ══════════════════════════════════════════════════════════════════════════
#  SWIGLU VARIATIONS (20 experiments)
#  All inherit from BaselineConfig; only ffn_type (and occasionally a
#  second flag) changes.
# ══════════════════════════════════════════════════════════════════════════

# ── 1. Hidden-dimension scaling ──────────────────────────────────────────

@dataclass
class SwiGLUNarrowConfig(BaselineConfig):
    """SwiGLU with hidden = d_ff × ½ — very parameter-efficient gated FFN."""
    experiment_name: str = "swiglu_narrow"
    ffn_type: str = "swiglu_narrow"

@dataclass
class SwiGLUThreeQuarterConfig(BaselineConfig):
    """SwiGLU with hidden = d_ff × ¾ — slightly wider than the 2/3 baseline."""
    experiment_name: str = "swiglu_3q"
    ffn_type: str = "swiglu_3q"

@dataclass
class SwiGLUFullWidthConfig(BaselineConfig):
    """SwiGLU with hidden = d_ff × 1 — un-compressed; ~50% more FFN params."""
    experiment_name: str = "swiglu_full"
    ffn_type: str = "swiglu_full"

@dataclass
class SwiGLUWideConfig(BaselineConfig):
    """SwiGLU with hidden = 8/3 × d_model — true LLaMA hidden-dim ratio."""
    experiment_name: str = "swiglu_wide"
    ffn_type: str = "swiglu_wide"

# ── 2. Alternative gate activations ──────────────────────────────────────

@dataclass
class GeGLUConfig(BaselineConfig):
    """GeGLU: GELU gate × linear branch (Shazeer 2020)."""
    experiment_name: str = "geglu"
    ffn_type: str = "geglu"

@dataclass
class ReGLUConfig(BaselineConfig):
    """ReGLU: ReLU gate × linear branch (sparse gating)."""
    experiment_name: str = "reglu"
    ffn_type: str = "reglu"

# ── 3. Architectural novelties inside the FFN ────────────────────────────

@dataclass
class SwiGLUDualGateConfig(BaselineConfig):
    """SwiGLU with two independent SiLU gates × linear branch."""
    experiment_name: str = "swiglu_dual_gate"
    ffn_type: str = "swiglu_dual_gate"

@dataclass
class SwiGLUResidualConfig(BaselineConfig):
    """SwiGLU + inner linear residual: gated_out + linear(x)."""
    experiment_name: str = "swiglu_residual"
    ffn_type: str = "swiglu_residual"

@dataclass
class SwiGLUSharedGateConfig(BaselineConfig):
    """SwiGLU where gate and up projections share one fused Linear (chunk)."""
    experiment_name: str = "swiglu_shared_gate"
    ffn_type: str = "swiglu_shared_gate"

@dataclass
class SwiGLUDeepConfig(BaselineConfig):
    """Two sequential SwiGLU sub-layers with residuals inside the FFN block."""
    experiment_name: str = "swiglu_deep"
    ffn_type: str = "swiglu_deep"

@dataclass
class SwiGLUSwiGLUConfig(BaselineConfig):
    """Sequential SwiGLU (no inner residual): stage-1 output feeds stage-2."""
    experiment_name: str = "swiglu_swiglu"
    ffn_type: str = "swiglu_swiglu"

@dataclass
class SwiGLUBiasConfig(BaselineConfig):
    """Standard SwiGLU (2/3 hidden) but with bias on all projections."""
    experiment_name: str = "swiglu_bias"
    ffn_type: str = "swiglu_bias"

# ── 4. SwiGLU combined with other ablation axes ──────────────────────────

@dataclass
class SwiGLUParallelConfig(BaselineConfig):
    """SwiGLU FFN inside a PaLM-style parallel block."""
    experiment_name: str = "swiglu_parallel"
    ffn_type: str = "swiglu"
    parallel_block: bool = True

@dataclass
class SwiGLUSandwichNormConfig(BaselineConfig):
    """SwiGLU + Sandwich norm (pre+post normalisation per sub-layer)."""
    experiment_name: str = "swiglu_sandwich"
    ffn_type: str = "swiglu"
    norm_position: str = "sandwich"

@dataclass
class SwiGLUMQAConfig(BaselineConfig):
    """SwiGLU + Multi-Query Attention (single KV head)."""
    experiment_name: str = "swiglu_mqa"
    ffn_type: str = "swiglu"
    n_kv_heads: int = 1

@dataclass
class SwiGLUDepthScaledInitConfig(BaselineConfig):
    """SwiGLU + depth-scaled weight init (std ∝ 1/√n_layers)."""
    experiment_name: str = "swiglu_depth_init"
    ffn_type: str = "swiglu"
    init_scheme: str = "depth_scaled"


# ══════════════════════════════════════════════════════════════════════════
#  SCISPACE-INSPIRED GLU VARIATIONS (20 experiments)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ScispaceSinGLUConfig(BaselineConfig):
    experiment_name: str = "scispace_singlu"
    ffn_type: str = "scispace_singlu"

@dataclass
class ScispaceTanhGLUConfig(BaselineConfig):
    experiment_name: str = "scispace_tanhglu"
    ffn_type: str = "scispace_tanhglu"

@dataclass
class ScispaceSigmoidGLUConfig(BaselineConfig):
    experiment_name: str = "scispace_sigmoidglu"
    ffn_type: str = "scispace_sigmoidglu"

@dataclass
class ScispaceSoftplusGLUConfig(BaselineConfig):
    experiment_name: str = "scispace_softplusglu"
    ffn_type: str = "scispace_softplusglu"

@dataclass
class ScispaceELUGLUConfig(BaselineConfig):
    experiment_name: str = "scispace_eluglu"
    ffn_type: str = "scispace_eluglu"

@dataclass
class ScispaceCELUGLUConfig(BaselineConfig):
    experiment_name: str = "scispace_celuglu"
    ffn_type: str = "scispace_celuglu"

@dataclass
class ScispaceHardswishGLUConfig(BaselineConfig):
    experiment_name: str = "scispace_hardswishglu"
    ffn_type: str = "scispace_hardswishglu"

@dataclass
class ScispaceLaplaceGLUConfig(BaselineConfig):
    experiment_name: str = "scispace_laplaceglu"
    ffn_type: str = "scispace_laplaceglu"

@dataclass
class ScispaceSinCosGLUConfig(BaselineConfig):
    experiment_name: str = "scispace_sincosglu"
    ffn_type: str = "scispace_sincosglu"

@dataclass
class ScispaceSingleProjGLUConfig(BaselineConfig):
    experiment_name: str = "scispace_singleprojglu"
    ffn_type: str = "scispace_singleprojglu"

@dataclass
class ScispaceTripleProjGLUConfig(BaselineConfig):
    experiment_name: str = "scispace_tripleprojglu"
    ffn_type: str = "scispace_tripleprojglu"

@dataclass
class ScispacePreGateGLUConfig(BaselineConfig):
    experiment_name: str = "scispace_pregatelu"
    ffn_type: str = "scispace_pregatelu"

@dataclass
class ScispacePostGateGLUConfig(BaselineConfig):
    experiment_name: str = "scispace_postgatelu"
    ffn_type: str = "scispace_postgatelu"

@dataclass
class ScispaceTopKGLUConfig(BaselineConfig):
    experiment_name: str = "scispace_topkglu"
    ffn_type: str = "scispace_topkglu"

@dataclass
class ScispaceLeakyGLUConfig(BaselineConfig):
    experiment_name: str = "scispace_leakyglu"
    ffn_type: str = "scispace_leakyglu"

@dataclass
class ScispaceAsymGLUConfig(BaselineConfig):
    experiment_name: str = "scispace_asymglu"
    ffn_type: str = "scispace_asymglu"

@dataclass
class ScispacePrenormDownConfig(BaselineConfig):
    experiment_name: str = "scispace_prenormdown"
    ffn_type: str = "scispace_prenormdown"

@dataclass
class ScispaceScaleGateConfig(BaselineConfig):
    experiment_name: str = "scispace_scalegate"
    ffn_type: str = "scispace_scalegate"

@dataclass
class ScispaceCompositeConfig(BaselineConfig):
    experiment_name: str = "scispace_composite"
    ffn_type: str = "scispace_composite"

@dataclass
class ScispaceMoELiteConfig(BaselineConfig):
    experiment_name: str = "scispace_moelite"
    ffn_type: str = "scispace_moelite"


# ══════════════════════════════════════════════════════════════════════════
#  REGISTRY
# ══════════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = {
    # ── Original (11 experiments) ──────────────────────────────────────
    "baseline":           BaselineConfig,
    "no_embed_scale":     NoEmbedScaleConfig,
    "no_qk_norm":         NoQKNormConfig,
    "polar_express_2":    PolarExpress2Config,
    "act_gelu":           ActGELUConfig,
    "act_silu":           ActSiLUConfig,
    "rope_base_500k":     RoPEBase500kConfig,
    "schedule_cosine":    ScheduleCosineConfig,
    "muon_no_momentum":   MuonNoMomentumConfig,
    "no_weight_decay":    NoWeightDecayConfig,
    "high_adam_lr":       HighAdamLRConfig,
    "high_muon_lr":       HighMuonLRConfig,

    # ── Normalization (6) ──────────────────────────────────────────────
    "post_norm":          PostNormConfig,
    "sandwich_norm":      SandwichNormConfig,
    "layer_norm":         LayerNormConfig,
    "layer_norm_post":    LayerNormPostConfig,
    "no_norm":            NoNormConfig,
    "no_final_norm":      NoFinalNormConfig,

    # ── FFN architecture (7) ───────────────────────────────────────────
    "swiglu":             SwiGLUConfig,
    "glu_ffn":            GLUConfig,
    "bilinear_ffn":       BilinearFFNConfig,
    "gated_sq_relu":      GatedSqReluConfig,
    "act_relu":           ActReLUConfig,
    "ffn_ratio_2":        FFNRatio2Config,
    "ffn_ratio_6":        FFNRatio6Config,

    # ── Attention (6) ─────────────────────────────────────────────────
    "full_mha":           FullMHAConfig,
    "mqa":                MQAConfig,
    "no_rope":            NoRoPEConfig,
    "learned_pos_embed":  LearnedPosEmbedConfig,
    "rope_base_1M":       RoPEBase1MConfig,
    "attn_bias":          AttnBiasConfig,

    # ── Block structure (2) ────────────────────────────────────────────
    "parallel_block":     ParallelBlockConfig,
    "residual_scale_05":  ResidualScale05Config,

    # ── Depth / width (2) ─────────────────────────────────────────────
    "deeper_narrower":    DeeperNarrowerConfig,
    "shallower_wider":    ShallowerWiderConfig,

    # ── Initialisation (3) ────────────────────────────────────────────
    "depth_scaled_init":  DepthScaledInitConfig,
    "gpt2_init":          GPT2InitConfig,
    "small_embed_init":   SmallEmbedInitConfig,

    # ── Weight tying (1) ──────────────────────────────────────────────
    "no_weight_tying":    NoWeightTyingConfig,

    # ── Optimizer / schedule (6) ──────────────────────────────────────
    "muon_ns_10":         MuonNS10Config,
    "low_muon_lr":        LowMuonLRConfig,
    "low_adam_lr":        LowAdamLRConfig,
    "linear_schedule":    LinearScheduleConfig,
    "no_grad_clip":       NoGradClipConfig,
    "dropout_01":         Dropout01Config,

    # ── Combo "best-of" ablations (5) ────────────────────────────────
    "swiglu_post_norm":   SwiGLUPostNormConfig,
    "swiglu_layernorm":   SwiGLULayerNormConfig,
    "parallel_swiglu":    ParallelSwiGLUConfig,
    "full_mha_swiglu":    FullMHASwiGLUConfig,
    "gpt2_style":         GPT2StyleConfig,

    # ── SwiGLU Variations (16) ────────────────────────────────────────
    # Hidden-dim scaling
    "swiglu_narrow":      SwiGLUNarrowConfig,
    "swiglu_3q":          SwiGLUThreeQuarterConfig,
    "swiglu_full":        SwiGLUFullWidthConfig,
    "swiglu_wide":        SwiGLUWideConfig,
    # Alternative gate activations
    "geglu":              GeGLUConfig,
    "reglu":              ReGLUConfig,
    # Architectural novelties
    "swiglu_dual_gate":   SwiGLUDualGateConfig,
    "swiglu_residual":    SwiGLUResidualConfig,
    "swiglu_shared_gate": SwiGLUSharedGateConfig,
    "swiglu_deep":        SwiGLUDeepConfig,
    "swiglu_swiglu":      SwiGLUSwiGLUConfig,
    "swiglu_bias":        SwiGLUBiasConfig,
    # Combined with other ablation axes
    "swiglu_parallel":    SwiGLUParallelConfig,
    "swiglu_sandwich":    SwiGLUSandwichNormConfig,
    "swiglu_mqa":         SwiGLUMQAConfig,
    "swiglu_depth_init":  SwiGLUDepthScaledInitConfig,

    # ── SciSpace-inspired Variations (20) ─────────────────────────────
    "scispace_singlu":        ScispaceSinGLUConfig,
    "scispace_tanhglu":       ScispaceTanhGLUConfig,
    "scispace_sigmoidglu":    ScispaceSigmoidGLUConfig,
    "scispace_softplusglu":   ScispaceSoftplusGLUConfig,
    "scispace_eluglu":        ScispaceELUGLUConfig,
    "scispace_celuglu":       ScispaceCELUGLUConfig,
    "scispace_hardswishglu":  ScispaceHardswishGLUConfig,
    "scispace_laplaceglu":    ScispaceLaplaceGLUConfig,
    "scispace_sincosglu":     ScispaceSinCosGLUConfig,
    "scispace_singleprojglu": ScispaceSingleProjGLUConfig,
    "scispace_tripleprojglu": ScispaceTripleProjGLUConfig,
    "scispace_pregatelu":     ScispacePreGateGLUConfig,
    "scispace_postgatelu":    ScispacePostGateGLUConfig,
    "scispace_topkglu":       ScispaceTopKGLUConfig,
    "scispace_leakyglu":      ScispaceLeakyGLUConfig,
    "scispace_asymglu":       ScispaceAsymGLUConfig,
    "scispace_prenormdown":   ScispacePrenormDownConfig,
    "scispace_scalegate":     ScispaceScaleGateConfig,
    "scispace_composite":     ScispaceCompositeConfig,
    "scispace_moelite":       ScispaceMoELiteConfig,
}

# Quick sanity check
assert len(ABLATION_CONFIGS) >= 72, f"Expected 72+ configs, got {len(ABLATION_CONFIGS)}"


def get_ablation_config(name: str, train_tokens: int = 10_000) -> LLMConfig:
    if name not in ABLATION_CONFIGS:
        raise ValueError(
            f"Unknown ablation: '{name}'.\n"
            f"Available ({len(ABLATION_CONFIGS)}): {sorted(ABLATION_CONFIGS.keys())}"
        )
    config = ABLATION_CONFIGS[name]()
    config.train_tokens = train_tokens
    return config
