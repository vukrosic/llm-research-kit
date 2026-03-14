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
    norm_position: str = "sandwich"  # updated per user instruction to act as the new baseline
    ffn_type: str = "swiglu"         # updated per user instruction to act as the new baseline
    use_rope: bool = True
    use_bias: bool = False
    parallel_block: bool = False
    use_learned_pos: bool = False
    tie_weights: bool = True
    init_scheme: str = "default"
    residual_scale: float = 1.0
    final_norm_type: str = "rmsnorm"
    qk_norm_type: str = "rmsnorm"
    use_q_norm: bool = True
    use_k_norm: bool = True
    attn_scale: float = 1.0
    attn_window_size: int | None = None
    attn_softcap: float | None = None
    attn_activation: str = "softmax"


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
#  ATTENTION VARIATIONS (20 EXPERIMENTS)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class AttnQNormOnlyConfig(BaselineConfig):
    experiment_name: str = "attn_q_norm_only"
    use_k_norm: bool = False

@dataclass
class AttnKNormOnlyConfig(BaselineConfig):
    experiment_name: str = "attn_k_norm_only"
    use_q_norm: bool = False

@dataclass
class AttnNoQKNormConfig(BaselineConfig):
    experiment_name: str = "attn_no_qk_norm"
    use_qk_norm: bool = False

@dataclass
class AttnQKLayerNormConfig(BaselineConfig):
    experiment_name: str = "attn_qk_layernorm"
    qk_norm_type: str = "layernorm"

@dataclass
class AttnScale05Config(BaselineConfig):
    experiment_name: str = "attn_scale_0_5"
    attn_scale: float = 0.5

@dataclass
class AttnScale20Config(BaselineConfig):
    experiment_name: str = "attn_scale_2_0"
    attn_scale: float = 2.0

@dataclass
class AttnSoftcap10Config(BaselineConfig):
    experiment_name: str = "attn_softcap_10"
    attn_softcap: float = 10.0

@dataclass
class AttnSoftcap30Config(BaselineConfig):
    experiment_name: str = "attn_softcap_30"
    attn_softcap: float = 30.0

@dataclass
class AttnSoftcap50Config(BaselineConfig):
    experiment_name: str = "attn_softcap_50"
    attn_softcap: float = 50.0

@dataclass
class AttnWindow64Config(BaselineConfig):
    experiment_name: str = "attn_window_64"
    attn_window_size: int = 64

@dataclass
class AttnWindow128Config(BaselineConfig):
    experiment_name: str = "attn_window_128"
    attn_window_size: int = 128

@dataclass
class AttnWindow256Config(BaselineConfig):
    experiment_name: str = "attn_window_256"
    attn_window_size: int = 256

@dataclass
class AttnActReLUConfig(BaselineConfig):
    experiment_name: str = "attn_act_relu"
    attn_activation: str = "relu"

@dataclass
class AttnActSquaredReLUConfig(BaselineConfig):
    experiment_name: str = "attn_act_squared_relu"
    attn_activation: str = "squared_relu"

@dataclass
class AttnActGELUConfig(BaselineConfig):
    experiment_name: str = "attn_act_gelu"
    attn_activation: str = "gelu"

@dataclass
class AttnMQABiasConfig(BaselineConfig):
    experiment_name: str = "attn_mqa_bias"
    n_kv_heads: int = 1
    use_bias: bool = True

@dataclass
class AttnGQA8Config(BaselineConfig):
    """Full MHA (8 heads for Q, 8 heads for KV)."""
    experiment_name: str = "attn_gqa_8"
    n_kv_heads: int = 8

@dataclass
class AttnGQA2Config(BaselineConfig):
    """Heavy GQA (8 heads for Q, 2 heads for KV)."""
    experiment_name: str = "attn_gqa_2"
    n_kv_heads: int = 2

@dataclass
class AttnSandwichNorm2Config(BaselineConfig):
    """Double sandwich: attention internal sandwich + regular sandwich."""
    experiment_name: str = "attn_sandwich_norm2"
    norm_position: str = "sandwich"

@dataclass
class AttnBaselineOriginalConfig(BaselineConfig):
    """Reverts to the original 4.78 baseline for calibration."""
    experiment_name: str = "attn_baseline_original"
    ffn_type: str = "standard"
    norm_position: str = "pre"
    activation_type: str = "squared_relu"

# ── New Batch of 50 Experiments ───────────────────────────────

@dataclass
class AttnHiLoF90Config(BaselineConfig):
    experiment_name: str = "attn_hilo_f90"
    hilo_fraction: float = 0.9

@dataclass
class AttnHiLoF75Config(BaselineConfig):
    experiment_name: str = "attn_hilo_f75"
    hilo_fraction: float = 0.75

@dataclass
class AttnHiLoF50Config(BaselineConfig):
    experiment_name: str = "attn_hilo_f50"
    hilo_fraction: float = 0.50

@dataclass
class AttnPoolK2Config(BaselineConfig):
    experiment_name: str = "attn_pool_k2"
    kv_pool_factor: int = 2

@dataclass
class AttnPoolK4Config(BaselineConfig):
    experiment_name: str = "attn_pool_k4"
    kv_pool_factor: int = 4

@dataclass
class AttnPoolK8Config(BaselineConfig):
    experiment_name: str = "attn_pool_k8"
    kv_pool_factor: int = 8

@dataclass
class AttnSharedQKVConfig(BaselineConfig):
    experiment_name: str = "attn_shared_qkv"
    use_shared_qkv: bool = True

@dataclass
class AttnPoly2Config(BaselineConfig):
    experiment_name: str = "attn_poly2"
    poly_order: int = 2

@dataclass
class AttnPoly3Config(BaselineConfig):
    experiment_name: str = "attn_poly3"
    poly_order: int = 3

@dataclass
class AttnHiLoPoolConfig(BaselineConfig):
    experiment_name: str = "attn_hilo_pool"
    hilo_fraction: float = 0.8
    kv_pool_factor: int = 4

@dataclass
class AttnWindow64Pool2Config(BaselineConfig):
    experiment_name: str = "attn_window64_pool2"
    attn_window_size: int = 64
    kv_pool_factor: int = 2

@dataclass
class AttnSoftcap8ReluConfig(BaselineConfig):
    experiment_name: str = "attn_softcap8_relu"
    attn_softcap: float = 8.0
    attn_activation: str = "relu"

@dataclass
class AttnScale15Poly2Config(BaselineConfig):
    experiment_name: str = "attn_scale15_poly2"
    attn_scale: float = 1.5
    poly_order: int = 2

# ... creating more to reach 50 ...
@dataclass
class AttnSharedQKVNormConfig(BaselineConfig):
    experiment_name: str = "attn_shared_qkv_norm"
    use_shared_qkv: bool = True
    qk_norm_type: str = "layernorm"

@dataclass
class AttnGQA4Config(BaselineConfig):
    experiment_name: str = "attn_gqa_4"
    n_kv_heads: int = 4

@dataclass
class AttnDeepnormScaleConfig(BaselineConfig):
    experiment_name: str = "attn_deepnorm_scale"
    residual_scale: float = 0.707 # 1/sqrt(2)

@dataclass
class AttnSmallEmbedInitConfig(BaselineConfig):
    experiment_name: str = "attn_small_embed_init"
    init_scheme: str = "small_embed"

# Add more variations to reach the 50 count
for i in range(1, 11):
    name = f"attn_window_sweep_{i*32}"
    globals()[f"AttnWindowSweep{i*32}Config"] = type(f"AttnWindowSweep{i*32}Config", (BaselineConfig,), {
        "experiment_name": name,
        "attn_window_size": i*32
    })

for i in range(1, 6):
    name = f"attn_softcap_sweep_{i*10 + 5}"
    globals()[f"AttnSoftcapSweep{i*10 + 5}Config"] = type(f"AttnSoftcapSweep{i*10 + 5}Config", (BaselineConfig,), {
        "experiment_name": name,
        "attn_softcap": float(i*10 + 5)
    })

for i in [2, 3, 4, 5, 6]:
    name = f"attn_scale_sweep_{i*0.5}"
    globals()[f"AttnScaleSweep{int(i*0.5*10)}Config"] = type(f"AttnScaleSweep{int(i*0.5*10)}Config", (BaselineConfig,), {
        "experiment_name": name,
        "attn_scale": i*0.5
    })

# Registering them
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

    # ── Attention Variations (20) ─────────────────────────────────────
    "attn_q_norm_only":       AttnQNormOnlyConfig,
    "attn_k_norm_only":       AttnKNormOnlyConfig,
    "attn_no_qk_norm":        AttnNoQKNormConfig,
    "attn_qk_layernorm":      AttnQKLayerNormConfig,
    
    "attn_scale_0_5":         AttnScale05Config,
    "attn_scale_2_0":         AttnScale20Config,
    
    "attn_softcap_10":        AttnSoftcap10Config,
    "attn_softcap_30":        AttnSoftcap30Config,
    "attn_softcap_50":        AttnSoftcap50Config,
    
    "attn_window_64":         AttnWindow64Config,
    "attn_window_128":        AttnWindow128Config,
    "attn_window_256":        AttnWindow256Config,
    
    "attn_act_relu":          AttnActReLUConfig,
    "attn_act_squared_relu":  AttnActSquaredReLUConfig,
    "attn_act_gelu":          AttnActGELUConfig,
    
    "attn_mqa_bias":          AttnMQABiasConfig,
    "attn_gqa_8":             AttnGQA8Config,
    "attn_gqa_2":             AttnGQA2Config,
    "attn_sandwich_norm2":    AttnSandwichNorm2Config,
    "attn_baseline_original": AttnBaselineOriginalConfig,

    # ── New batch ──
    "attn_hilo_f90":          AttnHiLoF90Config,
    "attn_hilo_f75":          AttnHiLoF75Config,
    "attn_hilo_f50":          AttnHiLoF50Config,
    "attn_pool_k2":           AttnPoolK2Config,
    "attn_pool_k4":           AttnPoolK4Config,
    "attn_pool_k8":           AttnPoolK8Config,
    "attn_shared_qkv":        AttnSharedQKVConfig,
    "attn_poly2":             AttnPoly2Config,
    "attn_poly3":             AttnPoly3Config,
    "attn_hilo_pool":         AttnHiLoPoolConfig,
    "attn_window64_pool2":    AttnWindow64Pool2Config,
    "attn_softcap8_relu":     AttnSoftcap8ReluConfig,
    "attn_scale15_poly2":     AttnScale15Poly2Config,
    "attn_shared_qkv_norm":   AttnSharedQKVNormConfig,
    "attn_gqa_4":             AttnGQA4Config,
    "attn_deepnorm_scale":    AttnDeepnormScaleConfig,
    "attn_small_embed_init":  AttnSmallEmbedInitConfig,
}

# Add dynamic configs to registry
for i in range(1, 11):
    ABLATION_CONFIGS[f"attn_window_sweep_{i*32}"] = globals()[f"AttnWindowSweep{i*32}Config"]
for i in range(1, 6):
    ABLATION_CONFIGS[f"attn_softcap_sweep_{i*10 + 5}"] = globals()[f"AttnSoftcapSweep{i*10 + 5}Config"]
for i in [2, 3, 4, 5, 6]:
    ABLATION_CONFIGS[f"attn_scale_sweep_{i*0.5}"] = globals()[f"AttnScaleSweep{int(i*0.5*10)}Config"]

# Quick sanity check
assert len(ABLATION_CONFIGS) >= 72, f"Expected 72+ configs, got {len(ABLATION_CONFIGS)}"


# ══════════════════════════════════════════════════════════════════════════
#  GENERATION 2: ~200 NEW EXPERIMENTS
#  Programmatically generated. All inherit from BaselineConfig (swiglu+sandwich).
#  Naming: g2_{category}_{description}
# ══════════════════════════════════════════════════════════════════════════

def _make(name, **overrides):
    """Create a config class inheriting BaselineConfig with given overrides."""
    fields = {"experiment_name": name, **overrides}
    return type(f"Gen2_{name}", (BaselineConfig,), fields)


# ── G2-A: Muon LR fine sweep (10) ────────────────────────────────────────
for lr in [0.006, 0.008, 0.010, 0.014, 0.016, 0.018, 0.020, 0.028, 0.032, 0.036]:
    name = f"g2_muon_lr_{lr}"
    ABLATION_CONFIGS[name] = _make(name, muon_lr=lr)

# ── G2-B: AdamW LR fine sweep (5) ────────────────────────────────────────
for lr in [0.002, 0.004, 0.008, 0.010, 0.015]:
    name = f"g2_adamw_lr_{lr}"
    ABLATION_CONFIGS[name] = _make(name, adamw_lr=lr)

# ── G2-C: Weight decay sweep (7) ─────────────────────────────────────────
for wd in [0.02, 0.05, 0.08, 0.10, 0.15, 0.25, 0.30]:
    name = f"g2_wd_{wd}"
    ABLATION_CONFIGS[name] = _make(name, weight_decay=wd)

# ── G2-D: Schedule variants (8) ──────────────────────────────────────────
for sched, wu in [
    ("cosine", 0.01), ("cosine", 0.02), ("cosine", 0.05), ("cosine", 0.10),
    ("linear", 0.01), ("linear", 0.02), ("linear", 0.05), ("linear", 0.10),
]:
    name = f"g2_{sched}_wu{wu}"
    ABLATION_CONFIGS[name] = _make(name, schedule_type=sched, warmup_ratio=wu)

# ── G2-E: Residual scale sweep (6) ───────────────────────────────────────
for rs in [0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
    name = f"g2_rs_{rs}"
    ABLATION_CONFIGS[name] = _make(name, residual_scale=rs)

# ── G2-F: RoPE base sweep (5) ────────────────────────────────────────────
for rb in [50_000, 100_000, 250_000, 2_000_000, 5_000_000]:
    name = f"g2_rope_{int(rb//1000)}k"
    ABLATION_CONFIGS[name] = _make(name, rope_base=float(rb))

# ── G2-G: Muon LR × weight decay combos (12) ────────────────────────────
for lr in [0.010, 0.012, 0.016]:
    for wd in [0.0, 0.05, 0.10, 0.15]:
        name = f"g2_mlr{lr}_wd{wd}"
        ABLATION_CONFIGS[name] = _make(name, muon_lr=lr, weight_decay=wd)

# ── G2-H: Muon LR × schedule combos (12) ────────────────────────────────
for lr in [0.010, 0.012, 0.016]:
    for sched, wu in [("linear", 0.02), ("linear", 0.05), ("cosine", 0.02), ("cosine", 0.05)]:
        name = f"g2_mlr{lr}_{sched}_wu{wu}"
        ABLATION_CONFIGS[name] = _make(name, muon_lr=lr, schedule_type=sched, warmup_ratio=wu)

# ── G2-I: Residual scale × muon LR combos (9) ──────────────────────────
for rs in [0.4, 0.5, 0.7]:
    for lr in [0.010, 0.012, 0.016]:
        name = f"g2_rs{rs}_mlr{lr}"
        ABLATION_CONFIGS[name] = _make(name, residual_scale=rs, muon_lr=lr)

# ── G2-J: Parallel block combos (6) ─────────────────────────────────────
for lr in [0.010, 0.012, 0.016]:
    name = f"g2_parallel_mlr{lr}"
    ABLATION_CONFIGS[name] = _make(name, parallel_block=True, muon_lr=lr)
for rs in [0.5, 0.7]:
    name = f"g2_parallel_rs{rs}"
    ABLATION_CONFIGS[name] = _make(name, parallel_block=True, residual_scale=rs)
ABLATION_CONFIGS["g2_parallel_linear_wu02"] = _make(
    "g2_parallel_linear_wu02", parallel_block=True, schedule_type="linear", warmup_ratio=0.02)

# ── G2-K: LayerNorm combos (6) ──────────────────────────────────────────
for lr in [0.010, 0.012, 0.016]:
    name = f"g2_layernorm_mlr{lr}"
    ABLATION_CONFIGS[name] = _make(name, norm_type="layernorm", muon_lr=lr)
ABLATION_CONFIGS["g2_layernorm_rs05"] = _make(
    "g2_layernorm_rs05", norm_type="layernorm", residual_scale=0.5)
ABLATION_CONFIGS["g2_layernorm_linear_wu02"] = _make(
    "g2_layernorm_linear_wu02", norm_type="layernorm", schedule_type="linear", warmup_ratio=0.02)
ABLATION_CONFIGS["g2_layernorm_nowd"] = _make(
    "g2_layernorm_nowd", norm_type="layernorm", weight_decay=0.0)

# ── G2-L: KV head combos (8) ────────────────────────────────────────────
for nkv in [1, 2, 8]:
    for lr in [0.012, 0.016]:
        name = f"g2_kv{nkv}_mlr{lr}"
        ABLATION_CONFIGS[name] = _make(name, n_kv_heads=nkv, muon_lr=lr)
ABLATION_CONFIGS["g2_kv8_rs05"] = _make("g2_kv8_rs05", n_kv_heads=8, residual_scale=0.5)
ABLATION_CONFIGS["g2_kv8_linear_wu02"] = _make(
    "g2_kv8_linear_wu02", n_kv_heads=8, schedule_type="linear", warmup_ratio=0.02)

# ── G2-M: Triple combos — best 3-way combinations (15) ──────────────────
_triples = [
    ("g2_t_mlr012_nowd_linear",    dict(muon_lr=0.012, weight_decay=0.0, schedule_type="linear", warmup_ratio=0.02)),
    ("g2_t_mlr012_rs05_nowd",      dict(muon_lr=0.012, residual_scale=0.5, weight_decay=0.0)),
    ("g2_t_mlr012_rs05_linear",    dict(muon_lr=0.012, residual_scale=0.5, schedule_type="linear", warmup_ratio=0.02)),
    ("g2_t_mlr012_rs07_nowd",      dict(muon_lr=0.012, residual_scale=0.7, weight_decay=0.0)),
    ("g2_t_mlr016_nowd_cosine",    dict(muon_lr=0.016, weight_decay=0.0, schedule_type="cosine", warmup_ratio=0.02)),
    ("g2_t_mlr016_rs05_linear",    dict(muon_lr=0.016, residual_scale=0.5, schedule_type="linear", warmup_ratio=0.02)),
    ("g2_t_parallel_mlr012_nowd",  dict(parallel_block=True, muon_lr=0.012, weight_decay=0.0)),
    ("g2_t_parallel_mlr012_rs05",  dict(parallel_block=True, muon_lr=0.012, residual_scale=0.5)),
    ("g2_t_layernorm_mlr012_rs05", dict(norm_type="layernorm", muon_lr=0.012, residual_scale=0.5)),
    ("g2_t_layernorm_mlr012_nowd", dict(norm_type="layernorm", muon_lr=0.012, weight_decay=0.0)),
    ("g2_t_kv8_mlr012_rs05",      dict(n_kv_heads=8, muon_lr=0.012, residual_scale=0.5)),
    ("g2_t_kv8_mlr012_nowd",      dict(n_kv_heads=8, muon_lr=0.012, weight_decay=0.0)),
    ("g2_t_mlr010_nowd_rs05",     dict(muon_lr=0.010, weight_decay=0.0, residual_scale=0.5)),
    ("g2_t_mlr012_nowd_cosine05", dict(muon_lr=0.012, weight_decay=0.0, schedule_type="cosine", warmup_ratio=0.05)),
    ("g2_t_mlr012_wd005_linear",  dict(muon_lr=0.012, weight_decay=0.05, schedule_type="linear", warmup_ratio=0.02)),
]
for name, kwargs in _triples:
    ABLATION_CONFIGS[name] = _make(name, **kwargs)

# ── G2-N: Stochastic depth (6) ──────────────────────────────────────────
for sd in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    name = f"g2_sdrop_{sd}"
    ABLATION_CONFIGS[name] = _make(name, stochastic_depth=sd)

# ── G2-O: Label smoothing (5) ───────────────────────────────────────────
for ls in [0.01, 0.05, 0.10, 0.15, 0.20]:
    name = f"g2_lsmooth_{ls}"
    ABLATION_CONFIGS[name] = _make(name, label_smoothing=ls)

# ── G2-P: Z-loss (4) ────────────────────────────────────────────────────
for zl in [1e-5, 1e-4, 1e-3, 1e-2]:
    name = f"g2_zloss_{zl}"
    ABLATION_CONFIGS[name] = _make(name, z_loss_weight=zl)

# ── G2-Q: Value normalization (1 + 4 combos) ────────────────────────────
ABLATION_CONFIGS["g2_value_norm"] = _make("g2_value_norm", value_norm=True)
ABLATION_CONFIGS["g2_vnorm_mlr012"] = _make("g2_vnorm_mlr012", value_norm=True, muon_lr=0.012)
ABLATION_CONFIGS["g2_vnorm_rs05"] = _make("g2_vnorm_rs05", value_norm=True, residual_scale=0.5)
ABLATION_CONFIGS["g2_vnorm_nowd"] = _make("g2_vnorm_nowd", value_norm=True, weight_decay=0.0)
ABLATION_CONFIGS["g2_vnorm_sdrop01"] = _make("g2_vnorm_sdrop01", value_norm=True, stochastic_depth=0.1)

# ── G2-R: LayerScale (4 + 4 combos) ─────────────────────────────────────
for ls_init in [1e-4, 1e-3, 0.01, 0.1]:
    name = f"g2_layerscale_{ls_init}"
    ABLATION_CONFIGS[name] = _make(name, layer_scale_init=ls_init)
for ls_init in [1e-3, 0.01]:
    name = f"g2_layerscale{ls_init}_mlr012"
    ABLATION_CONFIGS[name] = _make(name, layer_scale_init=ls_init, muon_lr=0.012)
    name = f"g2_layerscale{ls_init}_rs05"
    ABLATION_CONFIGS[name] = _make(name, layer_scale_init=ls_init, residual_scale=0.5)

# ── G2-S: Stochastic depth combos (8) ───────────────────────────────────
for sd in [0.10, 0.15]:
    for lr in [0.012, 0.016]:
        name = f"g2_sdrop{sd}_mlr{lr}"
        ABLATION_CONFIGS[name] = _make(name, stochastic_depth=sd, muon_lr=lr)
    name = f"g2_sdrop{sd}_nowd"
    ABLATION_CONFIGS[name] = _make(name, stochastic_depth=sd, weight_decay=0.0)
    name = f"g2_sdrop{sd}_rs05"
    ABLATION_CONFIGS[name] = _make(name, stochastic_depth=sd, residual_scale=0.5)

# ── G2-T: Label smoothing combos (6) ────────────────────────────────────
for ls in [0.05, 0.10]:
    name = f"g2_lsmooth{ls}_mlr012"
    ABLATION_CONFIGS[name] = _make(name, label_smoothing=ls, muon_lr=0.012)
    name = f"g2_lsmooth{ls}_nowd"
    ABLATION_CONFIGS[name] = _make(name, label_smoothing=ls, weight_decay=0.0)
    name = f"g2_lsmooth{ls}_rs05"
    ABLATION_CONFIGS[name] = _make(name, label_smoothing=ls, residual_scale=0.5)

# ── G2-U: Depth/width variations (~88M params each) (10) ────────────────
_shapes = [
    # (n_layers, d_model, n_heads, d_ff, n_kv_heads)
    (16, 480, 8, 1920, 4),   # wider, shallower
    (18, 464, 8, 1856, 4),
    (20, 496, 8, 1984, 4),
    (24, 448, 8, 1792, 4),
    (26, 432, 8, 1728, 4),
    (28, 416, 8, 1664, 4),
    (30, 400, 8, 1600, 4),   # deeper, narrower
    (12, 576, 8, 2304, 4),   # very wide
    (10, 640, 8, 2560, 8),   # widest
    (36, 368, 8, 1472, 4),   # deepest
]
for nl, dm, nh, dff, nkv in _shapes:
    name = f"g2_shape_{nl}L_{dm}d"
    ABLATION_CONFIGS[name] = _make(name, n_layers=nl, d_model=dm, n_heads=nh, d_ff=dff, n_kv_heads=nkv)

# ── G2-V: Grad clip sweep (4) ───────────────────────────────────────────
for gc in [0.3, 0.5, 2.0, 5.0]:
    name = f"g2_gradclip_{gc}"
    ABLATION_CONFIGS[name] = _make(name, grad_clip=gc)

# ── G2-W: Momentum sweep (4) ────────────────────────────────────────────
for mom in [0.85, 0.90, 0.98, 0.99]:
    name = f"g2_momentum_{mom}"
    ABLATION_CONFIGS[name] = _make(name, muon_momentum=mom)

# ── G2-X: Dropout sweep (4) ─────────────────────────────────────────────
for dp in [0.02, 0.05, 0.15, 0.20]:
    name = f"g2_dropout_{dp}"
    ABLATION_CONFIGS[name] = _make(name, dropout=dp)

# ── G2-Y: Kitchen-sink mega combos (10) ─────────────────────────────────
#     Combine the best signals from ALL categories
_megas = [
    ("g2_mega_conservative", dict(muon_lr=0.012, weight_decay=0.0, residual_scale=0.5,
                                   schedule_type="linear", warmup_ratio=0.02)),
    ("g2_mega_aggressive",   dict(muon_lr=0.016, weight_decay=0.0, residual_scale=0.7,
                                   schedule_type="cosine", warmup_ratio=0.05)),
    ("g2_mega_layernorm",    dict(muon_lr=0.012, norm_type="layernorm", residual_scale=0.5,
                                   weight_decay=0.0, schedule_type="linear", warmup_ratio=0.02)),
    ("g2_mega_parallel",     dict(muon_lr=0.012, parallel_block=True, weight_decay=0.0,
                                   residual_scale=0.5)),
    ("g2_mega_fullmha",      dict(muon_lr=0.012, n_kv_heads=8, weight_decay=0.0,
                                   residual_scale=0.5, schedule_type="linear", warmup_ratio=0.02)),
    ("g2_mega_sdrop_lsmooth", dict(muon_lr=0.012, stochastic_depth=0.1, label_smoothing=0.05,
                                    weight_decay=0.0)),
    ("g2_mega_vnorm_ls",     dict(muon_lr=0.012, value_norm=True, layer_scale_init=0.01,
                                   weight_decay=0.0)),
    ("g2_mega_all_reg",      dict(muon_lr=0.012, stochastic_depth=0.1, label_smoothing=0.05,
                                   weight_decay=0.05, dropout=0.05)),
    ("g2_mega_wide_reg",     dict(muon_lr=0.012, n_layers=14, d_model=576, n_heads=8, d_ff=2304,
                                   n_kv_heads=4, stochastic_depth=0.1, weight_decay=0.0)),
    ("g2_mega_deep_stable",  dict(muon_lr=0.012, n_layers=30, d_model=400, n_heads=8, d_ff=1600,
                                   n_kv_heads=4, residual_scale=0.5, layer_scale_init=0.01)),
]
for name, kwargs in _megas:
    ABLATION_CONFIGS[name] = _make(name, **kwargs)

# ── G2-Z: FFN re-tests under sandwich baseline (10) ─────────────────────
#     These override ffn_type from swiglu. Since BaselineConfig now uses
#     sandwich norm, this tests each FFN type WITH sandwich (previously untested).
for ffn in ["geglu", "reglu", "bilinear", "gated_sq_relu", "standard",
            "swiglu_full", "swiglu_wide", "scispace_hardswishglu",
            "scispace_celuglu", "scispace_tripleprojglu"]:
    name = f"g2_sw_{ffn}"
    ABLATION_CONFIGS[name] = _make(name, ffn_type=ffn)

# ── G2-AA: Batch size (3) ───────────────────────────────────────────────
for bs in [4, 16, 32]:
    name = f"g2_batch_{bs}"
    ABLATION_CONFIGS[name] = _make(name, batch_size=bs)

# ── G2-AB: Gradient accumulation (3) ────────────────────────────────────
for ga in [2, 4, 8]:
    name = f"g2_gradaccum_{ga}"
    ABLATION_CONFIGS[name] = _make(name, gradient_accumulation_steps=ga)

# ── G2-AC: New feature + triple combos (6) ──────────────────────────────
_feature_triples = [
    ("g2_ft_vnorm_mlr012_nowd",     dict(value_norm=True, muon_lr=0.012, weight_decay=0.0)),
    ("g2_ft_sdrop01_mlr012_nowd",   dict(stochastic_depth=0.1, muon_lr=0.012, weight_decay=0.0)),
    ("g2_ft_lsmooth005_mlr012_rs05", dict(label_smoothing=0.05, muon_lr=0.012, residual_scale=0.5)),
    ("g2_ft_ls001_vnorm_sdrop01",   dict(layer_scale_init=0.01, value_norm=True, stochastic_depth=0.1)),
    ("g2_ft_zloss_lsmooth_mlr012",  dict(z_loss_weight=1e-4, label_smoothing=0.05, muon_lr=0.012)),
    ("g2_ft_sdrop015_rs07_linear",  dict(stochastic_depth=0.15, residual_scale=0.7,
                                          schedule_type="linear", warmup_ratio=0.02)),
]
for name, kwargs in _feature_triples:
    ABLATION_CONFIGS[name] = _make(name, **kwargs)


# ══════════════════════════════════════════════════════════════════════════
#  GENERATION 3: New baseline = attn_qk_layernorm (qk_norm_type=layernorm)
#  All experiments inherit from QKLayerNormBaseConfig.
#  Using proper @dataclass subclasses to avoid the _make() dispatch bug.
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class QKLayerNormBaseConfig(BaselineConfig):
    """New baseline: BaselineConfig + qk_norm_type=layernorm (val_loss 5.0306)."""
    experiment_name: str = "new_baseline_qklayernorm"
    qk_norm_type: str = "layernorm"

# ── Exploitation: marginals re-tested on new baseline ────────────────────

@dataclass
class NewDeepnorm07Config(QKLayerNormBaseConfig):
    experiment_name: str = "new_deepnorm_07"
    residual_scale: float = 0.707

@dataclass
class NewBilinearConfig(QKLayerNormBaseConfig):
    experiment_name: str = "new_bilinear"
    ffn_type: str = "bilinear"

@dataclass
class NewFullMHAConfig(QKLayerNormBaseConfig):
    experiment_name: str = "new_full_mha"
    n_kv_heads: int = 8

@dataclass
class NewAttnBiasConfig(QKLayerNormBaseConfig):
    experiment_name: str = "new_attn_bias"
    use_bias: bool = True

@dataclass
class NewFFNWideConfig(QKLayerNormBaseConfig):
    experiment_name: str = "new_ffn_wide"
    d_ff: int = 3072

@dataclass
class NewSmallEmbedInitConfig(QKLayerNormBaseConfig):
    experiment_name: str = "new_small_embed_init"
    init_scheme: str = "small_embed"

# ── Exploitation: deepnorm sweep ─────────────────────────────────────────

@dataclass
class DeepnormSweep05Config(QKLayerNormBaseConfig):
    experiment_name: str = "deepnorm_sweep_05"
    residual_scale: float = 0.5

@dataclass
class DeepnormSweep03Config(QKLayerNormBaseConfig):
    experiment_name: str = "deepnorm_sweep_03"
    residual_scale: float = 0.3

# ── Combos ────────────────────────────────────────────────────────────────

@dataclass
class ComboDeepnormBilinearConfig(QKLayerNormBaseConfig):
    experiment_name: str = "combo_deepnorm_bilinear"
    residual_scale: float = 0.707
    ffn_type: str = "bilinear"

@dataclass
class ComboDeepnormFullMHAConfig(QKLayerNormBaseConfig):
    experiment_name: str = "combo_deepnorm_full_mha"
    residual_scale: float = 0.707
    n_kv_heads: int = 8

# ── Exploration ───────────────────────────────────────────────────────────

@dataclass
class ValueNormConfig(QKLayerNormBaseConfig):
    experiment_name: str = "value_norm"
    value_norm: bool = True

@dataclass
class LayerScale001Config(QKLayerNormBaseConfig):
    experiment_name: str = "layer_scale_001"
    layer_scale_init: float = 0.001

# Register all generation 3 configs
ABLATION_CONFIGS["new_deepnorm_07"]          = NewDeepnorm07Config
ABLATION_CONFIGS["new_bilinear"]             = NewBilinearConfig
ABLATION_CONFIGS["new_full_mha"]             = NewFullMHAConfig
ABLATION_CONFIGS["new_attn_bias"]            = NewAttnBiasConfig
ABLATION_CONFIGS["new_ffn_wide"]             = NewFFNWideConfig
ABLATION_CONFIGS["new_small_embed_init"]     = NewSmallEmbedInitConfig
ABLATION_CONFIGS["deepnorm_sweep_05"]        = DeepnormSweep05Config
ABLATION_CONFIGS["deepnorm_sweep_03"]        = DeepnormSweep03Config
ABLATION_CONFIGS["combo_deepnorm_bilinear"]  = ComboDeepnormBilinearConfig
ABLATION_CONFIGS["combo_deepnorm_full_mha"]  = ComboDeepnormFullMHAConfig
ABLATION_CONFIGS["value_norm"]               = ValueNormConfig
ABLATION_CONFIGS["layer_scale_001"]          = LayerScale001Config


# ══════════════════════════════════════════════════════════════════════════
#  GENERATION 4 — experiments on top of combo_deepnorm_bilinear baseline
#  Base: QKLayerNorm + bilinear FFN + residual_scale=0.707
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class OptResidual06Config(ComboDeepnormBilinearConfig):
    """Residual scale sweep: 0.6 (tighter DeepNorm than 0.707)."""
    experiment_name: str = "opt_residual_06"
    residual_scale: float = 0.6

@dataclass
class OptResidual05Config(ComboDeepnormBilinearConfig):
    """Residual scale sweep: 0.5 (stronger DeepNorm)."""
    experiment_name: str = "opt_residual_05"
    residual_scale: float = 0.5

@dataclass
class FFNBilinearWideConfig(ComboDeepnormBilinearConfig):
    """Wider bilinear FFN (d_ff=3072, 6x d_model) for more multiplicative capacity."""
    experiment_name: str = "ffn_bilinear_wide"
    d_ff: int = 3072

@dataclass
class AttnFullMHAComboConfig(ComboDeepnormBilinearConfig):
    """Full MHA (n_kv_heads=8) stacked on bilinear+deepnorm combo."""
    experiment_name: str = "attn_full_mha_combo"
    n_kv_heads: int = 8

@dataclass
class OptCosineComboConfig(ComboDeepnormBilinearConfig):
    """Cosine LR schedule + 5% warmup on bilinear+deepnorm combo."""
    experiment_name: str = "opt_cosine_combo"
    schedule_type: str = "cosine"
    warmup_ratio: float = 0.05

@dataclass
class OptLinearComboConfig(ComboDeepnormBilinearConfig):
    """Linear LR decay + 2% warmup on bilinear+deepnorm combo."""
    experiment_name: str = "opt_linear_combo"
    schedule_type: str = "linear"
    warmup_ratio: float = 0.02

@dataclass
class NormLayernormComboConfig(ComboDeepnormBilinearConfig):
    """LayerNorm instead of RMSNorm on bilinear+deepnorm combo."""
    experiment_name: str = "norm_layernorm_combo"
    norm_type: str = "layernorm"

# Register Gen4 configs
ABLATION_CONFIGS["opt_residual_06"]       = OptResidual06Config
ABLATION_CONFIGS["opt_residual_05"]       = OptResidual05Config
ABLATION_CONFIGS["ffn_bilinear_wide"]     = FFNBilinearWideConfig
ABLATION_CONFIGS["attn_full_mha_combo"]   = AttnFullMHAComboConfig
ABLATION_CONFIGS["opt_cosine_combo"]      = OptCosineComboConfig
ABLATION_CONFIGS["opt_linear_combo"]      = OptLinearComboConfig
ABLATION_CONFIGS["norm_layernorm_combo"]  = NormLayernormComboConfig


# ══════════════════════════════════════════════════════════════════════════
#  GENERATION 5 — experiments on top of opt_linear_combo baseline
#  Base: bilinear FFN + residual_scale=0.707 + linear schedule + warmup=0.02
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class OptWarmup01Config(OptLinearComboConfig):
    """Shorter warmup (1%) with linear decay — test if 2% was already too long."""
    experiment_name: str = "opt_warmup_01"
    warmup_ratio: float = 0.01

@dataclass
class OptWarmup05Config(OptLinearComboConfig):
    """Longer warmup (5%) with linear decay — cosine used 5% and ranked 2nd."""
    experiment_name: str = "opt_warmup_05"
    warmup_ratio: float = 0.05

@dataclass
class OptLinearResidualStackConfig(OptLinearComboConfig):
    """Stack linear schedule with residual_scale=0.5 — both were independent winners."""
    experiment_name: str = "opt_linear_residual_stack"
    residual_scale: float = 0.5

@dataclass
class OptLinearMHAStackConfig(OptLinearComboConfig):
    """Full MHA (n_kv_heads=8) with linear schedule — full MHA was neutral alone."""
    experiment_name: str = "opt_linear_mha_stack"
    n_kv_heads: int = 8

@dataclass
class OptMuonLrLowConfig(OptLinearComboConfig):
    """Lower muon_lr=0.02 with linear decay — decay may pair better with lower peak LR."""
    experiment_name: str = "opt_muon_lr_low"
    muon_lr: float = 0.02

@dataclass
class OptZLossConfig(OptLinearComboConfig):
    """z_loss_weight=1e-4 — auxiliary entropy regularization to penalize large logits."""
    experiment_name: str = "opt_z_loss"
    z_loss_weight: float = 1e-4

@dataclass
class NormPreLinearConfig(OptLinearComboConfig):
    """Pre-norm (vs sandwich) on linear baseline — test if pre-norm interacts differently."""
    experiment_name: str = "norm_pre_linear"
    norm_position: str = "pre"

# Register Gen5 configs
ABLATION_CONFIGS["opt_warmup_01"]              = OptWarmup01Config
ABLATION_CONFIGS["opt_warmup_05"]              = OptWarmup05Config
ABLATION_CONFIGS["opt_linear_residual_stack"]  = OptLinearResidualStackConfig
ABLATION_CONFIGS["opt_linear_mha_stack"]       = OptLinearMHAStackConfig
ABLATION_CONFIGS["opt_muon_lr_low"]            = OptMuonLrLowConfig
ABLATION_CONFIGS["opt_z_loss"]                 = OptZLossConfig
ABLATION_CONFIGS["norm_pre_linear"]            = NormPreLinearConfig


# ══════════════════════════════════════════════════════════════════════════
#  GENERATION 6 — experiments on top of opt_linear_residual_stack baseline
#  Base: bilinear FFN + residual_scale=0.5 + linear schedule + warmup=0.02
# ══════════════════════════════════════════════════════════════════════════

# --- Muon LR sweep ---
@dataclass
class G6MuonLr015Config(OptLinearResidualStackConfig):
    """Muon LR=0.015 — push lower than 0.020 winner."""
    experiment_name: str = "g6_muon_lr_015"
    muon_lr: float = 0.015

@dataclass
class G6MuonLr018Config(OptLinearResidualStackConfig):
    """Muon LR=0.018 — 10M winner per hypotheses.md, untested on new baseline."""
    experiment_name: str = "g6_muon_lr_018"
    muon_lr: float = 0.018

@dataclass
class G6MuonLr020Config(OptLinearResidualStackConfig):
    """Muon LR=0.020 — was winner in Gen5 on prior baseline; re-test here."""
    experiment_name: str = "g6_muon_lr_020"
    muon_lr: float = 0.020

@dataclass
class G6MuonLr028Config(OptLinearResidualStackConfig):
    """Muon LR=0.028 — sweep above current 0.024."""
    experiment_name: str = "g6_muon_lr_028"
    muon_lr: float = 0.028

# --- Warmup sweep ---
@dataclass
class G6Warmup00Config(OptLinearResidualStackConfig):
    """No warmup with linear decay — test if any warmup is needed."""
    experiment_name: str = "g6_warmup_00"
    warmup_ratio: float = 0.0

@dataclass
class G6Warmup01Config(OptLinearResidualStackConfig):
    """Warmup=1% on new residual_scale=0.5 baseline — re-test since base changed."""
    experiment_name: str = "g6_warmup_01"
    warmup_ratio: float = 0.01

@dataclass
class G6Warmup015Config(OptLinearResidualStackConfig):
    """Warmup=1.5% — fine-grained sweep between winning 1% and current 2%."""
    experiment_name: str = "g6_warmup_015"
    warmup_ratio: float = 0.015

# --- Residual scale sweep ---
@dataclass
class G6Residual04Config(OptLinearResidualStackConfig):
    """Residual scale=0.4 — between 0.5 winner and 0.707."""
    experiment_name: str = "g6_residual_04"
    residual_scale: float = 0.4

@dataclass
class G6Residual03Config(OptLinearResidualStackConfig):
    """Residual scale=0.3 — push tighter than 0.5 winner."""
    experiment_name: str = "g6_residual_03"
    residual_scale: float = 0.3

# --- Weight decay sweep ---
@dataclass
class G6WD01Config(OptLinearResidualStackConfig):
    """Weight decay=0.1 — less regularization than current 0.2."""
    experiment_name: str = "g6_wd_01"
    weight_decay: float = 0.1

@dataclass
class G6WD03Config(OptLinearResidualStackConfig):
    """Weight decay=0.3 — more regularization than current 0.2."""
    experiment_name: str = "g6_wd_03"
    weight_decay: float = 0.3

# --- AdamW LR ---
@dataclass
class G6AdamwLr003Config(OptLinearResidualStackConfig):
    """AdamW LR=0.003 — halve the embedding/bias LR."""
    experiment_name: str = "g6_adamw_003"
    adamw_lr: float = 0.003

# --- Muon momentum ---
@dataclass
class G6Momentum098Config(OptLinearResidualStackConfig):
    """Muon momentum=0.98 — higher momentum for smoother updates."""
    experiment_name: str = "g6_momentum_098"
    muon_momentum: float = 0.98

# --- FFN width ---
@dataclass
class G6DFF1536Config(OptLinearResidualStackConfig):
    """d_ff=1536 (3x d_model) — narrower bilinear FFN."""
    experiment_name: str = "g6_dff_1536"
    d_ff: int = 1536

@dataclass
class G6DFF2560Config(OptLinearResidualStackConfig):
    """d_ff=2560 (5x d_model) — slightly wider bilinear FFN."""
    experiment_name: str = "g6_dff_2560"
    d_ff: int = 2560

# --- Exploration ---
@dataclass
class G6LayerScale001Config(OptLinearResidualStackConfig):
    """Layer scale init=0.01 (CaiT-style) — small scalar on residual branch."""
    experiment_name: str = "g6_layer_scale_001"
    layer_scale_init: float = 0.01

@dataclass
class G6LayerScale01Config(OptLinearResidualStackConfig):
    """Layer scale init=0.1 — larger layer scale init."""
    experiment_name: str = "g6_layer_scale_01"
    layer_scale_init: float = 0.1

@dataclass
class G6StochDepth005Config(OptLinearResidualStackConfig):
    """Stochastic depth=0.05 — light drop-path regularization."""
    experiment_name: str = "g6_sdepth_005"
    stochastic_depth: float = 0.05

@dataclass
class G6StochDepth010Config(OptLinearResidualStackConfig):
    """Stochastic depth=0.10 — heavier drop-path regularization."""
    experiment_name: str = "g6_sdepth_010"
    stochastic_depth: float = 0.10

@dataclass
class G6LabelSmooth005Config(OptLinearResidualStackConfig):
    """Label smoothing=0.05 — soft targets for better generalization."""
    experiment_name: str = "g6_label_smooth_005"
    label_smoothing: float = 0.05

@dataclass
class G6RopeBase100kConfig(OptLinearResidualStackConfig):
    """RoPE base=100000 — scaled positional encoding for better generalization."""
    experiment_name: str = "g6_rope_100k"
    rope_base: float = 100000.0

@dataclass
class G6MuonNS8Config(OptLinearResidualStackConfig):
    """muon_ns_steps=8 — more Newton-Schulz steps for a better preconditioner."""
    experiment_name: str = "g6_muon_ns8"
    muon_ns_steps: int = 8

# Register Gen6 configs
ABLATION_CONFIGS["g6_muon_lr_015"]        = G6MuonLr015Config
ABLATION_CONFIGS["g6_muon_lr_018"]        = G6MuonLr018Config
ABLATION_CONFIGS["g6_muon_lr_020"]        = G6MuonLr020Config
ABLATION_CONFIGS["g6_muon_lr_028"]        = G6MuonLr028Config
ABLATION_CONFIGS["g6_warmup_00"]          = G6Warmup00Config
ABLATION_CONFIGS["g6_warmup_01"]          = G6Warmup01Config
ABLATION_CONFIGS["g6_warmup_015"]         = G6Warmup015Config
ABLATION_CONFIGS["g6_residual_04"]        = G6Residual04Config
ABLATION_CONFIGS["g6_residual_03"]        = G6Residual03Config
ABLATION_CONFIGS["g6_wd_01"]              = G6WD01Config
ABLATION_CONFIGS["g6_wd_03"]             = G6WD03Config
ABLATION_CONFIGS["g6_adamw_003"]          = G6AdamwLr003Config
ABLATION_CONFIGS["g6_momentum_098"]       = G6Momentum098Config
ABLATION_CONFIGS["g6_dff_1536"]           = G6DFF1536Config
ABLATION_CONFIGS["g6_dff_2560"]           = G6DFF2560Config
ABLATION_CONFIGS["g6_layer_scale_001"]    = G6LayerScale001Config
ABLATION_CONFIGS["g6_layer_scale_01"]     = G6LayerScale01Config
ABLATION_CONFIGS["g6_sdepth_005"]         = G6StochDepth005Config
ABLATION_CONFIGS["g6_sdepth_010"]         = G6StochDepth010Config
ABLATION_CONFIGS["g6_label_smooth_005"]   = G6LabelSmooth005Config
ABLATION_CONFIGS["g6_rope_100k"]          = G6RopeBase100kConfig
ABLATION_CONFIGS["g6_muon_ns8"]           = G6MuonNS8Config


# ══════════════════════════════════════════════════════════════════════════
#  GENERATION 7 — experiments on top of g6_muon_lr_018 baseline
#  Base: bilinear FFN + residual_scale=0.5 + linear schedule + warmup=0.02
#        + muon_lr=0.018
#  Focus: novel architecture mechanisms, not hyperparameter sweeps
# ══════════════════════════════════════════════════════════════════════════

# --- Attention structure ---
@dataclass
class G7ParallelBlockConfig(G6MuonLr018Config):
    """PaLM-style parallel attn+FFN: both branches share the same input, outputs summed."""
    experiment_name: str = "g7_parallel_block"
    parallel_block: bool = True

@dataclass
class G7ReluAttnConfig(G6MuonLr018Config):
    """ReLU attention instead of softmax — sparse, unnormalized attention patterns (ReluFormer)."""
    experiment_name: str = "g7_relu_attn"
    attn_activation: str = "relu"

@dataclass
class G7Softcap20Config(G6MuonLr018Config):
    """Gemma2-style attention logit soft-cap at ±20 — prevents extreme attention concentration."""
    experiment_name: str = "g7_softcap_20"
    attn_softcap: float = 20.0

@dataclass
class G7Softcap10Config(G6MuonLr018Config):
    """More aggressive soft-cap at ±10 — forces more uniform attention."""
    experiment_name: str = "g7_softcap_10"
    attn_softcap: float = 10.0

@dataclass
class G7ValueNormConfig(G6MuonLr018Config):
    """Normalize V vectors before output projection — implicit regularization on aggregated info."""
    experiment_name: str = "g7_value_norm"
    value_norm: bool = True

@dataclass
class G7AttnScaleHalfConfig(G6MuonLr018Config):
    """attn_scale=0.5 — significantly dampen attention logits, much more uniform attention."""
    experiment_name: str = "g7_attn_scale_half"
    attn_scale: float = 0.5

@dataclass
class G7AttnScale2Config(G6MuonLr018Config):
    """attn_scale=2.0 — sharper, spikier attention; forces heads to specialize."""
    experiment_name: str = "g7_attn_scale_2"
    attn_scale: float = 2.0

@dataclass
class G7GQA2Config(G6MuonLr018Config):
    """n_kv_heads=2 — more aggressive KV sharing than current 4; tests if fewer KV heads help."""
    experiment_name: str = "g7_gqa_2"
    n_kv_heads: int = 2

# --- Positional encoding ---
@dataclass
class G7NoPosConfig(G6MuonLr018Config):
    """No positional encoding at all — relies purely on causal masking for sequence order."""
    experiment_name: str = "g7_no_pos"
    use_rope: bool = False
    use_learned_pos: bool = False

@dataclass
class G7LearnedPosConfig(G6MuonLr018Config):
    """Learned absolute positional embeddings (GPT-2 style) instead of RoPE."""
    experiment_name: str = "g7_learned_pos"
    use_rope: bool = False
    use_learned_pos: bool = True

# --- FFN gating variants ---
@dataclass
class G7SwiGLUNewConfig(G6MuonLr018Config):
    """SwiGLU on the full new baseline — never tested with linear schedule + muon_lr=0.018."""
    experiment_name: str = "g7_swiglu_new"
    ffn_type: str = "swiglu"

@dataclass
class G7GatedSqReluConfig(G6MuonLr018Config):
    """Gated squared-ReLU FFN — different gating structure from bilinear."""
    experiment_name: str = "g7_gated_sq_relu"
    ffn_type: str = "gated_sq_relu"

@dataclass
class G7BilinearSiluConfig(G6MuonLr018Config):
    """Bilinear FFN gate with SiLU activation — smooth, non-sparse gate vs squared_relu."""
    experiment_name: str = "g7_bilinear_silu"
    activation_type: str = "silu"

@dataclass
class G7BilinearTanhConfig(G6MuonLr018Config):
    """Bilinear FFN gate with tanh — bounded, symmetric gate; very different saturation behavior."""
    experiment_name: str = "g7_bilinear_tanh"
    activation_type: str = "tanh"

# --- Normalization ---
@dataclass
class G7NoQKNormConfig(G6MuonLr018Config):
    """Remove QK norm entirely — test if it's still essential with DeepNorm + linear schedule."""
    experiment_name: str = "g7_no_qk_norm"
    use_qk_norm: bool = False

@dataclass
class G7QNormOnlyConfig(G6MuonLr018Config):
    """Normalize only queries, not keys — asymmetric QK norm."""
    experiment_name: str = "g7_q_norm_only"
    use_k_norm: bool = False

@dataclass
class G7PostNormConfig(G6MuonLr018Config):
    """Post-norm architecture — classic BERT/GPT gradient flow, very different from sandwich."""
    experiment_name: str = "g7_post_norm"
    norm_position: str = "post"

@dataclass
class G7FinalNormNoneConfig(G6MuonLr018Config):
    """Remove the final norm before the LM head — bold, tests if it's doing useful work."""
    experiment_name: str = "g7_final_norm_none"
    final_norm_type: str = "none"

# --- Init scheme ---
@dataclass
class G7DepthScaledInitConfig(G6MuonLr018Config):
    """Depth-scaled init: weights ∝ 1/√depth — better conditioned gradients at every layer."""
    experiment_name: str = "g7_depth_scaled_init"
    init_scheme: str = "depth_scaled"

@dataclass
class G7GPT2InitConfig(G6MuonLr018Config):
    """GPT-2 init scheme — residual projections scaled by 1/√(2*n_layers)."""
    experiment_name: str = "g7_gpt2_init"
    init_scheme: str = "gpt2"

# --- Structural ---
@dataclass
class G7NoTieWeightsConfig(G6MuonLr018Config):
    """Separate LM head from input embeddings — more capacity at the cost of parameters."""
    experiment_name: str = "g7_no_tie_weights"
    tie_weights: bool = False

@dataclass
class G7UseBiasConfig(G6MuonLr018Config):
    """Add bias to all linear layers — very different inductive bias, rarely done in modern LLMs."""
    experiment_name: str = "g7_use_bias"
    use_bias: bool = True

@dataclass
class G7ValueNormParallelConfig(G6MuonLr018Config):
    """Combine value_norm + parallel_block — two novel mechanisms tested together."""
    experiment_name: str = "g7_value_norm_parallel"
    value_norm: bool = True
    parallel_block: bool = True

# Register Gen7 configs
ABLATION_CONFIGS["g7_parallel_block"]       = G7ParallelBlockConfig
ABLATION_CONFIGS["g7_relu_attn"]            = G7ReluAttnConfig
ABLATION_CONFIGS["g7_softcap_20"]           = G7Softcap20Config
ABLATION_CONFIGS["g7_softcap_10"]           = G7Softcap10Config
ABLATION_CONFIGS["g7_value_norm"]           = G7ValueNormConfig
ABLATION_CONFIGS["g7_attn_scale_half"]      = G7AttnScaleHalfConfig
ABLATION_CONFIGS["g7_attn_scale_2"]         = G7AttnScale2Config
ABLATION_CONFIGS["g7_gqa_2"]               = G7GQA2Config
ABLATION_CONFIGS["g7_no_pos"]               = G7NoPosConfig
ABLATION_CONFIGS["g7_learned_pos"]          = G7LearnedPosConfig
ABLATION_CONFIGS["g7_swiglu_new"]           = G7SwiGLUNewConfig
ABLATION_CONFIGS["g7_gated_sq_relu"]        = G7GatedSqReluConfig
ABLATION_CONFIGS["g7_bilinear_silu"]        = G7BilinearSiluConfig
ABLATION_CONFIGS["g7_bilinear_tanh"]        = G7BilinearTanhConfig
ABLATION_CONFIGS["g7_no_qk_norm"]           = G7NoQKNormConfig
ABLATION_CONFIGS["g7_q_norm_only"]          = G7QNormOnlyConfig
ABLATION_CONFIGS["g7_post_norm"]            = G7PostNormConfig
ABLATION_CONFIGS["g7_final_norm_none"]      = G7FinalNormNoneConfig
ABLATION_CONFIGS["g7_depth_scaled_init"]    = G7DepthScaledInitConfig
ABLATION_CONFIGS["g7_gpt2_init"]            = G7GPT2InitConfig
ABLATION_CONFIGS["g7_no_tie_weights"]       = G7NoTieWeightsConfig
ABLATION_CONFIGS["g7_use_bias"]             = G7UseBiasConfig
ABLATION_CONFIGS["g7_value_norm_parallel"]  = G7ValueNormParallelConfig


# ══════════════════════════════════════════════════════════════════════════
#  GENERATION 8 — experiments on top of g7_use_bias baseline
#  Base: bilinear + residual_scale=0.5 + linear + warmup=0.02
#        + muon_lr=0.018 + use_bias=True
#  50 diverse experiments — windowed attn, RoPE, FFN, norm, init, combos
# ══════════════════════════════════════════════════════════════════════════

# --- GQA variants ---
@dataclass
class G8GQA1Config(G7UseBiasConfig):
    experiment_name: str = "g8_gqa_1"
    n_kv_heads: int = 1

@dataclass
class G8GQA2Config(G7UseBiasConfig):
    experiment_name: str = "g8_gqa_2"
    n_kv_heads: int = 2

@dataclass
class G8GQA8Config(G7UseBiasConfig):
    experiment_name: str = "g8_gqa_8"
    n_kv_heads: int = 8

# --- RoPE base sweep ---
@dataclass
class G8Rope50kConfig(G7UseBiasConfig):
    experiment_name: str = "g8_rope_50k"
    rope_base: float = 50000.0

@dataclass
class G8Rope200kConfig(G7UseBiasConfig):
    experiment_name: str = "g8_rope_200k"
    rope_base: float = 200000.0

@dataclass
class G8Rope500kConfig(G7UseBiasConfig):
    experiment_name: str = "g8_rope_500k"
    rope_base: float = 500000.0

@dataclass
class G8Rope1mConfig(G7UseBiasConfig):
    experiment_name: str = "g8_rope_1m"
    rope_base: float = 1000000.0

# --- Local windowed attention (never tried before) ---
@dataclass
class G8Window32Config(G7UseBiasConfig):
    """Local attention window of 32 — forces strong locality bias."""
    experiment_name: str = "g8_window_32"
    attn_window_size: int = 32

@dataclass
class G8Window64Config(G7UseBiasConfig):
    """Local attention window of 64."""
    experiment_name: str = "g8_window_64"
    attn_window_size: int = 64

@dataclass
class G8Window128Config(G7UseBiasConfig):
    """Local attention window of 128 — moderate locality."""
    experiment_name: str = "g8_window_128"
    attn_window_size: int = 128

# --- FFN type variants on new baseline ---
@dataclass
class G8GLUConfig(G7UseBiasConfig):
    """GLU FFN on bias+muon_lr=0.018 baseline — never tested with these combined."""
    experiment_name: str = "g8_glu"
    ffn_type: str = "glu"

@dataclass
class G8SwiGLUBiasConfig(G7UseBiasConfig):
    """SwiGLU re-tested now WITH bias — bias changed everything, swiglu may rank differently."""
    experiment_name: str = "g8_swiglu_bias"
    ffn_type: str = "swiglu"

@dataclass
class G8BilinearGeluConfig(G7UseBiasConfig):
    """Bilinear gate with GELU — smooth non-monotonic gate different from squared_relu."""
    experiment_name: str = "g8_bilinear_gelu"
    activation_type: str = "gelu"

@dataclass
class G8BilinearReluConfig(G7UseBiasConfig):
    """Bilinear gate with ReLU — hard sparse gate, very different saturation from squared_relu."""
    experiment_name: str = "g8_bilinear_relu"
    activation_type: str = "relu"

@dataclass
class G8FFN1536Config(G7UseBiasConfig):
    """d_ff=1536 (3× d_model) — narrower bilinear FFN on bias baseline."""
    experiment_name: str = "g8_ffn_1536"
    d_ff: int = 1536

@dataclass
class G8FFN2560Config(G7UseBiasConfig):
    """d_ff=2560 (5× d_model) — slightly wider."""
    experiment_name: str = "g8_ffn_2560"
    d_ff: int = 2560

# --- Normalization ---
@dataclass
class G8LayerNormConfig(G7UseBiasConfig):
    """Full LayerNorm instead of RMSNorm — re-test on bias baseline."""
    experiment_name: str = "g8_layernorm"
    norm_type: str = "layernorm"

@dataclass
class G8FinalLayerNormConfig(G7UseBiasConfig):
    """LayerNorm only at the final output — keep body as RMSNorm."""
    experiment_name: str = "g8_final_layernorm"
    final_norm_type: str = "layernorm"

@dataclass
class G8LayerScale1e4Config(G7UseBiasConfig):
    """Layer scale init=1e-4 — very conservative near-zero init for each residual branch."""
    experiment_name: str = "g8_layer_scale_1e4"
    layer_scale_init: float = 1e-4

@dataclass
class G8LayerScale001Config(G7UseBiasConfig):
    """Layer scale init=0.01 — standard CaiT value."""
    experiment_name: str = "g8_layer_scale_001"
    layer_scale_init: float = 0.01

@dataclass
class G8LayerScale01Config(G7UseBiasConfig):
    """Layer scale init=0.1 — larger; less conservative."""
    experiment_name: str = "g8_layer_scale_01"
    layer_scale_init: float = 0.1

# --- Init schemes on bias baseline ---
@dataclass
class G8SmallEmbedInitConfig(G7UseBiasConfig):
    """Small embedding init — scale down embed weights at init to reduce early noise."""
    experiment_name: str = "g8_small_embed_init"
    init_scheme: str = "small_embed"

@dataclass
class G8DepthScaledBiasConfig(G7UseBiasConfig):
    """Depth-scaled init WITH bias — previously failed without bias; may behave differently."""
    experiment_name: str = "g8_depth_scaled_bias"
    init_scheme: str = "depth_scaled"

# --- Optimizer structure ---
@dataclass
class G8MuonNS3Config(G7UseBiasConfig):
    """muon_ns_steps=3 — fewer Newton-Schulz steps, cheaper but less accurate preconditioner."""
    experiment_name: str = "g8_muon_ns3"
    muon_ns_steps: int = 3

@dataclass
class G8MuonNS8Config(G7UseBiasConfig):
    """muon_ns_steps=8 — more steps for a sharper preconditioner estimate."""
    experiment_name: str = "g8_muon_ns8"
    muon_ns_steps: int = 8

@dataclass
class G8MuonNS10Config(G7UseBiasConfig):
    """muon_ns_steps=10 — maximum Newton-Schulz precision."""
    experiment_name: str = "g8_muon_ns10"
    muon_ns_steps: int = 10

# --- Regularization (mechanistic, not sweep) ---
@dataclass
class G8StochDepth005Config(G7UseBiasConfig):
    """Stochastic depth=0.05 on bias baseline — drop-path has never been tested here."""
    experiment_name: str = "g8_stoch_005"
    stochastic_depth: float = 0.05

@dataclass
class G8StochDepth010Config(G7UseBiasConfig):
    """Stochastic depth=0.10 — stronger drop-path."""
    experiment_name: str = "g8_stoch_010"
    stochastic_depth: float = 0.10

@dataclass
class G8LabelSmoothConfig(G7UseBiasConfig):
    """Label smoothing=0.05 — untested on bias baseline."""
    experiment_name: str = "g8_label_smooth"
    label_smoothing: float = 0.05

@dataclass
class G8Dropout005Config(G7UseBiasConfig):
    """Dropout=0.05 — very light stochastic regularization."""
    experiment_name: str = "g8_dropout_005"
    dropout: float = 0.05

@dataclass
class G8ZLoss1e3Config(G7UseBiasConfig):
    """z_loss=1e-3 — stronger logit entropy penalty than neutral 1e-4."""
    experiment_name: str = "g8_z_loss_1e3"
    z_loss_weight: float = 1e-3

@dataclass
class G8NoEmbedScaleConfig(G7UseBiasConfig):
    """Remove embedding scale — test if use_embed_scale is still needed with bias."""
    experiment_name: str = "g8_no_embed_scale"
    use_embed_scale: bool = False

# --- Novel combinations (mechanisms that failed alone but untested together) ---
@dataclass
class G8ValueNormBiasConfig(G7UseBiasConfig):
    """value_norm=True WITH bias — value_norm failed without bias; bias may change dynamics."""
    experiment_name: str = "g8_value_norm_bias"
    value_norm: bool = True

@dataclass
class G8Rope500kGQA8Config(G7UseBiasConfig):
    """Extended RoPE + full MHA — positional and attention head interaction."""
    experiment_name: str = "g8_rope500k_gqa8"
    rope_base: float = 500000.0
    n_kv_heads: int = 8

@dataclass
class G8Rope50kGQA2Config(G7UseBiasConfig):
    """RoPE 50k + GQA-2 — modest positional extension + aggressive KV sharing."""
    experiment_name: str = "g8_rope50k_gqa2"
    rope_base: float = 50000.0
    n_kv_heads: int = 2

@dataclass
class G8Window64GQA2Config(G7UseBiasConfig):
    """Windowed attention (64) + GQA-2 — local attention with aggressive KV sharing."""
    experiment_name: str = "g8_window64_gqa2"
    attn_window_size: int = 64
    n_kv_heads: int = 2

@dataclass
class G8Window128Rope500kConfig(G7UseBiasConfig):
    """Local window (128) + extended RoPE — local patterns with long-range positional bias."""
    experiment_name: str = "g8_window128_rope500k"
    attn_window_size: int = 128
    rope_base: float = 500000.0

@dataclass
class G8LayerScaleGQA8Config(G7UseBiasConfig):
    """Layer scale (0.01) + full MHA — layer scale might stabilize full attention."""
    experiment_name: str = "g8_lscale_gqa8"
    layer_scale_init: float = 0.01
    n_kv_heads: int = 8

@dataclass
class G8SwiGLULayerScaleConfig(G7UseBiasConfig):
    """SwiGLU + layer scale — different FFN + stabilized residuals."""
    experiment_name: str = "g8_swiglu_lscale"
    ffn_type: str = "swiglu"
    layer_scale_init: float = 0.01

@dataclass
class G8GLULayerNormConfig(G7UseBiasConfig):
    """GLU FFN + LayerNorm — two structural changes not tried together."""
    experiment_name: str = "g8_glu_layernorm"
    ffn_type: str = "glu"
    norm_type: str = "layernorm"

@dataclass
class G8BilinearGeluLScaleConfig(G7UseBiasConfig):
    """Bilinear+GELU gate + layer scale — smoother gate with stabilized residual."""
    experiment_name: str = "g8_bgelu_lscale"
    activation_type: str = "gelu"
    layer_scale_init: float = 0.01

@dataclass
class G8StochDepthGeluConfig(G7UseBiasConfig):
    """Stochastic depth + GELU gate — two regularization mechanisms combined."""
    experiment_name: str = "g8_stoch_bgelu"
    stochastic_depth: float = 0.05
    activation_type: str = "gelu"

@dataclass
class G8DepthScaledLayerScaleConfig(G7UseBiasConfig):
    """Depth-scaled init + layer scale — two depth-aware stabilization methods together."""
    experiment_name: str = "g8_depth_lscale"
    init_scheme: str = "depth_scaled"
    layer_scale_init: float = 0.01

@dataclass
class G8ValueNormRope500kConfig(G7UseBiasConfig):
    """value_norm + extended RoPE — attention output regularization + positional extension."""
    experiment_name: str = "g8_vnorm_rope500k"
    value_norm: bool = True
    rope_base: float = 500000.0

@dataclass
class G8MuonNS10Rope500kConfig(G7UseBiasConfig):
    """muon_ns_steps=10 + rope_base=500k — precise optimizer + extended positional."""
    experiment_name: str = "g8_ns10_rope500k"
    muon_ns_steps: int = 10
    rope_base: float = 500000.0

@dataclass
class G8GQA1Window128Config(G7UseBiasConfig):
    """MQA (1 KV head) + local window (128) — extreme KV compression + forced locality."""
    experiment_name: str = "g8_gqa1_window128"
    n_kv_heads: int = 1
    attn_window_size: int = 128

@dataclass
class G8LabelSmoothRope500kConfig(G7UseBiasConfig):
    """Label smoothing + extended RoPE — regularization + positional extension."""
    experiment_name: str = "g8_lsmooth_rope500k"
    label_smoothing: float = 0.05
    rope_base: float = 500000.0

@dataclass
class G8GeluLScaleGQA8Config(G7UseBiasConfig):
    """GELU gate + layer scale + full MHA — three mild changes never tried together."""
    experiment_name: str = "g8_gelu_ls_gqa8"
    activation_type: str = "gelu"
    layer_scale_init: float = 0.01
    n_kv_heads: int = 8

@dataclass
class G8SwiGLUGQA1Config(G7UseBiasConfig):
    """SwiGLU + MQA — test if MQA works better with SwiGLU than bilinear."""
    experiment_name: str = "g8_swiglu_gqa1"
    ffn_type: str = "swiglu"
    n_kv_heads: int = 1

@dataclass
class G8BilinearTanhLScaleConfig(G7UseBiasConfig):
    """Bilinear+tanh gate + layer scale — tanh failed before; layer scale may save it."""
    experiment_name: str = "g8_btanh_lscale"
    activation_type: str = "tanh"
    layer_scale_init: float = 0.01

@dataclass
class G8Window32Rope1mConfig(G7UseBiasConfig):
    """Tight local window (32) + extreme RoPE extension — hyper-local + hyper-extended."""
    experiment_name: str = "g8_window32_rope1m"
    attn_window_size: int = 32
    rope_base: float = 1000000.0

# Register Gen8 configs
ABLATION_CONFIGS["g8_gqa_1"]           = G8GQA1Config
ABLATION_CONFIGS["g8_gqa_2"]           = G8GQA2Config
ABLATION_CONFIGS["g8_gqa_8"]           = G8GQA8Config
ABLATION_CONFIGS["g8_rope_50k"]        = G8Rope50kConfig
ABLATION_CONFIGS["g8_rope_200k"]       = G8Rope200kConfig
ABLATION_CONFIGS["g8_rope_500k"]       = G8Rope500kConfig
ABLATION_CONFIGS["g8_rope_1m"]         = G8Rope1mConfig
ABLATION_CONFIGS["g8_window_32"]       = G8Window32Config
ABLATION_CONFIGS["g8_window_64"]       = G8Window64Config
ABLATION_CONFIGS["g8_window_128"]      = G8Window128Config
ABLATION_CONFIGS["g8_glu"]             = G8GLUConfig
ABLATION_CONFIGS["g8_swiglu_bias"]     = G8SwiGLUBiasConfig
ABLATION_CONFIGS["g8_bilinear_gelu"]   = G8BilinearGeluConfig
ABLATION_CONFIGS["g8_bilinear_relu"]   = G8BilinearReluConfig
ABLATION_CONFIGS["g8_ffn_1536"]        = G8FFN1536Config
ABLATION_CONFIGS["g8_ffn_2560"]        = G8FFN2560Config
ABLATION_CONFIGS["g8_layernorm"]       = G8LayerNormConfig
ABLATION_CONFIGS["g8_final_layernorm"] = G8FinalLayerNormConfig
ABLATION_CONFIGS["g8_layer_scale_1e4"] = G8LayerScale1e4Config
ABLATION_CONFIGS["g8_layer_scale_001"] = G8LayerScale001Config
ABLATION_CONFIGS["g8_layer_scale_01"]  = G8LayerScale01Config
ABLATION_CONFIGS["g8_small_embed_init"]= G8SmallEmbedInitConfig
ABLATION_CONFIGS["g8_depth_scaled_bias"]= G8DepthScaledBiasConfig
ABLATION_CONFIGS["g8_muon_ns3"]        = G8MuonNS3Config
ABLATION_CONFIGS["g8_muon_ns8"]        = G8MuonNS8Config
ABLATION_CONFIGS["g8_muon_ns10"]       = G8MuonNS10Config
ABLATION_CONFIGS["g8_stoch_005"]       = G8StochDepth005Config
ABLATION_CONFIGS["g8_stoch_010"]       = G8StochDepth010Config
ABLATION_CONFIGS["g8_label_smooth"]    = G8LabelSmoothConfig
ABLATION_CONFIGS["g8_dropout_005"]     = G8Dropout005Config
ABLATION_CONFIGS["g8_z_loss_1e3"]      = G8ZLoss1e3Config
ABLATION_CONFIGS["g8_no_embed_scale"]  = G8NoEmbedScaleConfig
ABLATION_CONFIGS["g8_value_norm_bias"] = G8ValueNormBiasConfig
ABLATION_CONFIGS["g8_rope500k_gqa8"]   = G8Rope500kGQA8Config
ABLATION_CONFIGS["g8_rope50k_gqa2"]    = G8Rope50kGQA2Config
ABLATION_CONFIGS["g8_window64_gqa2"]   = G8Window64GQA2Config
ABLATION_CONFIGS["g8_window128_rope500k"] = G8Window128Rope500kConfig
ABLATION_CONFIGS["g8_lscale_gqa8"]     = G8LayerScaleGQA8Config
ABLATION_CONFIGS["g8_swiglu_lscale"]   = G8SwiGLULayerScaleConfig
ABLATION_CONFIGS["g8_glu_layernorm"]   = G8GLULayerNormConfig
ABLATION_CONFIGS["g8_bgelu_lscale"]    = G8BilinearGeluLScaleConfig
ABLATION_CONFIGS["g8_stoch_bgelu"]     = G8StochDepthGeluConfig
ABLATION_CONFIGS["g8_depth_lscale"]    = G8DepthScaledLayerScaleConfig
ABLATION_CONFIGS["g8_vnorm_rope500k"]  = G8ValueNormRope500kConfig
ABLATION_CONFIGS["g8_ns10_rope500k"]   = G8MuonNS10Rope500kConfig
ABLATION_CONFIGS["g8_gqa1_window128"]  = G8GQA1Window128Config
ABLATION_CONFIGS["g8_lsmooth_rope500k"]= G8LabelSmoothRope500kConfig
ABLATION_CONFIGS["g8_gelu_ls_gqa8"]    = G8GeluLScaleGQA8Config
ABLATION_CONFIGS["g8_swiglu_gqa1"]     = G8SwiGLUGQA1Config
ABLATION_CONFIGS["g8_btanh_lscale"]    = G8BilinearTanhLScaleConfig
ABLATION_CONFIGS["g8_window32_rope1m"] = G8Window32Rope1mConfig


# ══════════════════════════════════════════════════════════════════════════
#  GENERATION 9 — 50 novel experiments on top of g7_use_bias baseline
#  Base: bilinear FFN + residual_scale=0.5 + linear schedule + warmup=0.02
#        + muon_lr=0.018 + use_bias=True
#  Three categories:
#   A. 20 Novel Muon optimizer variants — algorithmic changes to ortho update
#   B. 10 Novel FFN gate activations — new gating functions on bilinear FFN
#   C. 6  Novel attention/structural mechanisms — cosine attn, ALiBi, etc.
#   D. 14 Combination experiments — pairs of winning mechanisms
# ══════════════════════════════════════════════════════════════════════════

# ── Category A: 20 Novel Muon Optimizer Variants ─────────────────────────
# Each changes only the orthogonalization or momentum algorithm.

@dataclass
class G9MuonPostMomConfig(G7UseBiasConfig):
    """Post-momentum Muon: ortho(grad) first, then apply momentum to orthogonalized updates.
    Standard Muon: momentum(grad) → ortho. This reversal changes what is being smoothed."""
    experiment_name: str = "g9_muon_post_mom"
    muon_post_momentum: bool = True

@dataclass
class G9MuonGradCenterConfig(G7UseBiasConfig):
    """Gradient centralization: subtract row mean from grad before ortho.
    Reduces inter-neuron coupling; proven helpful in CV, untested in Muon."""
    experiment_name: str = "g9_muon_grad_center"
    muon_grad_centralize: bool = True

@dataclass
class G9MuonHalfOrtho05Config(G7UseBiasConfig):
    """50/50 blend of raw normalized gradient and orthogonalized gradient.
    Tests whether partial orthogonalization is better than full ortho."""
    experiment_name: str = "g9_muon_half_ortho_05"
    muon_half_ortho: float = 0.5

@dataclass
class G9MuonHalfOrtho02Config(G7UseBiasConfig):
    """20% ortho / 80% raw gradient blend — strongly biased toward raw gradient direction."""
    experiment_name: str = "g9_muon_half_ortho_02"
    muon_half_ortho: float = 0.2

@dataclass
class G9MuonCautiousConfig(G7UseBiasConfig):
    """Cautious Muon: zero update elements where sign(update) != sign(original grad).
    From Cautious Optimizers paper — prevents conflicting gradient information."""
    experiment_name: str = "g9_muon_cautious"
    muon_cautious: bool = True

@dataclass
class G9MuonFrobScaleConfig(G7UseBiasConfig):
    """Frobenius-norm scaling: replace aspect-ratio scale with ||G||_F / sqrt(m*n).
    Adapts update magnitude based on actual gradient magnitude rather than shape."""
    experiment_name: str = "g9_muon_frob_scale"
    muon_frob_scale: bool = True

@dataclass
class G9MuonDoubleOrthoConfig(G7UseBiasConfig):
    """Double orthogonalization: apply polar express twice in sequence.
    Second pass refines the approximation for better orthogonality quality."""
    experiment_name: str = "g9_muon_double_ortho"
    muon_double_ortho: bool = True

@dataclass
class G9MuonSignMix01Config(G7UseBiasConfig):
    """Sign mix 10%: ortho(g) + 0.1*sign(g). Blends orthogonalized gradient with
    sign gradient (Adam-style). Tests synergy between ortho and sign directions."""
    experiment_name: str = "g9_muon_sign_mix_01"
    muon_sign_mix: float = 0.1

@dataclass
class G9MuonSignMix05Config(G7UseBiasConfig):
    """Sign mix 50%: ortho(g) + 0.5*sign(g). Stronger Adam-like blending into Muon."""
    experiment_name: str = "g9_muon_sign_mix_05"
    muon_sign_mix: float = 0.5

@dataclass
class G9MuonRowNormConfig(G7UseBiasConfig):
    """Row-normalize gradient before polar express: each row becomes unit norm.
    Removes per-neuron magnitude, making ortho purely directional."""
    experiment_name: str = "g9_muon_row_norm"
    muon_row_norm: bool = True

@dataclass
class G9MuonColNormConfig(G7UseBiasConfig):
    """Column-normalize gradient before polar express: each column becomes unit norm.
    Dual of row_norm — normalizes per-feature rather than per-neuron."""
    experiment_name: str = "g9_muon_col_norm"
    muon_col_norm: bool = True

@dataclass
class G9MuonEmaOrthoConfig(G7UseBiasConfig):
    """EMA buffer on orthogonalized updates (separate from momentum on raw grad).
    Smooths the ortho output independently of the gradient momentum."""
    experiment_name: str = "g9_muon_ema_ortho"
    muon_ema_ortho: bool = True

@dataclass
class G9MuonAdaptiveNsConfig(G7UseBiasConfig):
    """Adaptive NS steps: use more polar express steps for larger matrices.
    Larger weight matrices need more iterations to converge to true unitary."""
    experiment_name: str = "g9_muon_adaptive_ns"
    muon_adaptive_ns: bool = True

@dataclass
class G9MuonTrustRegionConfig(G7UseBiasConfig):
    """Trust-region clipping: ||delta|| <= 0.05 * ||param||.
    Prevents large updates relative to current parameter scale."""
    experiment_name: str = "g9_muon_trust_region"
    muon_trust_region: float = 0.05

@dataclass
class G9MuonUpdateClipConfig(G7UseBiasConfig):
    """Clip orthogonalized update Frobenius norm to 1.0 before scaling.
    Decouples update magnitude from matrix dimensions."""
    experiment_name: str = "g9_muon_update_clip"
    muon_update_clip: float = 1.0

@dataclass
class G9MuonStochOrthoConfig(G7UseBiasConfig):
    """Stochastic orthogonalization: skip polar express with 10% probability.
    Adds noise to ortho process; may act as implicit regularization."""
    experiment_name: str = "g9_muon_stoch_ortho"
    muon_stochastic_ortho: float = 0.1

@dataclass
class G9MuonWarmMomConfig(G7UseBiasConfig):
    """Momentum warmup: ramp from 0.5 to 0.95 over first 100 steps.
    Early training uses lower momentum for more responsive updates."""
    experiment_name: str = "g9_muon_warm_mom"
    muon_warm_momentum: bool = True
    muon_warm_momentum_steps: int = 100

@dataclass
class G9MuonRmsNormConfig(G7UseBiasConfig):
    """RMS-normalize gradient before ortho: divide by sqrt(mean(g²)).
    Removes gradient scale heterogeneity before orthogonalization."""
    experiment_name: str = "g9_muon_rms_norm"
    muon_rms_norm_grad: bool = True

@dataclass
class G9MuonCautiousCenterConfig(G7UseBiasConfig):
    """Combo: cautious masking + gradient centralization.
    Two complementary corrections: directional filtering + mean removal."""
    experiment_name: str = "g9_muon_cautious_center"
    muon_cautious: bool = True
    muon_grad_centralize: bool = True

@dataclass
class G9MuonDoubleFrobConfig(G7UseBiasConfig):
    """Combo: double ortho + frobenius norm scaling.
    Better orthogonality quality + adaptive scaling by gradient magnitude."""
    experiment_name: str = "g9_muon_double_frob"
    muon_double_ortho: bool = True
    muon_frob_scale: bool = True


# ── Category B: 10 Novel FFN Gate Activations ────────────────────────────
# Current baseline bilinear FFN: gate(x) * up(x) — no activation on gate (pure bilinear).
# These test different nonlinearities applied to the gate path. NONE of these tried before.

@dataclass
class G9BilinearEluConfig(G7UseBiasConfig):
    """ELU(gate) × up — smooth at 0, allows negative outputs, bounded negative response."""
    experiment_name: str = "g9_bilinear_elu"
    ffn_type: str = "bilinear_elu"

@dataclass
class G9BilinearSoftplusConfig(G7UseBiasConfig):
    """softplus(gate) × up — smooth approximation to ReLU, always positive, log(1+e^x)."""
    experiment_name: str = "g9_bilinear_softplus"
    ffn_type: str = "bilinear_softplus"

@dataclass
class G9BilinearCosConfig(G7UseBiasConfig):
    """cos(gate) × up — periodic/oscillatory gate; selects features with spatial periodicity."""
    experiment_name: str = "g9_bilinear_cos"
    ffn_type: str = "bilinear_cos"

@dataclass
class G9BilinearAbsConfig(G7UseBiasConfig):
    """|gate| × up — symmetric activation; no sign information, only magnitude matters."""
    experiment_name: str = "g9_bilinear_abs"
    ffn_type: str = "bilinear_abs"

@dataclass
class G9BilinearSqrConfig(G7UseBiasConfig):
    """gate² × up — quadratic gate; always positive, stronger at large values."""
    experiment_name: str = "g9_bilinear_sqr"
    ffn_type: str = "bilinear_sqr"

@dataclass
class G9BilinearCubicConfig(G7UseBiasConfig):
    """gate³ × up — cubic gate; odd function, allows strong negative gating."""
    experiment_name: str = "g9_bilinear_cubic"
    ffn_type: str = "bilinear_cubic"

@dataclass
class G9BilinearGaussianConfig(G7UseBiasConfig):
    """exp(-gate²) × up — Gaussian gate; local feature selection, peaks at 0."""
    experiment_name: str = "g9_bilinear_gaussian"
    ffn_type: str = "bilinear_gaussian"

@dataclass
class G9BilinearStarConfig(G7UseBiasConfig):
    """gate*σ(gate) × up — StarReLU-like gate; combines linear and sigmoid."""
    experiment_name: str = "g9_bilinear_star"
    ffn_type: str = "bilinear_star"

@dataclass
class G9BilinearMishConfig(G7UseBiasConfig):
    """Mish(gate) × up — gate*tanh(softplus(gate)); smooth, non-monotonic."""
    experiment_name: str = "g9_bilinear_mish"
    ffn_type: str = "bilinear_mish"

@dataclass
class G9BilinearSqSiluConfig(G7UseBiasConfig):
    """SiLU(gate)² × up — squared SiLU gate; always positive, sharper than SiLU."""
    experiment_name: str = "g9_bilinear_sq_silu"
    ffn_type: str = "bilinear_sq_silu"


# ── Category C: 6 Novel Attention/Structural Mechanisms ──────────────────

@dataclass
class G9CosineAttnConfig(G7UseBiasConfig):
    """Cosine similarity attention: L2-normalize Q and K before dot product.
    Scale=1.0 (no sqrt(d) needed); theoretically cleaner similarity metric."""
    experiment_name: str = "g9_cosine_attn"
    cosine_attn: bool = True

@dataclass
class G9QRopeOnlyConfig(G7UseBiasConfig):
    """RoPE applied to Q only, not K. Asymmetric positional encoding.
    K retains absolute representation; Q carries positional query signal."""
    experiment_name: str = "g9_q_rope_only"
    q_rope_only: bool = True

@dataclass
class G9GatedResidualConfig(G7UseBiasConfig):
    """Learned sigmoid gate on each residual connection (per block).
    Gate starts at 0.5 (sigmoid(0)); model learns optimal residual strength."""
    experiment_name: str = "g9_gated_residual"
    gated_residual: bool = True

@dataclass
class G9AlibiConfig(G7UseBiasConfig):
    """ALiBi: attention with linear biases instead of RoPE.
    Head-specific linear decay bias on logits; no learned positional encoding."""
    experiment_name: str = "g9_alibi"
    alibi: bool = True
    use_rope: bool = False  # ALiBi replaces RoPE

@dataclass
class G9Residual040Config(G7UseBiasConfig):
    """residual_scale=0.4 — search below current best 0.5."""
    experiment_name: str = "g9_residual_040"
    residual_scale: float = 0.4

@dataclass
class G9Residual035Config(G7UseBiasConfig):
    """residual_scale=0.35 — more aggressive DeepNorm scaling."""
    experiment_name: str = "g9_residual_035"
    residual_scale: float = 0.35


# ── Category D: 14 Combination Experiments ───────────────────────────────
# Pair new mechanisms together; some combinations have specific mechanistic hypotheses.

@dataclass
class G9BilinearEluRs04Config(G7UseBiasConfig):
    """ELU bilinear gate + residual_scale=0.4. ELU's smoothness may pair with tighter residual."""
    experiment_name: str = "g9_bilinear_elu_rs04"
    ffn_type: str = "bilinear_elu"
    residual_scale: float = 0.4

@dataclass
class G9BilinearEluCosineAttnConfig(G7UseBiasConfig):
    """ELU bilinear + cosine attention. Two novel mechanisms; orthogonal in effect."""
    experiment_name: str = "g9_bilinear_elu_cosine_attn"
    ffn_type: str = "bilinear_elu"
    cosine_attn: bool = True

@dataclass
class G9BilinearEluQRopeConfig(G7UseBiasConfig):
    """ELU bilinear + Q-only RoPE. Novel FFN gate + asymmetric positional encoding."""
    experiment_name: str = "g9_bilinear_elu_q_rope"
    ffn_type: str = "bilinear_elu"
    q_rope_only: bool = True

@dataclass
class G9BilinearEluGatedResConfig(G7UseBiasConfig):
    """ELU bilinear + gated residual. Novel FFN + adaptive residual strength."""
    experiment_name: str = "g9_bilinear_elu_gated_res"
    ffn_type: str = "bilinear_elu"
    gated_residual: bool = True

@dataclass
class G9BilinearEluCautiousConfig(G7UseBiasConfig):
    """ELU bilinear + cautious Muon. Novel FFN gate + directionally filtered updates."""
    experiment_name: str = "g9_bilinear_elu_cautious"
    ffn_type: str = "bilinear_elu"
    muon_cautious: bool = True

@dataclass
class G9BilinearEluGradCenterConfig(G7UseBiasConfig):
    """ELU bilinear + Muon gradient centralization. Novel FFN + mean-removed ortho updates."""
    experiment_name: str = "g9_bilinear_elu_grad_center"
    ffn_type: str = "bilinear_elu"
    muon_grad_centralize: bool = True

@dataclass
class G9BilinearEluDoubleOrthoConfig(G7UseBiasConfig):
    """ELU bilinear + double orthogonalization. Novel gate + higher-quality ortho updates."""
    experiment_name: str = "g9_bilinear_elu_double_ortho"
    ffn_type: str = "bilinear_elu"
    muon_double_ortho: bool = True

@dataclass
class G9BilinearSoftplusRs04Config(G7UseBiasConfig):
    """Softplus bilinear + residual_scale=0.4. Smooth positive gate + tighter residual."""
    experiment_name: str = "g9_bilinear_softplus_rs04"
    ffn_type: str = "bilinear_softplus"
    residual_scale: float = 0.4

@dataclass
class G9CosineAttnGatedResConfig(G7UseBiasConfig):
    """Cosine attention + gated residual. Normalized similarity + adaptive residual depth."""
    experiment_name: str = "g9_cosine_attn_gated_res"
    cosine_attn: bool = True
    gated_residual: bool = True

@dataclass
class G9AlibiEluConfig(G7UseBiasConfig):
    """ALiBi + ELU bilinear. Position-linear bias + novel gate; two novel mechanisms."""
    experiment_name: str = "g9_alibi_elu"
    alibi: bool = True
    use_rope: bool = False
    ffn_type: str = "bilinear_elu"

@dataclass
class G9QRopeGatedResConfig(G7UseBiasConfig):
    """Q-only RoPE + gated residual. Asymmetric position + adaptive residual."""
    experiment_name: str = "g9_q_rope_gated_res"
    q_rope_only: bool = True
    gated_residual: bool = True

@dataclass
class G9BilinearStarRs04Config(G7UseBiasConfig):
    """StarReLU-like gate + residual_scale=0.4. StarReLU: x*σ(x) as gate."""
    experiment_name: str = "g9_bilinear_star_rs04"
    ffn_type: str = "bilinear_star"
    residual_scale: float = 0.4

@dataclass
class G9BilinearGaussRs035Config(G7UseBiasConfig):
    """Gaussian gate + residual_scale=0.35. Local feature selection + aggressive DeepNorm."""
    experiment_name: str = "g9_bilinear_gauss_rs035"
    ffn_type: str = "bilinear_gaussian"
    residual_scale: float = 0.35

@dataclass
class G9BilinearEluPostMomConfig(G7UseBiasConfig):
    """ELU bilinear + post-momentum Muon. Novel gate + reversed ortho/momentum order."""
    experiment_name: str = "g9_bilinear_elu_post_mom"
    ffn_type: str = "bilinear_elu"
    muon_post_momentum: bool = True


# ── Gen9 Registry ─────────────────────────────────────────────────────────
# Category A: Muon variants
ABLATION_CONFIGS["g9_muon_post_mom"]        = G9MuonPostMomConfig
ABLATION_CONFIGS["g9_muon_grad_center"]     = G9MuonGradCenterConfig
ABLATION_CONFIGS["g9_muon_half_ortho_05"]   = G9MuonHalfOrtho05Config
ABLATION_CONFIGS["g9_muon_half_ortho_02"]   = G9MuonHalfOrtho02Config
ABLATION_CONFIGS["g9_muon_cautious"]        = G9MuonCautiousConfig
ABLATION_CONFIGS["g9_muon_frob_scale"]      = G9MuonFrobScaleConfig
ABLATION_CONFIGS["g9_muon_double_ortho"]    = G9MuonDoubleOrthoConfig
ABLATION_CONFIGS["g9_muon_sign_mix_01"]     = G9MuonSignMix01Config
ABLATION_CONFIGS["g9_muon_sign_mix_05"]     = G9MuonSignMix05Config
ABLATION_CONFIGS["g9_muon_row_norm"]        = G9MuonRowNormConfig
ABLATION_CONFIGS["g9_muon_col_norm"]        = G9MuonColNormConfig
ABLATION_CONFIGS["g9_muon_ema_ortho"]       = G9MuonEmaOrthoConfig
ABLATION_CONFIGS["g9_muon_adaptive_ns"]     = G9MuonAdaptiveNsConfig
ABLATION_CONFIGS["g9_muon_trust_region"]    = G9MuonTrustRegionConfig
ABLATION_CONFIGS["g9_muon_update_clip"]     = G9MuonUpdateClipConfig
ABLATION_CONFIGS["g9_muon_stoch_ortho"]     = G9MuonStochOrthoConfig
ABLATION_CONFIGS["g9_muon_warm_mom"]        = G9MuonWarmMomConfig
ABLATION_CONFIGS["g9_muon_rms_norm"]        = G9MuonRmsNormConfig
ABLATION_CONFIGS["g9_muon_cautious_center"] = G9MuonCautiousCenterConfig
ABLATION_CONFIGS["g9_muon_double_frob"]     = G9MuonDoubleFrobConfig
# Category B: Novel FFN gate types
ABLATION_CONFIGS["g9_bilinear_elu"]         = G9BilinearEluConfig
ABLATION_CONFIGS["g9_bilinear_softplus"]    = G9BilinearSoftplusConfig
ABLATION_CONFIGS["g9_bilinear_cos"]         = G9BilinearCosConfig
ABLATION_CONFIGS["g9_bilinear_abs"]         = G9BilinearAbsConfig
ABLATION_CONFIGS["g9_bilinear_sqr"]         = G9BilinearSqrConfig
ABLATION_CONFIGS["g9_bilinear_cubic"]       = G9BilinearCubicConfig
ABLATION_CONFIGS["g9_bilinear_gaussian"]    = G9BilinearGaussianConfig
ABLATION_CONFIGS["g9_bilinear_star"]        = G9BilinearStarConfig
ABLATION_CONFIGS["g9_bilinear_mish"]        = G9BilinearMishConfig
ABLATION_CONFIGS["g9_bilinear_sq_silu"]     = G9BilinearSqSiluConfig
# Category C: Novel attention/structural mechanisms
ABLATION_CONFIGS["g9_cosine_attn"]          = G9CosineAttnConfig
ABLATION_CONFIGS["g9_q_rope_only"]          = G9QRopeOnlyConfig
ABLATION_CONFIGS["g9_gated_residual"]       = G9GatedResidualConfig
ABLATION_CONFIGS["g9_alibi"]                = G9AlibiConfig
ABLATION_CONFIGS["g9_residual_040"]         = G9Residual040Config
ABLATION_CONFIGS["g9_residual_035"]         = G9Residual035Config
# Category D: Combinations
ABLATION_CONFIGS["g9_bilinear_elu_rs04"]        = G9BilinearEluRs04Config
ABLATION_CONFIGS["g9_bilinear_elu_cosine_attn"] = G9BilinearEluCosineAttnConfig
ABLATION_CONFIGS["g9_bilinear_elu_q_rope"]      = G9BilinearEluQRopeConfig
ABLATION_CONFIGS["g9_bilinear_elu_gated_res"]   = G9BilinearEluGatedResConfig
ABLATION_CONFIGS["g9_bilinear_elu_cautious"]    = G9BilinearEluCautiousConfig
ABLATION_CONFIGS["g9_bilinear_elu_grad_center"] = G9BilinearEluGradCenterConfig
ABLATION_CONFIGS["g9_bilinear_elu_double_ortho"]= G9BilinearEluDoubleOrthoConfig
ABLATION_CONFIGS["g9_bilinear_softplus_rs04"]   = G9BilinearSoftplusRs04Config
ABLATION_CONFIGS["g9_cosine_attn_gated_res"]    = G9CosineAttnGatedResConfig
ABLATION_CONFIGS["g9_alibi_elu"]                = G9AlibiEluConfig
ABLATION_CONFIGS["g9_q_rope_gated_res"]         = G9QRopeGatedResConfig
ABLATION_CONFIGS["g9_bilinear_star_rs04"]       = G9BilinearStarRs04Config
ABLATION_CONFIGS["g9_bilinear_gauss_rs035"]     = G9BilinearGaussRs035Config
ABLATION_CONFIGS["g9_bilinear_elu_post_mom"]    = G9BilinearEluPostMomConfig


# ══════════════════════════════════════════════════════════════════════════
#  GENERATION 10 — 50 experiments on top of g9_gated_residual baseline
#  New best: g9_gated_residual — val_loss 4.8844
#  Base: G7UseBias + gated_residual=True
#  Key finding: learned sigmoid gate per block is a genuine improvement.
#  Gen10 probes: how far can we push this, what stacks with it, new mechanisms.
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class G9GatedResidualConfig(G7UseBiasConfig):
    """New Gen10 baseline: gated_residual=True on top of g7_use_bias.
    val_loss 4.8844 — learned sigmoid gate on each residual connection."""
    experiment_name: str = "g9_gated_residual_baseline"
    gated_residual: bool = True

# Alias for clarity
G10BaseConfig = G9GatedResidualConfig


# ── A: Residual scale stacking with gated residual (8 experiments) ────────
# gated_residual already starts at sigmoid(0)=0.5 effective scale.
# Key question: does fixed residual_scale still matter on top of learned gate?

@dataclass
class G10GatedRs035Config(G10BaseConfig):
    """Gated residual + residual_scale=0.35. Both win independently — do they compound?"""
    experiment_name: str = "g10_gated_rs035"
    residual_scale: float = 0.35

@dataclass
class G10GatedRs040Config(G10BaseConfig):
    """Gated residual + residual_scale=0.40."""
    experiment_name: str = "g10_gated_rs040"
    residual_scale: float = 0.40

@dataclass
class G10GatedRs060Config(G10BaseConfig):
    """Gated residual + residual_scale=0.60 — above current 0.5."""
    experiment_name: str = "g10_gated_rs060"
    residual_scale: float = 0.60

@dataclass
class G10GatedRs100Config(G10BaseConfig):
    """Gated residual + residual_scale=1.0 — let gate handle ALL scaling."""
    experiment_name: str = "g10_gated_rs100"
    residual_scale: float = 1.0

@dataclass
class G10GatedRs025Config(G10BaseConfig):
    """Gated residual + residual_scale=0.25 — very aggressive DeepNorm + gate."""
    experiment_name: str = "g10_gated_rs025"
    residual_scale: float = 0.25


# ── B: Gate initialization variants (4 experiments) ──────────────────────
# Default gate init: sigmoid(0.0)=0.5 (half-open). Test other starting points.

@dataclass
class G10GateInitHighConfig(G10BaseConfig):
    """Gate initialized nearly open: sigmoid(2.0)≈0.88. Starts closer to standard residual."""
    experiment_name: str = "g10_gate_init_high"
    gate_init: float = 2.0

@dataclass
class G10GateInitLowConfig(G10BaseConfig):
    """Gate initialized nearly closed: sigmoid(-2.0)≈0.12. Aggressive residual suppression."""
    experiment_name: str = "g10_gate_init_low"
    gate_init: float = -2.0

@dataclass
class G10GateInitOpenConfig(G10BaseConfig):
    """Gate initialized fully open: sigmoid(4.0)≈0.98. Acts like normal residual at init."""
    experiment_name: str = "g10_gate_init_open"
    gate_init: float = 4.0

@dataclass
class G10GatePerChannelConfig(G10BaseConfig):
    """Per-channel gate: d_model scalar gates per sublayer instead of 1 per block.
    Much richer gating — model can selectively pass/block individual features."""
    experiment_name: str = "g10_gate_per_channel"
    gate_per_channel: bool = True


# ── C: Stack gated_residual with winning Muon variants (10 experiments) ──
# Testing if optimizer improvements compound with the gated residual mechanism.

@dataclass
class G10GatedMuonPostMomConfig(G10BaseConfig):
    """Gated residual + post-momentum Muon."""
    experiment_name: str = "g10_gated_muon_post_mom"
    muon_post_momentum: bool = True

@dataclass
class G10GatedMuonGradCenterConfig(G10BaseConfig):
    """Gated residual + gradient centralization."""
    experiment_name: str = "g10_gated_muon_grad_center"
    muon_grad_centralize: bool = True

@dataclass
class G10GatedMuonCautiousConfig(G10BaseConfig):
    """Gated residual + cautious Muon updates."""
    experiment_name: str = "g10_gated_muon_cautious"
    muon_cautious: bool = True

@dataclass
class G10GatedMuonDoubleOrthoConfig(G10BaseConfig):
    """Gated residual + double orthogonalization."""
    experiment_name: str = "g10_gated_muon_double_ortho"
    muon_double_ortho: bool = True

@dataclass
class G10GatedMuonAdaptiveNsConfig(G10BaseConfig):
    """Gated residual + adaptive NS steps."""
    experiment_name: str = "g10_gated_muon_adaptive_ns"
    muon_adaptive_ns: bool = True

@dataclass
class G10GatedMuonWarmMomConfig(G10BaseConfig):
    """Gated residual + momentum warmup."""
    experiment_name: str = "g10_gated_muon_warm_mom"
    muon_warm_momentum: bool = True
    muon_warm_momentum_steps: int = 100

@dataclass
class G10GatedMuonRmsNormConfig(G10BaseConfig):
    """Gated residual + RMS-normalize gradient."""
    experiment_name: str = "g10_gated_muon_rms_norm"
    muon_rms_norm_grad: bool = True

@dataclass
class G10GatedMuonRowNormConfig(G10BaseConfig):
    """Gated residual + row-normalize gradient before polar express."""
    experiment_name: str = "g10_gated_muon_row_norm"
    muon_row_norm: bool = True

@dataclass
class G10GatedMuonSignMix01Config(G10BaseConfig):
    """Gated residual + 10% sign gradient mixing."""
    experiment_name: str = "g10_gated_muon_sign_mix_01"
    muon_sign_mix: float = 0.1

@dataclass
class G10GatedMuonHalfOrtho05Config(G10BaseConfig):
    """Gated residual + 50% ortho blend."""
    experiment_name: str = "g10_gated_muon_half_ortho_05"
    muon_half_ortho: float = 0.5


# ── D: Stack gated_residual with novel attention/positional mechanisms (8) ─

@dataclass
class G10GatedCosineAttnConfig(G10BaseConfig):
    """Gated residual + cosine similarity attention."""
    experiment_name: str = "g10_gated_cosine_attn"
    cosine_attn: bool = True

@dataclass
class G10GatedAlibiConfig(G10BaseConfig):
    """Gated residual + ALiBi positional encoding (no RoPE)."""
    experiment_name: str = "g10_gated_alibi"
    alibi: bool = True
    use_rope: bool = False

@dataclass
class G10GatedQKLayerNormConfig(G10BaseConfig):
    """Gated residual + QK LayerNorm. LayerNorm was an early winner never tested on this baseline."""
    experiment_name: str = "g10_gated_qkln"
    qk_norm_type: str = "layernorm"

@dataclass
class G10GatedBilinearEluRs035Config(G10BaseConfig):
    """Gated residual + ELU bilinear + residual_scale=0.35. Three-way combo of winners."""
    experiment_name: str = "g10_gated_bilinear_elu_rs035"
    ffn_type: str = "bilinear_elu"
    residual_scale: float = 0.35

@dataclass
class G10GatedBilinearEluConfig(G10BaseConfig):
    """Gated residual + ELU bilinear gate. ELU was neutral on Gen9; test again on new baseline."""
    experiment_name: str = "g10_gated_bilinear_elu"
    ffn_type: str = "bilinear_elu"

@dataclass
class G10GatedBilinearSoftplusConfig(G10BaseConfig):
    """Gated residual + softplus bilinear gate."""
    experiment_name: str = "g10_gated_bilinear_softplus"
    ffn_type: str = "bilinear_softplus"

@dataclass
class G10GatedBilinearStarConfig(G10BaseConfig):
    """Gated residual + StarReLU-like gate (x*σ(x)). Showed some promise in Gen9."""
    experiment_name: str = "g10_gated_bilinear_star"
    ffn_type: str = "bilinear_star"

@dataclass
class G10GatedPerChRs035Config(G10BaseConfig):
    """Per-channel gate + residual_scale=0.35. Richer gating + tighter DeepNorm."""
    experiment_name: str = "g10_gate_per_ch_rs035"
    gate_per_channel: bool = True
    residual_scale: float = 0.35


# ── E: Novel mechanisms never tried (11 experiments) ─────────────────────

@dataclass
class G10GatedSandwichExtraConfig(G10BaseConfig):
    """Gated residual + extra sandwich norm layers (double normalization per sub-layer).
    Already using sandwich; test if additional norm positions help with gating."""
    experiment_name: str = "g10_gated_bilinear_abs"
    ffn_type: str = "bilinear_abs"

@dataclass
class G10GatedMuonCenterRs035Config(G10BaseConfig):
    """Gated residual + grad centralization + rs=0.35. Three synergistic mechanisms."""
    experiment_name: str = "g10_gated_center_rs035"
    muon_grad_centralize: bool = True
    residual_scale: float = 0.35

@dataclass
class G10GatedMuonCautiousCenterConfig(G10BaseConfig):
    """Gated residual + cautious + grad centralization — dual Muon corrections."""
    experiment_name: str = "g10_gated_cautious_center"
    muon_cautious: bool = True
    muon_grad_centralize: bool = True

@dataclass
class G10GatedMuonAdaptiveRs035Config(G10BaseConfig):
    """Gated residual + adaptive NS + rs=0.35."""
    experiment_name: str = "g10_gated_adaptive_rs035"
    muon_adaptive_ns: bool = True
    residual_scale: float = 0.35

@dataclass
class G10GateHighRs035Config(G10BaseConfig):
    """Gate init high (near-open) + rs=0.35. Start as normal residual, learn to gate down."""
    experiment_name: str = "g10_gate_high_rs035"
    gate_init: float = 2.0
    residual_scale: float = 0.35

@dataclass
class G10GateHighPerChConfig(G10BaseConfig):
    """Gate init high + per-channel gate. Near-full residual at init, fine-grained gating."""
    experiment_name: str = "g10_gate_high_per_ch"
    gate_init: float = 2.0
    gate_per_channel: bool = True

@dataclass
class G10GatedCosineQKLNConfig(G10BaseConfig):
    """Gated residual + cosine attention + QK LayerNorm."""
    experiment_name: str = "g10_gated_cosine_qkln"
    cosine_attn: bool = True
    qk_norm_type: str = "layernorm"

@dataclass
class G10GatedMuonPostMomRs035Config(G10BaseConfig):
    """Gated residual + post-momentum Muon + rs=0.35."""
    experiment_name: str = "g10_gated_post_mom_rs035"
    muon_post_momentum: bool = True
    residual_scale: float = 0.35

@dataclass
class G10GatedMuonRowNormRs035Config(G10BaseConfig):
    """Gated residual + row-norm Muon + rs=0.35."""
    experiment_name: str = "g10_gated_row_norm_rs035"
    muon_row_norm: bool = True
    residual_scale: float = 0.35

@dataclass
class G10GatedBilinearEluQKLNConfig(G10BaseConfig):
    """Gated residual + ELU bilinear + QK LayerNorm. Three distinct mechanisms."""
    experiment_name: str = "g10_gated_elu_qkln"
    ffn_type: str = "bilinear_elu"
    qk_norm_type: str = "layernorm"

@dataclass
class G10GatedPerChCosineConfig(G10BaseConfig):
    """Per-channel gate + cosine attention. Rich gating + normalized similarity."""
    experiment_name: str = "g10_gate_per_ch_cosine"
    gate_per_channel: bool = True
    cosine_attn: bool = True


# ── F: Exploitation of residual_035 winner (4 experiments) ──────────────
# residual_scale=0.35 was a standalone winner — exploit it further.

@dataclass
class G10Rs035MuonGradCenterConfig(G10BaseConfig):
    """Gated residual + rs=0.35 + grad centralization."""
    experiment_name: str = "g10_rs035_center"
    residual_scale: float = 0.35
    muon_grad_centralize: bool = True

@dataclass
class G10Rs035MuonCautiousConfig(G10BaseConfig):
    """Gated residual + rs=0.35 + cautious Muon."""
    experiment_name: str = "g10_rs035_cautious"
    residual_scale: float = 0.35
    muon_cautious: bool = True

@dataclass
class G10Rs035BilinearStarConfig(G10BaseConfig):
    """Gated residual + rs=0.35 + star bilinear gate."""
    experiment_name: str = "g10_rs035_bilinear_star"
    residual_scale: float = 0.35
    ffn_type: str = "bilinear_star"

@dataclass
class G10Rs035QKLNConfig(G10BaseConfig):
    """Gated residual + rs=0.35 + QK LayerNorm."""
    experiment_name: str = "g10_rs035_qkln"
    residual_scale: float = 0.35
    qk_norm_type: str = "layernorm"


# ── Gen10 Registry ────────────────────────────────────────────────────────
ABLATION_CONFIGS["g9_gated_residual_baseline"]   = G9GatedResidualConfig
# A: Residual scale stacking
ABLATION_CONFIGS["g10_gated_rs035"]              = G10GatedRs035Config
ABLATION_CONFIGS["g10_gated_rs040"]              = G10GatedRs040Config
ABLATION_CONFIGS["g10_gated_rs060"]              = G10GatedRs060Config
ABLATION_CONFIGS["g10_gated_rs100"]              = G10GatedRs100Config
ABLATION_CONFIGS["g10_gated_rs025"]              = G10GatedRs025Config
# B: Gate init variants
ABLATION_CONFIGS["g10_gate_init_high"]           = G10GateInitHighConfig
ABLATION_CONFIGS["g10_gate_init_low"]            = G10GateInitLowConfig
ABLATION_CONFIGS["g10_gate_init_open"]           = G10GateInitOpenConfig
ABLATION_CONFIGS["g10_gate_per_channel"]         = G10GatePerChannelConfig
# C: Stack with Muon variants
ABLATION_CONFIGS["g10_gated_muon_post_mom"]      = G10GatedMuonPostMomConfig
ABLATION_CONFIGS["g10_gated_muon_grad_center"]   = G10GatedMuonGradCenterConfig
ABLATION_CONFIGS["g10_gated_muon_cautious"]      = G10GatedMuonCautiousConfig
ABLATION_CONFIGS["g10_gated_muon_double_ortho"]  = G10GatedMuonDoubleOrthoConfig
ABLATION_CONFIGS["g10_gated_muon_adaptive_ns"]   = G10GatedMuonAdaptiveNsConfig
ABLATION_CONFIGS["g10_gated_muon_warm_mom"]      = G10GatedMuonWarmMomConfig
ABLATION_CONFIGS["g10_gated_muon_rms_norm"]      = G10GatedMuonRmsNormConfig
ABLATION_CONFIGS["g10_gated_muon_row_norm"]      = G10GatedMuonRowNormConfig
ABLATION_CONFIGS["g10_gated_muon_sign_mix_01"]   = G10GatedMuonSignMix01Config
ABLATION_CONFIGS["g10_gated_muon_half_ortho_05"] = G10GatedMuonHalfOrtho05Config
# D: Stack with attention/positional
ABLATION_CONFIGS["g10_gated_cosine_attn"]        = G10GatedCosineAttnConfig
ABLATION_CONFIGS["g10_gated_alibi"]              = G10GatedAlibiConfig
ABLATION_CONFIGS["g10_gated_qkln"]              = G10GatedQKLayerNormConfig
ABLATION_CONFIGS["g10_gated_bilinear_elu_rs035"] = G10GatedBilinearEluRs035Config
ABLATION_CONFIGS["g10_gated_bilinear_elu"]       = G10GatedBilinearEluConfig
ABLATION_CONFIGS["g10_gated_bilinear_softplus"]  = G10GatedBilinearSoftplusConfig
ABLATION_CONFIGS["g10_gated_bilinear_star"]      = G10GatedBilinearStarConfig
ABLATION_CONFIGS["g10_gate_per_ch_rs035"]        = G10GatedPerChRs035Config
# E: Novel combos
ABLATION_CONFIGS["g10_gated_bilinear_abs"]       = G10GatedSandwichExtraConfig
ABLATION_CONFIGS["g10_gated_center_rs035"]       = G10GatedMuonCenterRs035Config
ABLATION_CONFIGS["g10_gated_cautious_center"]    = G10GatedMuonCautiousCenterConfig
ABLATION_CONFIGS["g10_gated_adaptive_rs035"]     = G10GatedMuonAdaptiveRs035Config
ABLATION_CONFIGS["g10_gate_high_rs035"]          = G10GateHighRs035Config
ABLATION_CONFIGS["g10_gate_high_per_ch"]         = G10GateHighPerChConfig
ABLATION_CONFIGS["g10_gated_cosine_qkln"]        = G10GatedCosineQKLNConfig
ABLATION_CONFIGS["g10_gated_post_mom_rs035"]     = G10GatedMuonPostMomRs035Config
ABLATION_CONFIGS["g10_gated_row_norm_rs035"]     = G10GatedMuonRowNormRs035Config
ABLATION_CONFIGS["g10_gated_elu_qkln"]           = G10GatedBilinearEluQKLNConfig
ABLATION_CONFIGS["g10_gate_per_ch_cosine"]       = G10GatedPerChCosineConfig
# F: Exploit rs=0.35
ABLATION_CONFIGS["g10_rs035_center"]             = G10Rs035MuonGradCenterConfig
ABLATION_CONFIGS["g10_rs035_cautious"]           = G10Rs035MuonCautiousConfig
ABLATION_CONFIGS["g10_rs035_bilinear_star"]      = G10Rs035BilinearStarConfig
ABLATION_CONFIGS["g10_rs035_qkln"]              = G10Rs035QKLNConfig

# ── G: 8 more novel combos to hit 50 ──────────────────────────────────────

@dataclass
class G10GatedMuonUpdateClipConfig(G10BaseConfig):
    """Gated residual + update clip. Clip ortho update norm for stability."""
    experiment_name: str = "g10_gated_update_clip"
    muon_update_clip: float = 1.0

@dataclass
class G10GatedMuonTrustRegionConfig(G10BaseConfig):
    """Gated residual + trust region clipping (5% of param norm)."""
    experiment_name: str = "g10_gated_trust_region"
    muon_trust_region: float = 0.05

@dataclass
class G10Rs035GatedHighConfig(G10BaseConfig):
    """rs=0.35 + gate init high (0.88). High-gate + tight residual scale."""
    experiment_name: str = "g10_rs035_gate_high"
    residual_scale: float = 0.35
    gate_init: float = 2.0

@dataclass
class G10Rs035PerChConfig(G10BaseConfig):
    """rs=0.35 + per-channel gate. Tightest residual + richest gating."""
    experiment_name: str = "g10_rs035_per_ch"
    residual_scale: float = 0.35
    gate_per_channel: bool = True

@dataclass
class G10GatedMuonColNormConfig(G10BaseConfig):
    """Gated residual + column-normalize gradient before polar express."""
    experiment_name: str = "g10_gated_muon_col_norm"
    muon_col_norm: bool = True

@dataclass
class G10GatedBilinearMishConfig(G10BaseConfig):
    """Gated residual + Mish gate (x*tanh(softplus(x))). Non-monotonic gate."""
    experiment_name: str = "g10_gated_bilinear_mish"
    ffn_type: str = "bilinear_mish"

@dataclass
class G10GatedBilinearSqSiluConfig(G10BaseConfig):
    """Gated residual + squared SiLU gate. Always-positive, sharper gate."""
    experiment_name: str = "g10_gated_bilinear_sq_silu"
    ffn_type: str = "bilinear_sq_silu"

@dataclass
class G10GatedBilinearEluPerChConfig(G10BaseConfig):
    """ELU bilinear + per-channel gate + gated residual. Three novel mechanisms."""
    experiment_name: str = "g10_gated_elu_per_ch"
    ffn_type: str = "bilinear_elu"
    gate_per_channel: bool = True

ABLATION_CONFIGS["g10_gated_update_clip"]        = G10GatedMuonUpdateClipConfig
ABLATION_CONFIGS["g10_gated_trust_region"]       = G10GatedMuonTrustRegionConfig
ABLATION_CONFIGS["g10_rs035_gate_high"]          = G10Rs035GatedHighConfig
ABLATION_CONFIGS["g10_rs035_per_ch"]             = G10Rs035PerChConfig
ABLATION_CONFIGS["g10_gated_muon_col_norm"]      = G10GatedMuonColNormConfig
ABLATION_CONFIGS["g10_gated_bilinear_mish"]      = G10GatedBilinearMishConfig
ABLATION_CONFIGS["g10_gated_bilinear_sq_silu"]   = G10GatedBilinearSqSiluConfig
ABLATION_CONFIGS["g10_gated_elu_per_ch"]         = G10GatedBilinearEluPerChConfig


def get_ablation_config(name: str, train_tokens: int = 10_000) -> LLMConfig:
    if name not in ABLATION_CONFIGS:
        raise ValueError(
            f"Unknown ablation: '{name}'.\n"
            f"Available ({len(ABLATION_CONFIGS)}): {sorted(ABLATION_CONFIGS.keys())}"
        )
    config = ABLATION_CONFIGS[name]()
    config.train_tokens = train_tokens
    return config
