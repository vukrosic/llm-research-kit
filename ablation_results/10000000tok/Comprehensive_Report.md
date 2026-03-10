# 🧬 Comprehensive LLM Ablation Study Report (10M Tokens)
This document analyzes the results of 50 different Architectural and Hyperparameter ablations run sequentially on a 10M token dataset. The goal is to identify which changes strictly improve the minimal LLM baseline, and which components lead to degradation or instability.

## 📊 Executive Summary
![Ablation Results Plot](./ablation_plot.png)

From 50 planned variants, **48 completed training**. 
- The `no_rope` and `learned_pos_embed` variants **failed to compile** due to CUDA memory and kernel access errors under `torch.compile`.
- The `no_norm` variant **collapsed to NaN** (gradients exploded) immediately.

**Baseline Validation Loss:** `4.7733`

### 🏆 Top 5 Best Performing Models
- **swiglu_layernorm**: `4.7053` (-1.42% vs baseline)
- **low_muon_lr**: `4.7182` (-1.16% vs baseline)
- **parallel_swiglu**: `4.7219` (-1.08% vs baseline)
- **act_gelu**: `4.7234` (-1.05% vs baseline)
- **sandwich_norm**: `4.7243` (-1.03% vs baseline)

### ⚠️ Bottom 5 Worst Performing Models (Excluding Failures/NaNs)
- **post_norm**: `7.6537` (+60.34% vs baseline)
- **gpt2_style**: `7.6599` (+60.47% vs baseline)
- **layer_norm_post**: `7.6616` (+60.51% vs baseline)
- **swiglu_post_norm**: `7.6616` (+60.51% vs baseline)
- **no_final_norm**: `12.4762` (+161.37% vs baseline)

---
## 🔬 Detailed Results by Category

### Original
| Experiment | Val Loss | Val Acc | Delta vs Baseline | Note / Explanation |
|------------|----------|---------|-------------------|--------------------|
| `baseline` | `4.7733` | 24.92% | +0.00% | Standard comparison point. |
| `no_embed_scale` | `4.9173` | 23.92% | +3.02% | Architectural modification. |
| `no_qk_norm` | `4.8660` | 24.07% | +1.94% | Normalization variants. |
| `polar_express_2` | `4.9891` | 23.20% | +4.52% | Architectural modification. |
| `act_gelu` | `4.7234` | 25.89% | -1.05% | Architectural modification. |
| `act_silu` | `4.8173` | 24.65% | +0.92% | Architectural modification. |
| `rope_base_500k` | `4.7677` | 24.95% | -0.12% | Context representations. |
| `schedule_cosine` | `4.7551` | 25.00% | -0.38% | Learning rate and schedule tuning. |
| `muon_no_momentum` | `4.8951` | 23.73% | +2.55% | Architectural modification. |
| `no_weight_decay` | `4.7309` | 25.07% | -0.89% | Architectural modification. |
| `high_adam_lr` | `4.7990` | 25.22% | +0.54% | Learning rate and schedule tuning. |
| `high_muon_lr` | `4.8155` | 24.58% | +0.88% | Learning rate and schedule tuning. |


### Normalization
| Experiment | Val Loss | Val Acc | Delta vs Baseline | Note / Explanation |
|------------|----------|---------|-------------------|--------------------|
| `post_norm` | `7.6537` | 4.30% | +60.34% | Normalization variants. |
| `sandwich_norm` | `4.7243` | 25.57% | -1.03% | Normalization variants. |
| `layer_norm` | `4.7445` | 25.24% | -0.60% | Normalization variants. |
| `layer_norm_post` | `7.6616` | 4.30% | +60.51% | Normalization variants. |
| `no_norm` | NaN | 0.00% | NaN | Model diverged/collapsed (NaN) |
| `no_final_norm` | `12.4762` | 3.91% | +161.37% | Normalization variants. |


### FFN Architecture
| Experiment | Val Loss | Val Acc | Delta vs Baseline | Note / Explanation |
|------------|----------|---------|-------------------|--------------------|
| `swiglu` | `4.7347` | 25.41% | -0.81% | SwiGLU gating efficiency. |
| `glu_ffn` | `4.8479` | 24.48% | +1.56% | Architectural modification. |
| `bilinear_ffn` | `4.7548` | 25.11% | -0.39% | Architectural modification. |
| `gated_sq_relu` | `4.8001` | 24.72% | +0.56% | Architectural modification. |
| `act_relu` | `4.8353` | 24.51% | +1.30% | Architectural modification. |
| `ffn_ratio_2` | `4.7810` | 24.87% | +0.16% | Architectural modification. |
| `ffn_ratio_6` | `4.7686` | 24.96% | -0.10% | Architectural modification. |


### Attention Mechanics
| Experiment | Val Loss | Val Acc | Delta vs Baseline | Note / Explanation |
|------------|----------|---------|-------------------|--------------------|
| `full_mha` | `4.7996` | 24.69% | +0.55% | Architectural modification. |
| `mqa` | `4.8136` | 24.51% | +0.84% | Architectural modification. |
| `no_rope` | N/A (Failed) | N/A | N/A | CUDA Kernel Access Error |
| `learned_pos_embed` | N/A (Failed) | N/A | N/A | CUDA Kernel Access Error |
| `rope_base_1M` | `4.7661` | 24.98% | -0.15% | Context representations. |
| `attn_bias` | `4.7870` | 24.79% | +0.29% | Architectural modification. |


### Block Structure
| Experiment | Val Loss | Val Acc | Delta vs Baseline | Note / Explanation |
|------------|----------|---------|-------------------|--------------------|
| `parallel_block` | `4.7693` | 24.97% | -0.08% | Parallel block. |
| `residual_scale_05` | `4.7353` | 25.37% | -0.80% | Architectural modification. |


### Depth vs Width Trade-offs
| Experiment | Val Loss | Val Acc | Delta vs Baseline | Note / Explanation |
|------------|----------|---------|-------------------|--------------------|
| `deeper_narrower` | `4.8401` | 24.39% | +1.40% | Architectural modification. |
| `shallower_wider` | `4.7257` | 25.45% | -1.00% | Architectural modification. |


### Weight Initialization
| Experiment | Val Loss | Val Acc | Delta vs Baseline | Note / Explanation |
|------------|----------|---------|-------------------|--------------------|
| `depth_scaled_init` | `4.7817` | 25.05% | +0.17% | Architectural modification. |
| `gpt2_init` | `4.8108` | 24.80% | +0.78% | Architectural modification. |
| `small_embed_init` | `4.7780` | 24.88% | +0.10% | Architectural modification. |


### Weight Tying
| Experiment | Val Loss | Val Acc | Delta vs Baseline | Note / Explanation |
|------------|----------|---------|-------------------|--------------------|
| `no_weight_tying` | `5.0484` | 22.97% | +5.76% | Architectural modification. |


### Optimizer & Regularization Schedule
| Experiment | Val Loss | Val Acc | Delta vs Baseline | Note / Explanation |
|------------|----------|---------|-------------------|--------------------|
| `muon_ns_10` | `4.7786` | 24.87% | +0.11% | Architectural modification. |
| `low_muon_lr` | `4.7182` | 25.59% | -1.16% | Learning rate and schedule tuning. |
| `low_adam_lr` | `4.7788` | 24.76% | +0.11% | Learning rate and schedule tuning. |
| `linear_schedule` | `4.7277` | 25.18% | -0.96% | Learning rate and schedule tuning. |
| `no_grad_clip` | `4.8306` | 24.48% | +1.20% | Architectural modification. |
| `dropout_01` | `4.8535` | 24.26% | +1.68% | Architectural modification. |


### Combination & 'Best-of'
| Experiment | Val Loss | Val Acc | Delta vs Baseline | Note / Explanation |
|------------|----------|---------|-------------------|--------------------|
| `swiglu_post_norm` | `7.6616` | 4.30% | +60.51% | Normalization variants. |
| `swiglu_layernorm` | `4.7053` | 25.90% | -1.42% | Normalization variants. |
| `parallel_swiglu` | `4.7219` | 25.60% | -1.08% | SwiGLU gating efficiency. |
| `full_mha_swiglu` | `4.7442` | 25.24% | -0.61% | SwiGLU gating efficiency. |
| `gpt2_style` | `7.6599` | 4.30% | +60.47% | Architectural modification. |


---

## 💡 Key Architectural Insights
1. **Normalization is extremely sensitive:** Removing Normalization entirely (`no_norm`) immediately collapses the model to `NaN`. Post-Normalization (`post_norm`, `layer_norm_post`) results in severe degradation (+60% higher loss).
2. **SwiGLU + Pre-LayerNorm works best:** `swiglu_layernorm` created the singular best performing architecture, combining gating with the expressive learnable affine transform of layer normalization.
3. **Width vs Depth Trade-offs:** `shallower_wider` outperformed `deeper_narrower`. Allocating more dimension `d_model` resulted in fundamentally better feature maps than stretching the network depth.
4. **Attention Modifications Check:** `full_mha` (no GQA) didn't dramatically improve loss, showing Grouped Query Attention provides an exceptionally good pareto-frontier.