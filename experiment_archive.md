# 📓 Experiment Archive: All Attempts & Ablations

This document serves as a comprehensive registry of all tried architectural variants. To avoid redundant experimentation, review this list before starting a new run.

## 🚀 FFN & Gating Variations

| Experiment ID | Description | Result vs Baseline | Verdict |
|:--------------|:------------|:-------------------|:--------|
| **swiglu_sandwich** | RMSNorm added after gating multiplication. | **-2.20% to -2.92%** | **WINNER** |
| **swiglu_full** | Standard SwiGLU with wide expansion ratio. | -1.06% | Strong |
| **geglu** | GELU-gated linear unit. | -0.61% | Good |
| **reglu** | ReLU-gated linear unit (sparse). | -0.25% | Neutral |
| **swiglu_narrow** | Parameter-efficient hidden scaling. | -0.05% | Neutral |
| **swiglu_shared_gate**| Tied weights between gate and value projections. | +0.08% | Fail |
| **swiglu_swiglu** | Two stacked SwiGLU blocks in one layer. | +3.37% | Major Fail |

## 🧪 SciSpace-Inspired Activations

| Experiment ID | Key Feature | Result vs Baseline | Verdict |
|:--------------|:------------|:-------------------|:--------|
| **scispace_tripleprojglu** | 3 independent projections (Triple-GLU). | -1.16% | Success |
| **scispace_hardswishglu** | HardSwish activation for gating. | -1.00% | Success |
| **scispace_celuglu** | CELU activation for gating. | -0.89% | Success |
| **scispace_scalegate** | Learned scalar on gating branch. | -0.48% | Neutral |
| **scispace_singlu** | Periodic (Sine) gating activation. | +0.25% | Fail |
| **scispace_laplaceglu** | Laplace CDF gating activation. | +2.93% | Major Fail |

## 🏗️ Structural & Normalization Ablations

| Experiment ID | Focus Area | Result vs Baseline | Verdict |
|:--------------|:-----------|:-------------------|:--------|
| **swiglu_parallel** | PaLM-style parallel FFN+Attn blocks. | -0.71% | Success |
| **post_norm** | Post-LN position (BERT/GPT-1 style). | Significant Regr. | Fail |
| **no_qk_norm** | Removed Query-Key normalization. | Regressive | Fail |
| **no_rope** | Removed Rotary Positional Embeddings. | CUDA Errors | Skip |
| **swiglu_deep** | 2 sequential reduced SwiGLU sub-layers. | +1.55% | Fail |

## ⚙️ Optimizer & Training Settings

| Experiment ID | Change | Result vs Baseline | Verdict |
|:--------------|:-------|:-------------------|:--------|
| **muon_ns_10** | 10 polar express steps (Orthogonalization). | Minor Perf Gain | Neutral |
| **high_muon_lr** | Muon LR = 0.048 (2x baseline). | Unstable | Fail |
| **no_weight_decay** | 0.0 weight decay. | Early Overfitting | Fail |

## 🧠 Attention Mechanism Experiments (Current Batch)
*Phase: Debugging (5k tokens) and initial valuation (10M tokens).*
*New Baseline: **SwiGLU Sandwich** (-2.92% over original).*

| Experiment ID | Category | Description | Verdict |
|:--------------|:---------|:------------|:--------|
| **attn_qk_norm_only** | Normalization | Tests Q-only and K-only normalization variants. | TBD |
| **attn_softcap_X** | Score Scoring | Gemma-style logit softcapping (range 10-55). | TBD |
| **attn_window_X** | Sparsity | Local windowed attention (32-320 tokens). | TBD |
| **attn_poly_X** | Kernel | Polynomial kernel scoring (order 2, 3). | TBD |
| **attn_shared_qkv** | Structural | Shared projection weights for Q, K, and V. | TBD |
| **attn_hilo_X** | Architecture | HiLo-style frequency splitting of heads. | TBD |
| **attn_pool_kX** | Efficiency | Temporal KV pooling (factors 2, 4, 8). | TBD |
| **attn_act_X** | Activation | ReLU, Squared ReLU, GELU attention scoring. | TBD |

---
*Note: Comparisons are relative to the Squared ReLU Baseline at 10M tokens.*
