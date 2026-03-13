# 🏆 LLM Ablation Leaderboard (10M Tokens)

This leaderboard tracks architectural improvements tested against a standardized baseline. Each experiment is trained for 10M tokens using identical seeds and data to ensure fair comparison.

## 📊 Rankings

| Rank | Model Variant | Val Loss | Delta % | Mathematical / Architectural Description |
|:-----|:--------------|:---------|:--------|:-----------------------------------------|
| 1 | **SwiGLU Sandwich** | **4.6436** | **-2.92%** | Applies an RMSNorm layer immediately after the element-wise product of the gate and value branches, before the final projection. See [Full Archive](experiment_archive.md) for all tried variants. |
| 2 | **Baseline** | 4.7834 | +0.00% | Standard FFN using a squared ReLU activation function (`ReLU(x)**2`). |

---

## 🔬 Experiment Details

### 1. Baseline
- **Description**: The reference model using a standard Transformer Feed-Forward Network.
- **Activation**: Squared ReLU.
- **Normalization**: Pre-layer RMSNorm.
- **Results**: Replicated across multiple runs with a stable validation loss around ~4.78.

### 2. SwiGLU Sandwich (Improvement)
- **Description**: Combines the SwiGLU gating mechanism with an internal "Sandwich" normalization step.
- **Activation**: SiLU (Swish) gating.
- **Key Change**: `x = RMSNorm(SiLU(xW_gate) * xW_val); output = xW_out`.
- **Insight**: Adding normalization *inside* the gated block significantly stabilizes the representations before the down-projection, leading to superior convergence.

---

## 🔄 Replication Log

| Run ID | Model | Val Loss | Status |
|:-------|:------|:---------|:-------|
| #1 | Baseline | 4.7733 | ✅ Verified |
| #1 | SwiGLU Sandwich | 4.6682 | ✅ Verified |
| #2 (Rep) | Baseline | 4.7834 | ✅ Confirmed |
| #2 (Rep) | SwiGLU Sandwich | 4.6436 | ✅ Confirmed |

*Last updated: 2026-03-13*
