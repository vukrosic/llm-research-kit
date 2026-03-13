# Hypotheses — Ideas To Test

AI-maintained list of testable ideas, sourced from papers in `inbox/` and user suggestions.
Each entry gets a priority, source, and status. When an experiment is created for it, link the `exp_id`.

---

## Status key
- `queued` — experiment added to queue.json
- `running` — currently training
- `done` — completed, see history.json
- `skipped` — decided not to test (with reason)

---

## Attention Mechanisms

| Priority | Hypothesis | Source | Experiment | Status |
|---|---|---|---|---|
| 1 | QK LayerNorm (instead of RMSNorm) improves training stability at longer sequences | 6M results show `attn_qk_layernorm` best performer | `attn_qk_layernorm` | done — winner at 6M |
| 1 | DeepNorm scale on attention outputs reduces gradient variance | Observed in 6M results | `attn_deepnorm_scale` | done — winner at 6M |
| 2 | GQA with 2 KV heads may be optimal tradeoff for 512-dim model | Mistral paper | `attn_gqa_2` | done at 6M (needs 10M) |
| 2 | Softcap on attention logits (Gemma-style) prevents extreme activations | Gemma2 | `attn_softcap_*` | running (10M pending) |
| 3 | Local windowed attention improves locality bias | Longformer / BigBird | `attn_window_*` | running (10M pending) |
| 3 | Polynomial kernel attention (no softmax) could generalize better | Performer paper | `attn_poly*` | running (10M pending) |
| 3 | HiLo: splitting heads into high/low frequency may improve representation | HiLo paper | `attn_hilo_*` | running (10M pending) |

## FFN / Gating

| Priority | Hypothesis | Source | Experiment | Status |
|---|---|---|---|---|
| 1 | Bilinear FFN (no activation, pure multiplicative) competes with SwiGLU | 10M results | `bilinear_ffn` | done — +0.39% |
| 1 | Triple projection GLU (3 independent projections) beats 2-projection variants | SciSpace ablations | `scispace_tripleprojglu` | done — winner +1.16% |
| 2 | HardSwish gating combines ReLU-like sparsity with smooth gradient | SciSpace | `scispace_hardswishglu` | done — winner +1.00% |
| 2 | CELU gating has negative saturation that may help regularization | SciSpace | `scispace_celuglu` | done — +0.89% |
| 3 | Combining triple-proj GLU + sandwich norm may stack winners | Combination hypothesis | not yet queued | — |

## Normalization

| Priority | Hypothesis | Source | Experiment | Status |
|---|---|---|---|---|
| 1 | Sandwich norm (norm inside FFN block) is consistently the best position | Current g2 baseline | `sandwich_norm` | done — winner +1.03% |
| 2 | Layer scale (small init scalar on residual) helps very deep models | CaiT paper | not yet queued | — |
| 2 | Value norm (normalize attention values before projection) may act as implicit regularization | 6M results show `g2_value_norm` | queued | — |

## Optimizer / Training

| Priority | Hypothesis | Source | Experiment | Status |
|---|---|---|---|---|
| 1 | Lower Muon LR (0.018) is optimal for this architecture | 10M: `low_muon_lr` +1.15% | `g2_muon_lr_*` | partial (6M bug) |
| 2 | Linear warmup schedule (even short 1%) consistently helps | 10M: `linear_schedule` +0.96% | `g2_linear_wu*` | queued |
| 2 | Cosine schedule with 2-5% warmup may outperform constant | schedule sweep | `g2_cosine_wu*` | queued |
| 3 | Stochastic depth (0.05-0.1) provides regularization at this scale | Literature | `g2_sdrop_*` | queued |
| 3 | Label smoothing 0.05 may help generalization without hurting perplexity | Standard trick | `g2_lsmooth_*` | queued |

## Architecture Shape

| Priority | Hypothesis | Source | Experiment | Status |
|---|---|---|---|---|
| 2 | Shallower + wider (fewer layers, larger d_model) might generalize better | 10M: `shallower_wider` +1.00% | `shallower_wider` | done |
| 3 | Optimal depth/width tradeoff may shift with the g2 baseline | Scaling law literature | `g2_shape_*` | queued |

---

*Last updated by AI: 2026-03-13*
