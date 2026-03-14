# Improvements Bank

Track of every mechanism that showed a real, above-noise improvement over its baseline.
Use this when designing experiments for larger models or different architectures.

**Noise threshold:** Δ > 0.002 to count as real. Entries below that line aren't here.

---

## Already in the Active Baseline

These work — they're cumulative in the current best config (`g9_gated_residual`, val_loss 4.8844).
Carry all of these forward to any new model.

| mechanism | flag(s) | Δ when introduced | notes |
|-----------|---------|-------------------|-------|
| QK normalization | `use_qk_norm=True`, `qk_norm_type=layernorm` | +0.0305 | Normalize Q and K before dot product. One of the biggest single wins found. |
| DeepNorm / residual scaling | `residual_scale=0.5` | +0.0240 → refined to 0.5 | Pre-scale residuals by α<1. 0.5 beat 0.707. Possibly 0.35 beats 0.5 (see below). |
| Bilinear FFN | `ffn_type=bilinear` | +0.0197 | Gate×Up with no activation. Consistently beats SwiGLU at this scale. |
| Linear LR schedule + short warmup | `schedule_type=linear`, `warmup_ratio=0.02` | +0.0541 | Biggest single gain found. Linear decay from peak to near-zero beats cosine. |
| Muon LR 0.018 | `muon_lr=0.018` | +0.0108 | Slightly lower than default 0.02; cleaner convergence on linear schedule. |
| Bias terms everywhere | `use_bias=True` | +0.0185 | Add bias to all Linear layers. Rarely used in modern LLMs, clear win here. |
| Sandwich norm | `norm_position=sandwich` | — | Pre-norm + post-norm per block. Tested early, never beaten. |
| GQA with 4 KV heads | `n_kv_heads=4` | — | Current default; balances memory and quality better than full MHA at 88M. |
| Gated residual | `gated_residual=True` | +0.0104 | Learned sigmoid gate (scalar per block) on residual connections. Lets each layer control its own residual contribution. Init = sigmoid(0) = 0.5. |

---

## Promising — Not Yet in Baseline

These beat their baseline clearly but haven't been incorporated yet.
High-value targets for the next round or for testing on a larger model.

| exp_id | val_loss | tested vs | Δ | mechanism | status |
|--------|----------|-----------|---|-----------|--------|
| `g8_layernorm` | 4.8852 | g7 (4.8948) | **+0.0096** | `norm_type=layernorm` — full LayerNorm instead of RMSNorm for all norms | Not yet tested on g9 baseline. High priority. |
| `g9_residual_035` | 4.8853 | g7 (4.8948) | **+0.0095** | `residual_scale=0.35` — tighter residual scaling than current 0.5 | Gen10 is testing this combined with gated residual. |
| `g9_bilinear_elu_gated_res` | 4.8862 | g7 (4.8948) | **+0.0086** | `ffn_type=bilinear_elu` + `gated_residual=True` — ELU-gated bilinear FFN stacked with gated residual | Gen10 is testing further variations of this combo. |
| `g9_bilinear_elu_rs04` | 4.8869 | g7 (4.8948) | **+0.0079** | `ffn_type=bilinear_elu` + `residual_scale=0.4` — ELU gate on bilinear with tighter residual | Gen10 is testing with rs=0.35 as well. |
| `g8_final_layernorm` | 4.8916 | g7 (4.8948) | **+0.0033** | `final_norm_type=layernorm` — only the output norm changed to LayerNorm | Lighter version of g8_layernorm; easier to stack. Not yet tested on g9. |

---

## Combinations to Try Next

Built from mechanisms above that haven't been tested together yet.

| combination | expected Δ | rationale |
|-------------|-----------|-----------|
| `norm_type=layernorm` + `gated_residual=True` | +0.01–0.02 | g8_layernorm (+0.0096) and g9_gated_residual (+0.0104) are independent mechanisms — good stacking candidate. |
| `norm_type=layernorm` + `gated_residual=True` + `residual_scale=0.35` | +0.015–0.025 | Three-way stack of top independent wins. |
| `ffn_type=bilinear_elu` + `gated_residual=True` + `residual_scale=0.35` | +0.012–0.020 | All three Gen9 runner-up wins together. |
| `gate_per_channel=True` + `residual_scale=0.35` | +0.005–0.015 | Per-channel gates (d_model scalars instead of scalar per block) may give more expressiveness. |

---

## Notes for Larger Models

When scaling to 400M+ parameters or 100B+ tokens, these findings are expected to transfer and possibly strengthen:

- **LayerNorm > RMSNorm** — the +0.0096 win here likely grows at scale; LN's mean subtraction term becomes more important with deeper/wider models
- **Gated residual** — adaptive per-layer residual strength is architecturally sound; expect it to help more with 40+ layer models
- **residual_scale < 0.5** — deep models benefit more from tight residual scaling to prevent gradient explosion early in training
- **bilinear_elu gate** — ELU's non-saturating positive region may help gradient flow in very deep models more than pure bilinear
- **Linear LR schedule** — may need to be tuned for longer runs; cosine may win back at 100B+ tokens

*Last updated: 2026-03-14*
