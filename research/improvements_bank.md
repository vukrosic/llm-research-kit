# Improvements Bank

Track of every mechanism that showed a real, above-noise improvement over its baseline.
Use this when designing experiments for larger models or different architectures.

**Noise threshold:** Δ > 0.002 to count as real. Entries below that line aren't here.

---

## Already in the Active Baseline

These work — they're cumulative in the current best config (`gen12_warm150`, val_loss 4.7888).
Carry all of these forward to any new model.

| mechanism | flag(s) | Δ when introduced | notes |
|-----------|---------|-------------------|-------|
| QK normalization | `use_qk_norm=True`, `qk_norm_type=layernorm` | +0.0305 | Normalize Q and K before dot product. One of the biggest single wins found. |
| DeepNorm / residual scaling | `residual_scale=0.5` | +0.0240 → refined to 0.5 | Pre-scale residuals by α<1. 0.5 beat 0.707. 0.30 is close but 0.5 wins when combined with warm150. |
| SinGLU FFN | `ffn_type=scispace_singlu` | +0.0097 | sin(Wx)*Vx — periodic gating outperforms monotonic gates (bilinear, SwiGLU, ELU). Exploration win from gen11. |
| Linear LR schedule + short warmup | `schedule_type=linear`, `warmup_ratio=0.02` | +0.0541 | Biggest single gain found. Linear decay from peak to near-zero beats cosine. |
| Muon LR 0.018 | `muon_lr=0.018` | +0.0108 | Slightly lower than default 0.02; cleaner convergence on linear schedule. |
| Bias terms everywhere | `use_bias=True` | +0.0185 | Add bias to all Linear layers. Rarely used in modern LLMs, clear win here. |
| Sandwich norm | `norm_position=sandwich` | — | Pre-norm + post-norm per block. Tested early, never beaten. |
| GQA with 4 KV heads | `n_kv_heads=4` | — | Current default; balances memory and quality better than full MHA at 88M. |
| Muon warm momentum | `muon_warm_momentum=True`, `muon_warm_momentum_steps=150` | +0.0125 (+steps: +0.0124) | Ramp momentum 0.5→0.95 over first 150 steps. 150 > 200 > 100 > 50. |
| Muon row norm | `muon_row_norm=True` | +0.0189 | L2-normalize each row of gradient before orthogonalization. |
| Muon RMS norm grad | `muon_rms_norm_grad=True` | +0.0042 (alone) | Divide gradient by RMS. Weak alone but synergizes with row_norm (1.64x synergy). |

**Note:** `gated_residual=True` was a gen9 winner (+0.0104) but was **removed** from the baseline — it conflicts with the row+rms Muon improvements (consistent loser on gen11+ baselines, -0.28% to -0.82%).

---

## Promising — Not Yet in Baseline

These beat their baseline clearly but haven't been incorporated into the current best.

| exp_id | val_loss | tested vs | Δ | mechanism | status |
|--------|----------|-----------|---|-----------|--------|
| `g8_layernorm` | 4.8852 | g7 (4.8948) | **+0.0096** | `norm_type=layernorm` — full LayerNorm instead of RMSNorm | Not tested on gen12 baseline. May interact differently with singlu+warm150. |
| `gen12_rs030` | 4.7917 | gen12 (4.8012) | **+0.0095** | `residual_scale=0.30` — tighter than current 0.5 | Very close to warm150. But rs=0.5+warm150 still beats rs=0.30 alone. |
| `gen12_rs030_warm150` | 4.7915 | gen12 (4.8012) | **+0.0097** | `residual_scale=0.30` + `warm_steps=150` | Nearly ties the record but doesn't beat it. |
| `gen12_trust_warm200` | 4.7939 | gen12 (4.8012) | **+0.0073** | `trust_region` + `warm_steps=200` | Positive but below record. |

---

## Combinations to Try Next

| combination | expected Δ | rationale |
|-------------|-----------|-----------|
| `norm_type=layernorm` on gen12_warm150 baseline | +0.005–0.01 | Was +0.0096 on g7 baseline. Never tested with singlu+warm150+row_rms. Orthogonal mechanism. |
| Longer token budget (12M) for top-3 experiments | unknown | At 367 steps, learning curve analysis shows rank correlation of 0.64 — some experiments change rank late. Longer runs may reveal different winners. |
| Novel optimizer schedules (warmup-stable-decay, cyclic) | unknown | Linear schedule was the biggest single win. Other schedules are unexplored. |

---

## Notes for Larger Models

When scaling to 400M+ parameters or 100B+ tokens, these findings are expected to transfer and possibly strengthen:

- **SinGLU** — periodic activation is a novel finding; test whether it scales or if monotonic gates recover at larger width
- **LayerNorm > RMSNorm** — the +0.0096 win here likely grows at scale; LN's mean subtraction term becomes more important with deeper/wider models
- **Muon row+rms norm** — dual gradient normalization had 1.64x synergy; likely even more important at scale where gradient heterogeneity increases
- **Warm momentum** — optimal ramp length (150/367 = 41% of training) may need adjustment for longer runs
- **residual_scale < 0.5** — deep models benefit more from tight residual scaling to prevent gradient explosion early in training
- **Linear LR schedule** — may need to be tuned for longer runs; cosine may win back at 100B+ tokens

*Last updated: 2026-03-15*
