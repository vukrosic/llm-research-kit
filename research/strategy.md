# Research Strategy

AI-maintained log of what each batch tested, what was learned, and what comes next. Most recent entry at top.

---

## 2026-03-14 — Gen11 results: new baseline gen11_x_singlu (4.8012)

### What this batch tested
50 experiments on `muon_warm_row_rms` baseline (4.8109). Exploitation (35): gated_residual stacking, grad_centralize, bilinear_elu, residual_scale sweep, trust_region, double_ortho, warm_momentum tuning, and two-way combos. Exploration (15): col_norm, sign_mix, stoch_ortho, half_ortho, scispace FFNs (singlu/tanhglu/laplaceglu/tripleprojglu), cosine_attn, HiLo attention.

### Results: 8 winners, 8 neutral, 28 losers, 6 failed
| exp_id | val_loss | Δ | key change |
|--------|----------|---|------------|
| gen11_x_singlu | 4.8012 | +0.0097 | scispace_singlu FFN (sin-gated!) — EXPLORATION WIN |
| gen11_rs030 | 4.8034 | +0.0075 | residual_scale=0.30 |
| gen11_rs045 | 4.8057 | +0.0052 | residual_scale=0.45 |
| gen11_trust_rs035 | 4.8062 | +0.0047 | trust_region + residual_scale=0.35 |
| gen11_rs035 | 4.8067 | +0.0042 | residual_scale=0.35 |
| gen11_warm_200 | 4.8068 | +0.0041 | warm_momentum_steps=200 |
| gen11_elu_rs035 | 4.8075 | +0.0034 | bilinear_elu + residual_scale=0.35 |
| gen11_trust | 4.8077 | +0.0032 | trust_region alone |

### Key findings
- **SinGLU is the new champion.** `scispace_singlu` (sin(Wx)*Vx) beat all 35 exploitation experiments. Periodic activation provides fundamentally different representation capacity than monotonic gates. This was pure exploration.
- **Residual scale 0.30 is the optimal static scale.** 0.30 > 0.35 > 0.45 > 0.40 > 0.50 (current) > 0.60. The trend is clear: tighter DeepNorm helps, with 0.30 as the sweet spot.
- **Gated residual is a CONSISTENT LOSER on this baseline.** Every single gated_residual combo was negative (-0.28% to -0.82%). It was a gen9 winner but conflicts with the row+rms Muon improvements. Add to banned list for this baseline.
- **warm_momentum_steps=200 > 100 > 50.** Longer warm ramp is better. 50 was catastrophic (-0.54%).
- **grad_centralize was neutral** (+0.0% alone). Disappointing given it was #2 in the previous batch — the improvement was likely from row_norm, not grad_center itself.
- **tanhglu was neutral** — another scispace FFN, not as special as singlu.
- **sign_mix is catastrophic** (-15% to -16%). **half_ortho is a big loser** (-1% to -2%). **laplaceglu big loser** (-3.4%).
- **post_momentum crashed** (dtype bug). **cosine_attn and hilo_050 OOM.**
- **hilo_025 is anomalous** (val_loss 0.007, 99.9% accuracy) — broken eval, exclude like attn_pool.

### Failures and anomalies
- `gen11_post_mom*` (4 experiments): dtype bug in muon_post_momentum implementation
- `gen11_x_cosine_attn`, `gen11_x_hilo_050`: CUDA OOM
- `gen11_x_hilo_025`: anomalous val_loss 0.007 — broken eval, exclude

### New baseline: gen11_x_singlu (4.8012)
Flags: all of muon_warm_row_rms baseline + `ffn_type=scispace_singlu` instead of `ffn_type=bilinear`

### What the next batch should focus on
- **Exploit singlu aggressively:** singlu + rs030, singlu + trust_region, singlu + warm_200
- **Explore more scispace FFNs:** only 4 were tested out of 20+ variants. Try eluglu, hardswishglu, sigmoidglu, pregateglu, etc.
- **Residual scale on singlu baseline:** is 0.30 still optimal with singlu?
- **Fix post_momentum dtype bug** and re-test — it was untested
- **Ban gated_residual** on current baseline direction

---

## 2026-03-14 — Gen10 queue batch results: new baseline muon_warm_row_rms (4.8109)

### What this batch tested
48 experiments run via `experiments/queue.json` on the `g9_muon_warm_mom` baseline (4.8488). Three axes:
1. **Muon gradient preprocessing combos** — stacking row_norm, rms_norm, grad_centralize, trust_region, cautious, double_ortho, ema_ortho with warm momentum
2. **FFN gate activations** — bilinear_elu/gaussian/softplus/cubic/mish/star/sqr/sq_silu with row_norm or gated_residual
3. **Structural** — gated_residual variants (cautious, trust_region, update_clip, frob_scale), residual_scale=0.45, pos_q_rope_only

### Results: 12 winners, 7 neutral, 29 losers
| exp_id | val_loss | Δ | key change |
|--------|----------|---|------------|
| muon_warm_row_rms | 4.8109 | +0.0379 | row_norm + rms_norm_grad (dual normalization) |
| muon_warm_row_grad_center | 4.8128 | +0.0360 | row_norm + grad_centralize |
| muon_warm_row_trust | 4.8139 | +0.0349 | row_norm + trust_region |
| muon_warm_row_double_ortho | 4.8149 | +0.0339 | row_norm + double_ortho |
| ffn_elu_row_norm | 4.8167 | +0.0321 | bilinear_elu FFN + muon_row_norm |
| muon_warm_row_norm | 4.8169 | +0.0319 | row_norm (standalone on warm_mom base) |

### Key findings
- **`muon_row_norm` is the dominant Muon improvement.** Every top-6 winner includes it. Row-normalizing gradients before polar express orthogonalization is the single most impactful Muon tweak.
- **Stacking a second normalization on top of row_norm helps.** The top 4 are all `row_norm + X`. RMS norm, grad centralize, trust region, and double ortho all compound with row_norm.
- **RMS-norm alone is weak, but row+rms is the best combo.** `muon_warm_rms_norm` was neutral (+0.04%), but combining it with row_norm yields the batch winner at +0.78%.
- **ELU bilinear FFN + row_norm** (4.8167) beats plain row_norm (4.8169) — bilinear_elu continues to stack positively.
- **Cautious Muon is a catastrophic loser** (-2.6% to -3.2% across all variants). Do not revisit.
- **EMA ortho is a big loser** (-5% to -6%). Do not revisit.
- **Update clip and frob scale are catastrophic** (-12% to -19%). Do not revisit.
- **27 of 48 experiments had Permission Denied crashes** (post-training file save, not training failure). All completed 6M tokens. Results are valid.

### New baseline: muon_warm_row_rms (4.8109)
Flags: `muon_row_norm=True, muon_rms_norm_grad=True, muon_warm_momentum=True, muon_warm_momentum_steps=100` on top of the cumulative best config.

### What the next batch should focus on
- **Exploit row_norm + rms_norm_grad aggressively:** Try stacking with gated_residual, bilinear_elu, residual_scale=0.35, grad_centralize
- **Three-way Muon combos:** row_norm + rms_norm + grad_center was not tested (only row+rms and row+center separately)
- **FFN on new baseline:** bilinear_elu showed promise — re-test on this new baseline
- **Structural exploration:** gate_per_channel, novel attention mechanisms on this stronger baseline

---

## 2026-03-14 — Gen9 results: new baseline g9_gated_residual, Gen10 launched

### Gen9 Winners (vs g7_use_bias baseline 4.8948)
| exp_id | val_loss | Δ | key change |
|--------|----------|---|------------|
| g9_gated_residual | 4.8844 | +0.0104 | Learned sigmoid gates on residual connections |
| g9_residual_035 | 4.8853 | +0.0095 | residual_scale=0.35 (tighter than 0.5) |
| g9_bilinear_elu_gated_res | 4.8862 | +0.0086 | bilinear_elu FFN + gated residual |
| g9_bilinear_elu_rs04 | 4.8869 | +0.0079 | bilinear_elu + residual_scale=0.4 |

### Key findings
- **Gated residual is the clear winner.** Learned sigmoid gates (per-block scalar, init=sigmoid(0)=0.5) that adapt the residual strength per layer beat all other Gen9 mechanisms.
- **residual_scale=0.35** beats 0.4 and 0.5 — worth sweeping further (try 0.25, 0.30).
- **bilinear_elu FFN** consistently appears in Gen9 winners — the ELU gate activation outperforms the pure bilinear gate. Will test in Gen10 combined with gated residual.
- **Frob scale** (-18%), **q_rope_only** (-14%), **cosine_attn** (crashed — alibi OOM issue), **bilinear_softplus/gauss/cubic/sq_silu** — all losers or neutral.
- Muon variants (grad_center, double_ortho, ema_ortho): grad_center slightly negative, double_ortho slightly negative, ema_ortho big loser. Re-running 27 crashed Muon experiments now.

### New baseline: g9_gated_residual (4.8844)
All Gen10 experiments built on top of this config.

### Gen10 strategy (50 experiments — currently running)
- **A (5):** Residual scale sweeps with gated residual: 0.25, 0.35, 0.40, 0.60, 1.0
- **B (4):** Gate init variants: high(2.0), low(-2.0), open(4.0), per_channel
- **C (10):** Gated residual + Muon variants (post_mom, grad_center, cautious, double_ortho, adaptive_ns, warm_mom, rms_norm, row_norm, sign_mix, half_ortho)
- **D (8):** Gated residual + attention/FFN combos (cosine_attn, alibi, qkln, bilinear_elu variants)
- **E+F+G (23):** Novel combos: gated+per_channel+rs035, gated+cosine+qkln, update_clip, trust_region, bilinear_mish, bilinear_sq_silu

### What to watch in Gen10
- If rs035 + gated_residual wins → try 0.30, 0.25
- If gate_per_channel wins → it becomes the default residual mechanism
- If bilinear_elu + gated wins → three-way combo (elu + gated + rs035)
- Any Muon variant that stacks with gated_residual is particularly interesting

---

## 2026-03-14 — New baseline: combo_deepnorm_bilinear (4.9869), Gen4 queue built

### Baseline update
`combo_deepnorm_bilinear` (val_loss **4.9869**, +1.47% over original) is the new best — bilinear FFN + residual_scale=0.707 stack additively. Both `new_bilinear` (5.0052) and `new_deepnorm_07` (5.0066) are independent wins that compose well together.

### Gen4 queue (7 experiments)
**Exploitation (5):** residual sweep around 0.707 (0.5, 0.6), wider bilinear (d_ff=3072), full MHA stacked on combo, cosine schedule on combo.
**Exploration (2):** linear schedule on combo, LayerNorm instead of RMSNorm on combo.

### What to watch
- Residual sweep: if 0.5 or 0.6 beats 0.707, update baseline and try 0.4
- If `ffn_bilinear_wide` wins, try bilinear + deepnorm + wide together
- If schedule experiments win, add warmup to all future combos

---

## 2026-03-13 (update) — Gen3 partial results: two new winners

### New findings
- `new_bilinear` (val_loss **5.0052**) — new #1, beats `attn_qk_layernorm` by 0.0254 (+0.50%)
- `new_deepnorm_07` (val_loss **5.0066**) — new #2, bilinear and deepnorm are orthogonal wins

### Active baseline is now `new_bilinear` (5.0052)
Key: `qk_norm_type=layernorm` + `ffn_type=bilinear` on top of sandwich+swiglu baseline

### Next: run remaining 10 Gen3 experiments
Priority watch: `combo_deepnorm_bilinear` (stacks both winners), `deepnorm_sweep_*` (is 0.5 < 0.707?), `new_ffn_wide` (more bilinear capacity)

---

## 2026-03-13 — Baseline established, winner-based queue built

### Current state
- **Baseline**: `baseline` — val_loss **5.0611** at 6M tokens
- **Best experiment**: `attn_qk_layernorm` — **5.0306** (+0.60%)
- **Runner-up**: `attn_deepnorm_scale` (residual_scale=0.707) — **5.0309** (+0.60%)
- **Marginals worth combining**: `bilinear_ffn` (+0.40%), `full_mha_swiglu` (+0.39%), `attn_bias` (+0.32%)

### What we know
- QK normalization with LayerNorm beats RMSNorm — small but consistent win
- DeepNorm residual scaling (0.707) matches it — different mechanism, same magnitude
- bilinear FFN is competitive with swiglu at this scale
- 166 g2_* experiments are invalid (config dispatch bug) — ignore all g2_ results

### Next batch (queue order)
1. `deepnorm_residual_sweep_05` — does 0.5 beat 0.707?
2. `deepnorm_residual_sweep_03` — does 0.3 beat 0.5?
3. `combo_qklayernorm_deepnorm` — stack the two winners (orthogonal mechanisms)
4. `combo_qklayernorm_bilinear` — stack attn winner + FFN marginal
5. `bilinear_ffn_wide` — more capacity for bilinear
6. `value_norm_qklayernorm` — normalize V vectors too (exploration)
7. `layer_scale_deepnorm` — CaiT layer scale + deepnorm (exploration)

### Decision rules
- If `combo_qklayernorm_deepnorm` wins → it becomes new baseline, sweep residual around it
- If residual sweep finds optimum below 0.707 → new baseline candidate for combos
- If 3+ consecutive exploitations are neutral → add 2 exploration experiments from hypotheses.md

---
