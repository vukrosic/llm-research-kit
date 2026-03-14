# Research Strategy

AI-maintained log of what each batch tested, what was learned, and what comes next. Most recent entry at top.

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
