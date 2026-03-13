# Research Strategy

AI-maintained log of what each batch tested, what was learned, and what comes next. Most recent entry at top.

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
