# Research Strategy

AI-maintained log of what each batch tested, what was learned, and what comes next. Most recent entry at top.

---

## 2026-03-13 — System Setup + Current State Summary

### What we know so far

**Confirmed wins (10M tokens, pre-norm baseline):**
- `swiglu_sandwich` — +2.20% — **new g2 baseline**
- `swiglu_layernorm` — +1.42%
- `scispace_tripleprojglu` — +1.16%
- `low_muon_lr` (0.018) — +1.15%
- `parallel_swiglu` — +1.08%
- `act_gelu` — +1.04%
- `sandwich_norm` — +1.03%
- `shallower_wider` — +1.00%

**Confirmed as load-bearing (regressions if removed):**
- QK-norm (removing = −1.94%)
- Embed scale (removing = −3.02%)
- Final norm (removing = divergence)
- Pre/sandwich norm position (post = divergence without warmup)
- Weight tying (removing = −5.76%)
- Muon momentum (removing = −2.55%)

**Confirmed failures (don't retry without specific new hypothesis):**
- `glu_ffn`, `dropout_01`, `swiglu_swiglu`, `swiglu_deep`
- All post-LN variants without warmup

### Current batch (running)
- ~57 attention mechanism experiments at 10M (softcap, windowed, pooling, GQA, QK variants)
- ~200 g2_* hyperparameter sweeps at 6M — **UNRELIABLE due to config dispatch bug**

### Next batch priorities
1. **Fix config dispatch bug** in `configs/ablation_configs.py` — all 6M g2_* results invalid
2. **Validate 6M top performers at 10M**: `attn_qk_layernorm`, `attn_deepnorm_scale`
3. **Test optimizer on g2 baseline**: `muon_lr=0.018` on new baseline (currently only tested on old)
4. **Combo experiment**: QK-LayerNorm + TripleProjGLU (stack two orthogonal winners)
5. **Exploration**: `layer_scale_init=0.001` (CaiT paper, not yet tried)

### Open questions
- Does `shallower_wider` still win on g2 baseline?
- Do any of the attention experiments (softcap, windowed) show wins at 10M?
- Does `muon_lr=0.018` still help when g2 baseline already has better convergence?
- Can post_norm work with proper warmup?

---
