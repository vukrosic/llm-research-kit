# Leaderboard

**Current best**: `swiglu_sandwich` — val_loss **4.6682** (+2.20% over pre-norm baseline)  
**Active baseline for new experiments**: g2 baseline (sandwich + swiglu) — val_loss **4.6682** at 10M, **5.0611** at 6M  
*Last updated: 2026-03-13 | Total experiments run: 84 at 10M, 115 at 6M*

---

## Top 30 — 10M Tokens

| Rank | exp_id | val_loss | Δ vs pre-norm baseline | % improvement | Key change |
|------|--------|----------|------------------------|---------------|------------|
| 1 | `swiglu_sandwich` | 4.6682 | +0.1051 | +2.20% | {'norm_position': 'sandwich', 'ffn_type': 'swiglu'} |
| 2 | `swiglu_layernorm` | 4.7053 | +0.0680 | +1.42% | {'ffn_type': 'swiglu'} |
| 3 | `scispace_tripleprojglu` | 4.7180 | +0.0553 | +1.16% | {'ffn_type': 'scispace_tripleprojglu'} |
| 4 | `low_muon_lr` | 4.7182 | +0.0551 | +1.15% | {'muon_lr': 0.012} |
| 5 | `parallel_swiglu` | 4.7219 | +0.0514 | +1.08% | {'ffn_type': 'swiglu', 'parallel_block': True} |
| 6 | `swiglu_full` | 4.7226 | +0.0507 | +1.06% | {'ffn_type': 'swiglu_full'} |
| 7 | `act_gelu` | 4.7234 | +0.0499 | +1.04% | {'activation_type': 'gelu'} |
| 8 | `sandwich_norm` | 4.7243 | +0.0490 | +1.03% | {'norm_position': 'sandwich'} |
| 9 | `scispace_hardswishglu` | 4.7256 | +0.0478 | +1.00% | {'ffn_type': 'scispace_hardswishglu'} |
| 10 | `shallower_wider` | 4.7257 | +0.0476 | +1.00% | {'n_layers': 14, 'd_model': 576} |
| 11 | `linear_schedule` | 4.7277 | +0.0456 | +0.96% | {'schedule_type': 'linear', 'warmup_ratio': 0.02} |
| 12 | `scispace_celuglu` | 4.7309 | +0.0424 | +0.89% | {'ffn_type': 'scispace_celuglu'} |
| 13 | `no_weight_decay` | 4.7309 | +0.0423 | +0.89% | {'weight_decay': 0.0} |
| 14 | `swiglu` | 4.7347 | +0.0386 | +0.81% | {'ffn_type': 'swiglu'} |
| 15 | `swiglu_bias` | 4.7352 | +0.0381 | +0.80% | {'ffn_type': 'swiglu_bias'} |
| 16 | `residual_scale_05` | 4.7353 | +0.0380 | +0.80% | {'residual_scale': 0.5} |
| 17 | `swiglu_wide` | 4.7375 | +0.0357 | +0.75% | {'ffn_type': 'swiglu_wide'} |
| 18 | `swiglu_parallel` | 4.7396 | +0.0337 | +0.71% | {'ffn_type': 'swiglu', 'parallel_block': True} |
| 19 | `full_mha_swiglu` | 4.7442 | +0.0291 | +0.61% | {'ffn_type': 'swiglu', 'n_kv_heads': 8} |
| 20 | `scispace_eluglu` | 4.7443 | +0.0290 | +0.61% | {'ffn_type': 'scispace_eluglu'} |
| 21 | `scispace_moelite` | 4.7443 | +0.0290 | +0.61% | {'ffn_type': 'scispace_moelite'} |
| 22 | `geglu` | 4.7444 | +0.0289 | +0.61% | {'ffn_type': 'geglu'} |
| 23 | `layer_norm` | 4.7445 | +0.0288 | +0.60% | {} |
| 24 | `scispace_composite` | 4.7457 | +0.0276 | +0.58% | {'ffn_type': 'scispace_composite'} |
| 25 | `swiglu_3q` | 4.7486 | +0.0247 | +0.52% | {'ffn_type': 'swiglu_3q'} |
| 26 | `scispace_scalegate` | 4.7502 | +0.0231 | +0.48% | {'ffn_type': 'scispace_scalegate'} |
| 27 | `bilinear_ffn` | 4.7548 | +0.0185 | +0.39% | {'ffn_type': 'bilinear'} |
| 28 | `schedule_cosine` | 4.7551 | +0.0182 | +0.38% | {'schedule_type': 'cosine', 'warmup_ratio': 0.05} |
| 29 | `scispace_singleprojglu` | 4.7569 | +0.0164 | +0.34% | {'ffn_type': 'scispace_singleprojglu'} |
| 30 | `reglu` | 4.7613 | +0.0120 | +0.25% | {'ffn_type': 'reglu'} |

---

## Bottom 10 — 10M Tokens (Worst Performers)

| Rank | exp_id | val_loss | % vs baseline | Verdict |
|------|--------|----------|---------------|---------|
| 1 | `no_final_norm` | 12.4762 | -161.37% | fail |
| 2 | `swiglu_post_norm` | 7.6616 | -60.51% | fail |
| 3 | `layer_norm_post` | 7.6616 | -60.51% | fail |
| 4 | `gpt2_style` | 7.6599 | -60.47% | fail |
| 5 | `post_norm` | 7.6537 | -60.34% | fail |
| 6 | `no_weight_tying` | 5.0484 | -5.76% | fail |
| 7 | `polar_express_2` | 4.9891 | -4.52% | fail |
| 8 | `swiglu_swiglu` | 4.9342 | -3.37% | fail |
| 9 | `no_embed_scale` | 4.9173 | -3.02% | fail |
| 10 | `scispace_laplaceglu` | 4.9130 | -2.93% | fail |

---

## 6M Token Results — Reliable (non-g2_ experiments)

> Note: All `g2_*` 6M experiments are unreliable due to config dispatch bug. See `CLAUDE.md` §9.

| Rank | exp_id | val_loss | Δ vs g2 baseline | % |
|------|--------|----------|------------------|---|