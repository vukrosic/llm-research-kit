# Leaderboard

**Current best / active baseline**: `g6_muon_lr_018` — val_loss **4.9133**
**Original baseline**: `baseline` — val_loss **5.0611**
*Last updated: 2026-03-14*

| Rank | exp_id | val_loss | Δ vs previous record | % improvement | Key change |
|------|--------|----------|-----------------------|---------------|------------|
| 1 | `g6_muon_lr_018` | 4.9133 | +0.0108 | +0.22% | `muon_lr=0.018` — lower Muon learning rate on the residual_scale=0.5+linear baseline yields cleaner convergence. |
| 2 | `opt_linear_residual_stack` | 4.9241 | +0.0087 | +0.18% | `residual_scale=0.5` stacked on top of linear schedule baseline — tighter DeepNorm scaling compounds with LR decay. |
| 2 | `opt_linear_combo` | 4.9328 | +0.0541 | +1.09% | `schedule_type=linear` + `warmup_ratio=0.02` on top of bilinear+DeepNorm — linear LR decay with a short warmup appears to significantly improve convergence quality over constant LR. |
| 2 | `combo_deepnorm_bilinear` | 4.9869 | +0.0197 | +0.39% | `ffn_type=bilinear` + `residual_scale=0.707` — combining bilinear FFN and DeepNorm scaling compound over either change alone. |
| 3 | `new_deepnorm_07` | 5.0066 | +0.0240 | +0.48% | `residual_scale=0.707` — DeepNorm-style residual scaling stabilizes training and improves final loss. |
| 4 | `attn_qk_layernorm` | 5.0306 | +0.0305 | +0.60% | `qk_norm_type=layernorm` — normalizing queries and keys before attention improves training stability. |
