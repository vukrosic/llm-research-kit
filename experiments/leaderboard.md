# Leaderboard

**Current best / active baseline**: `muon_warm_row_rms` — val_loss **4.8109**
**Original baseline**: `baseline` — val_loss **5.0611**
*Last updated: 2026-03-14*

| Rank | exp_id | val_loss | Δ vs previous record | % improvement | Key change |
|------|--------|----------|-----------------------|---------------|------------|
| 1 | `muon_warm_row_rms` | 4.8109 | +0.0379 | +0.78% | `muon_row_norm=True` + `muon_rms_norm_grad=True` on warm-momentum baseline — row-normalize then RMS-normalize gradients before Muon orthogonalization. Dual normalization removes both per-neuron magnitude and global scale, making ortho purely directional. |
| 2 | `g9_muon_warm_mom` | 4.8488 | +0.0125 | +0.26% | `muon_warm_momentum=True` — ramp Muon momentum from 0.5→0.95 over first 100 steps. Early training uses lower momentum for more responsive updates, then ramps to full momentum. |
| 3 | `g9_muon_row_norm` | 4.8613 | +0.0189 | +0.39% | `muon_row_norm=True` — L2-normalize each row of gradient before polar express orthogonalization. Removes per-neuron magnitude, making ortho purely directional. |
| 4 | `g9_muon_rms_norm` | 4.8802 | +0.0042 | +0.09% | `muon_rms_norm_grad=True` — divide gradient by its RMS before orthogonalization. Removes gradient scale heterogeneity. |
| 5 | `g9_gated_residual` | 4.8844 | +0.0104 | +0.21% | `gated_residual=True` — learned sigmoid gates on residual connections (per-block scalar parameter) let each layer control its own residual strength dynamically. |
| 6 | `g7_use_bias` | 4.8948 | +0.0185 | +0.38% | `use_bias=True` — adding bias terms to all linear layers, rarely done in modern LLMs, yields a clear improvement. |
| 7 | `g6_muon_lr_018` | 4.9133 | +0.0108 | +0.22% | `muon_lr=0.018` — lower Muon learning rate on the residual_scale=0.5+linear baseline yields cleaner convergence. |
| 8 | `opt_linear_residual_stack` | 4.9241 | +0.0087 | +0.18% | `residual_scale=0.5` stacked on top of linear schedule baseline — tighter DeepNorm scaling compounds with LR decay. |
| 9 | `opt_linear_combo` | 4.9328 | +0.0541 | +1.09% | `schedule_type=linear` + `warmup_ratio=0.02` on top of bilinear+DeepNorm — linear LR decay with a short warmup appears to significantly improve convergence quality over constant LR. |
| 10 | `combo_deepnorm_bilinear` | 4.9869 | +0.0197 | +0.39% | `ffn_type=bilinear` + `residual_scale=0.707` — combining bilinear FFN and DeepNorm scaling compound over either change alone. |
| 11 | `new_deepnorm_07` | 5.0066 | +0.0240 | +0.48% | `residual_scale=0.707` — DeepNorm-style residual scaling stabilizes training and improves final loss. |
| 12 | `attn_qk_layernorm` | 5.0306 | +0.0305 | +0.60% | `qk_norm_type=layernorm` — normalizing queries and keys before attention improves training stability. |
