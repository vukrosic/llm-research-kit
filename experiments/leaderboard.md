# Leaderboard

**Current best / active baseline**: `new_bilinear` — val_loss **5.0052** (+1.10% over original baseline)
**Previous baseline**: `attn_qk_layernorm` — val_loss **5.0306**
**Original baseline**: `baseline` — val_loss **5.0611**
*Last updated: 2026-03-13*

| Rank | exp_id | val_loss | Δ vs original baseline | % improvement | Key change |
|------|--------|----------|------------------------|---------------|------------|
| 1 | `new_bilinear` | 5.0052 | +0.0559 | +1.10% | `qk_norm_type=layernorm` + `ffn_type=bilinear` |
| 2 | `new_deepnorm_07` | 5.0066 | +0.0545 | +1.08% | `qk_norm_type=layernorm` + `residual_scale=0.707` |
| 3 | `attn_qk_layernorm` | 5.0306 | +0.0305 | +0.60% | `qk_norm_type=layernorm` |
