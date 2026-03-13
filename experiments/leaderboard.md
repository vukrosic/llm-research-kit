# Leaderboard

**Current best**: `attn_qk_layernorm` — val_loss **5.0306** (+0.60% over baseline)
**Active baseline**: `baseline` — val_loss **5.0611** at 6M tokens
*Last updated: 2026-03-13*

| Rank | exp_id | val_loss | Δ vs baseline | % improvement | Key change |
|------|--------|----------|---------------|---------------|------------|
| 1 | `attn_qk_layernorm` | 5.0306 | +0.0305 | +0.60% | QK normalization: LayerNorm instead of RMSNorm |
