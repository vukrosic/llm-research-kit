# Frontier Architecture Leaderboard

**Active baseline**: Transformer — val_loss = 3.4486

This leaderboard compares architectures across families. The goal: find something that beats the transformer.

## Current Rankings

| Rank | exp_id | Family | val_loss | Δ vs transformer | Params | Speed (ms/step) | Notes |
|------|--------|--------|----------|------------------|--------|-----------------|-------|
| 1 | novel_value_residual_12M | novel | 3.4486 | +0.0000 | 107.4M | 383 | |
| 2 | novel_conv_ts_qknorm_12M | novel | 3.5575 | +0.1088 | 107.4M | 383 | |
| 3 | novel_differential_gqa_12M | novel | 3.5666 | +0.1180 | 110.3M | 467 | |
| 4 | novel_soft_router_12M | novel | 3.6697 | +0.2211 | 99.1M | 416 | |

## History

Last updated: 2026-03-16 02:36
