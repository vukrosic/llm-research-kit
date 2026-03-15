# Frontier Architecture Leaderboard

**Active baseline**: Transformer (ablation system gen12_warm150) — val_loss = 4.7888

This leaderboard compares architectures across families. The goal: find something that beats the transformer.

## Current Rankings

| Rank | exp_id | Family | val_loss | Δ vs transformer | Params | Complexity | Recurrent |
|------|--------|--------|----------|------------------|--------|------------|-----------|
| 1 | gen12_warm150 | transformer | 4.7888 | baseline | 88M | O(n²) | No |

*No frontier experiments completed yet. Run `python -m frontier.experiments.run_frontier` to begin.*

## Family Leaderboard

Best result per architecture family:

| Family | Best exp_id | val_loss | Δ vs transformer |
|--------|-------------|----------|------------------|
| transformer | gen12_warm150 | 4.7888 | baseline |

## History

Entries are added here as experiments beat the previous best in their family.
