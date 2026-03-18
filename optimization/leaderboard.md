# Leaderboard

This leaderboard is scoped to LR scaling only. Rankings across durations are used to study transfer, not to expand into other hyperparameters.

## 5s (~250K tokens, ~15 steps)
| Rank | Exp ID | Val Loss | Changes |
|------|--------|----------|---------|
| 1 | 5s_bs4_lr0.008 | 6.7637 | muon_lr=0.008, adamw_lr=0.002 |
| 2 | 5s_bs4_lr0.012 | 6.7772 | muon_lr=0.012, adamw_lr=0.003 |
| 3 | 5s_bs4_lr0.006 | 6.8935 | muon_lr=0.006, adamw_lr=0.0015 |

## 10s (~500K tokens, ~31 steps)
| Rank | Exp ID | Val Loss | Changes |
|------|--------|----------|---------|
| 1 | 10s_bs4_lr0.006 | 6.5143 | muon_lr=0.006, adamw_lr=0.0015 |
| 2 | 10s_bs4_lr0.008 | 6.5250 | muon_lr=0.008, adamw_lr=0.002 |
| 3 | 10s_bs4_lr0.012 | 6.5508 | muon_lr=0.012, adamw_lr=0.003 |

## 20s (~1M tokens, ~62 steps)
| Rank | Exp ID | Val Loss | Changes |
|------|--------|----------|---------|
| 1 | 20s_bs4_lr0.006 | 6.2623 | muon_lr=0.006, adamw_lr=0.0015 |
| 2 | 20s_bs4_lr0.008 | 6.2695 | muon_lr=0.008, adamw_lr=0.002 |
| 3 | 20s_bs4_lr0.012 | 6.2737 | muon_lr=0.012, adamw_lr=0.003 |

## Transfer Table
| Muon LR | 5s Rank | 10s Rank | 20s Rank | Current Read |
|--------|---------|----------|----------|--------------|
| 0.006 | 3 | 1 | 1 | Becomes best by 10s and stays best at 20s |
| 0.008 | 1 | 2 | 2 | Best at 5s, but not the 20s winner |
| 0.012 | 2 | 3 | 3 | Competitive at 5s, weaker as duration increases |

## Takeaway
- `5s` alone does not identify the `20s` winner in the clean rerun
- `10s` does identify the `20s` winner
- `20s` difference between `0.006` and `0.008` is small enough that a tie-break multi-seed check is reasonable if you want a stronger publishable claim
