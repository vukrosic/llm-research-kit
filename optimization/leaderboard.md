# Leaderboard

## 5s (~250K tokens, ~15 steps)
| Rank | Exp ID | Val Loss | Changes |
|------|--------|----------|---------|
| 1 | 5s_lr_0.012 | 6.9551 | muon_lr=0.012, adamw_lr=0.003 |
| 2 | 5s_lr_0.015 | 6.9711 | muon_lr=0.015 |
| 3 | 5s_lr_0.018 | 6.9816 | muon_lr=0.018 |
| 4 | 5s_lr_0.010 | 7.0147 | muon_lr=0.010 |
| 5 | 5s_baseline | 7.0717 | (default lr=0.024) |

## 10s (~500K tokens, ~31 steps)
| Rank | Exp ID | Val Loss | Changes |
|------|--------|----------|---------|
| 1 | 10s_lr_0.008 | 6.6364 | muon_lr=0.008, adamw_lr=0.002 |
| 2 | 10s_lr_0.012 | 6.6547 | muon_lr=0.012 |
| 3 | 10s_lr_0.006 | 6.6687 | muon_lr=0.006 |
| 4 | 10s_baseline | 6.7437 | (default lr=0.024) |

## 20s (~1M tokens, ~62 steps)
| Rank | Exp ID | Val Loss | Changes |
|------|--------|----------|---------|
| 1 | lr_muon_0.006 | 6.3934 | muon_lr=0.006, adamw_lr=0.0015 |
| 2 | lr_muon_0.012 | 6.3976 | muon_lr=0.012 |
| 3 | baseline | 6.4691 | (default lr=0.024) |
