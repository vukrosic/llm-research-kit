# Insights

## What Works
- Lower LR than default at all durations

## What Doesn't
- muon_lr >= 0.048: always worse, banned
- LR schedules (cosine, linear) at 5s: all worse than constant (only 15 steps, can't afford decay)

## Scaling Laws Discovered
### 1. LR-Duration Power Law
**optimal_muon_lr = 0.012 / sqrt(duration_seconds / 5)**
| Duration | Predicted | Measured | Error |
|----------|-----------|----------|-------|
| 5s | 0.012 | 0.012 | 0% |
| 10s | 0.0085 | 0.008 | 6% |
| 20s | 0.006 | 0.006 | 0% |

Implies: adamw_lr = muon_lr / 4 (ratio preserved)

### 2. Schedule Irrelevance at Short Duration
At 5s (15 steps): constant > linear ≈ cosine ≈ cosine+warmup
Hypothesis: schedules help only when there are enough steps for the decay to matter.
Need to verify: do schedules help at 20s+?

## Open Questions
- Does weight decay interact with LR at different durations?
- Does momentum interact with LR?
- Do schedules help at 20s?
- What predicts longer-run loss from short-run metrics?
- Can we use the LR scaling law to predict optimal LR at 1B tokens?
  - 1B tokens ≈ 20,000s → optimal_lr ≈ 0.012/sqrt(4000) ≈ 0.00019
