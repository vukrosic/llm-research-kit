# Insights

## What Works
- Lower LR than default at all three active durations
- Keeping `adamw_lr = muon_lr / 4` is a reasonable default for this phase

## What Doesn't
- `muon_lr >= 0.048` is consistently bad and stays banned for this study

## Current Evidence
### Tier winners
| Duration | Best observed Muon LR |
|----------|------------------------|
| 5s | 0.008 |
| 10s | 0.006 |
| 20s | 0.006 |

### Working interpretation
- In the clean transfer batch, `5s` prefers `0.008`, while both `10s` and `20s` prefer `0.006`
- `5s` alone does not recover the exact `20s` winner in this rerun
- `10s` does recover the `20s` winner
- The active question is still transfer quality, not the exact functional form

## Current Rule
- Keep schedule constant
- Compare only LR transfer across `5s`, `10s`, and `20s`
- Use a fixed LR grid across all active durations
- Use two seeds for the main benchmark
- Use extra seeds beyond that only as tie-breakers for near-equal results

## Fresh Transfer Batch
| Duration | 1st | 2nd | 3rd |
|----------|-----|-----|-----|
| 5s | `0.008` (`6.7637`) | `0.012` (`6.7772`) | `0.006` (`6.8935`) |
| 10s | `0.006` (`6.5143`) | `0.008` (`6.5250`) | `0.012` (`6.5508`) |
| 20s | `0.006` (`6.2623`) | `0.008` (`6.2695`) | `0.012` (`6.2737`) |

Interpretation:
- `5s -> 20s` misses the exact winner by one slot
- `10s -> 20s` matches the winner exactly
- `20s` gap between `0.006` and `0.008` is `0.0072`, so a tie-break multi-seed run is optional but defensible if you want to publish the claim more confidently

## Next Benchmark
- Fixed grid: `0.005, 0.006, 0.007, 0.008, 0.010, 0.012`
- Durations: `5s`, `10s`, `20s`
- Seeds: `42`, `137`
- Main goal: answer whether `5s` is good for elimination and whether `10s` is the minimum reliable selection tier

## Fixed-Grid Benchmark Result
### Mean ranking by duration
| Duration | 1st | 2nd | 3rd |
|----------|-----|-----|-----|
| 5s | `0.007` (`6.7623`) | `0.008` (`6.7635`) | `0.010` (`6.7706`) |
| 10s | `0.007` (`6.5239`) | `0.005` (`6.5260`) | `0.006` (`6.5266`) |
| 20s | `0.006` (`6.2679`) | `0.007` (`6.2685`) | `0.005` (`6.2719`) |

### Protocol metrics
- `5s` exact winner match: no
- `10s` exact winner match: no
- `5s -> 20s` regret: `0.000656`
- `10s -> 20s` regret: `0.000656`
- Spearman rank correlation `5s vs 20s`: `-0.0857`
- Spearman rank correlation `10s vs 20s`: `0.6571`

### Interpretation
- `5s` is not reliable for exact winner selection
- `10s` is better aligned with `20s` than `5s`, but still misses the exact winner in the two-seed mean
- The practical good news is that both `5s` and `10s` have extremely small regret relative to the `20s` winner
- The exact winner at `20s` (`0.006`) is very close to `0.007`, so protocol quality should be judged more by regret and containment than by exact-match alone

## Open Questions
- Is top-3 containment a better practical metric than top-2 for this setup?
- Is `0.006` meaningfully better than `0.007`, or are they effectively tied at `20s`?
- Should the practical protocol target a narrow final bracket around `0.006-0.007` rather than a single winner?
