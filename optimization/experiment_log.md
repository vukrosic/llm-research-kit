# Experiment Log

## Active Scope
- Only Muon LR scaling across `5s`, `10s`, and `20s`
- Same model, dataset, optimizer family, and constant schedule
- Two-seed main benchmark: `42`, `137`

## Archived / Not Active
- `15s`, `45s`, `80s` duration sweep is not the active research track
- Schedule comparisons are not part of the current question
- Architecture search is deferred until LR transfer is clearly established

## Completed Batch: `batch_27_lr_transfer`
- Queue: [`queue_lr_transfer.json`](/root/llm-research-kit/optimization/queue_lr_transfer.json)
- Purpose: clean rerun of the exact LR transfer question
- Candidates: `0.006`, `0.008`, `0.012`
- Durations: `5s`, `10s`, `20s`
- Seed: `42`

### Results
- 5s:
  - `0.008` -> `6.7637`
  - `0.012` -> `6.7772`
  - `0.006` -> `6.8935`
- 10s:
  - `0.006` -> `6.5143`
  - `0.008` -> `6.5250`
  - `0.012` -> `6.5508`
- 20s:
  - `0.006` -> `6.2623`
  - `0.008` -> `6.2695`
  - `0.012` -> `6.2737`

### Read
- `5s` does not exactly predict the `20s` winner
- `10s` does predict the `20s` winner
- `20s` ranking is close enough between `0.006` and `0.008` that one tie-break multi-seed comparison is optional if the result will be published

## Next Required Batch
## Completed Batch: `batch_28_lr_prediction_protocol`
- Queue: [`queue_lr_prediction_protocol.json`](/root/llm-research-kit/optimization/queue_lr_prediction_protocol.json)
- Grid: `0.005, 0.006, 0.007, 0.008, 0.010, 0.012`
- Durations: `5s`, `10s`, `20s`
- Seeds: `42`, `137`

### Mean ranking
- 5s: `0.007 < 0.008 < 0.010 < 0.006 < 0.012 < 0.005`
- 10s: `0.007 < 0.005 < 0.006 < 0.008 < 0.010 < 0.012`
- 20s: `0.006 < 0.007 < 0.005 < 0.012 < 0.008 < 0.010`

### Read
- `5s` misses the exact `20s` winner
- `10s` also misses the exact `20s` winner in the two-seed mean
- `10s` is still substantially better aligned with `20s` than `5s`
- Regret is tiny for both `5s` and `10s`, so the practical protocol may still be strong even when exact winner match fails

## Next Required Batch
- Narrow tie-resolution around `0.006` and `0.007` at `20s`
- Optionally check whether top-3 from `5s` or `10s` is the right practical elimination rule
