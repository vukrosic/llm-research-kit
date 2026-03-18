# Insights

## Current Priors

- `OneBConfig` prior:
  - `muon_lr = 0.008`
  - `adamw_lr = 0.0015`
  - `warmup_ratio = 0.01`
  - `schedule_type = cosine`
- Current AdamW-to-Muon LR ratio prior: `0.1875`
- For long pretraining, prefer the highest LR that remains stable after warmup and still looks good at a non-trivial proxy budget.

## Decision Standard

- A short run is useful only if it preserves candidate ranking into a longer run.
- Rank candidates by validation-loss trajectory, not only by the final short-run point.
- Promote only stable candidates to longer runs.

## What To Write Here After Each Sweep

Record:

- sweep date
- exact config
- token budget
- winning candidates
- failures or instability patterns
- whether the ranking transferred to the longer run
- next experiment to run

## First Expected Outcome

For a 1B GPT-style model trained for a long time, I expect the eventual production LR to be slightly lower than the most aggressive LR that looks best in the first few million tokens.
