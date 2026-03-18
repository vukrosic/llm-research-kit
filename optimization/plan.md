# Optimization Plan

## Setup
- GPU: L40S 46GB | Mode: eager (no `torch.compile`) | Throughput: ~51K tps
- Model: 88M MinimalLLM | Muon+AdamW | BF16
- Seeds: `42`, `137`
- 5s: ~250K tokens (~15 steps)
- 10s: ~500K tokens (~31 steps)
- 20s: ~1M tokens (~62 steps)

## Active Research Question
Can `5s` and `10s` LR sweeps predict the best `20s` LR well enough to publish the result as:

`Can 5-second LLM runs predict 20-second winners?`

## Core Questions To Answer
1. Is `5s` good for elimination but not final selection?
2. Is `10s` the minimum reliable selection tier?
3. Does the best LR move smoothly or abruptly from `5s` to `20s`?
4. Is top-2 prediction easier than exact winner prediction?
5. What is the compute-optimal protocol for choosing a `20s` LR?

## Protocol
- Fixed LR grid at all three durations:
  - `0.005`
  - `0.006`
  - `0.007`
  - `0.008`
  - `0.010`
  - `0.012`
- Same batch size, same schedule, same optimizer family
- `adamw_lr = muon_lr / 4`
- Two seeds per point: `42`, `137`
- Total main batch size: `6 LRs x 3 durations x 2 seeds = 36 runs`

## Evaluation Metrics
- Exact winner match at `20s`
- Top-2 containment of the `20s` winner
- Regret at `20s`
- Rank correlation between `5s` and `20s`
- Rank correlation between `10s` and `20s`

## Prediction Rules To Compare
1. `5s` winner -> predict `20s`
2. `10s` winner -> predict `20s`
3. Top-2 from `5s`, then choose by `10s`
4. Top-3 from `5s`, then choose by `10s`

## Current Read
- The two-seed fixed-grid benchmark shows that exact winner prediction is fragile at both `5s` and `10s`
- The more robust signal is low regret: short runs get very close to the `20s` winner even when they miss the exact best LR
- The next step should focus on practical protocol quality, not overclaiming exact-match selection

## Current Queue
- Main benchmark queue: [`queue_lr_prediction_protocol.json`](/root/llm-research-kit/optimization/queue_lr_prediction_protocol.json)

## Banlist
- `muon_lr >= 0.048` for the current study

## Out Of Scope
- Schedules
- Weight decay search
- Momentum search
- Batch size search
- Architecture changes
- 40s+ validation for now
