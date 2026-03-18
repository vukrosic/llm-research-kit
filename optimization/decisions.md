# Decisions

One entry per completed sweep. Reasoning first, then winner, then next step.

---

## 2026-03-18 - LR-SWEEP-EXISTING-EVIDENCE

- `date`: 2026-03-18
- `sweep_id`: `LR-SWEEP-EXISTING-EVIDENCE`
- `question`: Which LR candidates stay competitive as training duration increases from 5s to 45s?
- `winner`: No locked winner yet. Current long-enough leader is `0.016 / 0.004` at 45s, but confidence is too low to lock.
- `evidence`:
  - 5s best is around `0.007-0.008`
  - 10s best is around `0.007`
  - 20s best is around `0.006-0.007`
  - 45s validation gives `0.016 -> 5.6321`, `0.012 -> 5.6381`, `0.010 -> 5.6502`
  - ranking has shifted several times, so the current proxy is still too short
- `rejected_candidates`:
  - `0.010 / 0.0025`: behind both `0.012` and `0.016` at 45s
  - any claim that LR is already locked from 5s, 10s, or 20s data
- `confidence`: low
- `next_step`: Run the 8M-token LR sweep at 88M with seeds `42` and `137`, then rank by mean validation loss after warmup

## 2026-03-18 - EXP-0-1B-FEASIBILITY

- `date`: 2026-03-18
- `sweep_id`: `EXP-0-1B-FEASIBILITY`
- `question`: Can `OneBConfig` train on one L40S 48GB without OOM, and what throughput does it achieve?
- `winner`: Feasible. The 1B model completed a 500k-token run without OOM.
- `evidence`:
  - `501,760` tokens seen
  - `245` steps
  - `93.3s` active training time
  - `160.6s` wall time
  - about `5.38k` tokens/sec during active training
- `rejected_candidates`:
  - the assumption that 1B fit is still unknown
  - the assumption that throughput must be estimated before the next 1B confirmation run
- `confidence`: medium
- `next_step`: Capture peak VRAM in the next 1B run and use the result as the baseline confirmation setup for Phase 3 LR transfer
