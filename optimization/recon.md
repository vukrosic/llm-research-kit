# Recon

## Research Loop

For each optimization variable:

1. Ask a question narrow enough to falsify in 1-3 days.
2. Use the cheapest proxy that preserves the long-run ranking of candidates.
3. Keep everything else fixed.
4. Promote only the top 1-2 candidates to a longer confirmation run.
5. Record the result and the next question in `optimization/insights.md`.

## LR-001: Long-Run Transferable Learning Rate Sweep

### Question

Which `muon_lr` and `adamw_lr` pair gives the best loss reduction per unit time for long training, without creating instability later in training?

### Why this question first

- LR is the highest-leverage training hyperparameter.
- A bad LR can waste a full long run.
- Your code already exposes the right knobs:
  - `muon_lr`
  - `adamw_lr`
  - `warmup_ratio`
  - `schedule_type`

### Main risk

The best LR for the first few hundred steps is often too high for a long run. So the short sweep must preserve ranking, not just maximize immediate speed.

## Working Hypothesis

- The best long-run LR is usually slightly below the highest stable short-run LR.
- Short runs transfer better when they preserve:
  - optimizer split,
  - global batch size,
  - sequence length,
  - data mixture,
  - schedule shape,
  - warmup fraction.
- Ranking candidates by area under the validation-loss curve over an early-but-not-tiny window transfers better than ranking by the very first loss drop.

## Sweep Design

### Fixed settings

Keep these identical between proxy runs and the eventual long run:

- model shape (`OneBConfig`)
- `batch_size`
- `gradient_accumulation_steps`
- `max_seq_len`
- tokenizer and dataset mix
- precision
- optimizer assignment rule in `training/trainer.py`

Do not change several knobs at once. For LR sweeps, keep:

- `muon_momentum`
- `weight_decay`
- `warmup_ratio`
- `schedule_type`

fixed.

### Proxy budget

Use a two-stage sweep:

1. Short screening run: 10M to 30M tokens
2. Transfer check: 100M to 300M tokens

For this repo, I would start with:

- Stage 1: `20_000_000` tokens
- Stage 2: `200_000_000` tokens

That is long enough for divergence and bad late-warmup behavior to show up, but still much cheaper than a multi-billion-token run.

## Candidate Parameterization

Do not sweep `muon_lr` and `adamw_lr` independently over a huge grid first. That wastes runs.

Start with a ratio-based sweep:

- Hold `adamw_lr / muon_lr` near the current prior.
- Current prior from `OneBConfig`: `0.0015 / 0.008 = 0.1875`

### Stage 1 grid

Sweep `muon_lr` on a multiplicative grid around the prior and derive `adamw_lr` from the ratio:

- `muon_lr`: `0.004`, `0.006`, `0.008`, `0.010`, `0.012`
- `adamw_lr = 0.1875 * muon_lr`

This gives:

- `(0.004, 0.00075)`
- `(0.006, 0.001125)`
- `(0.008, 0.0015)`
- `(0.010, 0.001875)`
- `(0.012, 0.00225)`

If all runs are stable, run a second local sweep around the winner with smaller spacing.

If the top LR diverges or shows instability, drop the upper range and refine below it.

## What to Measure

### Primary metric

- Validation loss integrated over training, not just final point.

Practical approximation:

- compute mean validation loss across milestones after warmup,
- or compute trapezoidal area under the val-loss curve.

This is better than using only the last point from a short run.

### Secondary metrics

- final validation loss at the proxy budget
- tokens/sec or wall-clock to target loss
- gradient norm clipping frequency
- NaN / overflow / divergence
- training loss spikes after warmup

### Reject immediately if

- loss spikes repeatedly after optimizer steps,
- validation loss is materially worse after initial warmup,
- gradients clip almost every step,
- run produces NaNs or unstable oscillation.

## Decision Rule

Promote the top 2 LR candidates from the short sweep using this order:

1. stable runs only
2. lowest area-under-val-loss curve
3. lowest final val loss
4. fastest wall-clock if metrics are effectively tied

Then run only those 2 candidates for the longer transfer check.

## What Makes a Short Sweep Transfer

The short run should answer:

- does this LR survive warmup cleanly?
- is it still good after the initial fast-loss regime?
- does it remain stable once optimizer state starts to matter?

That means the proxy should not be extremely tiny. A 1M-token sweep is usually too short for this purpose on a 1B model.

My practical rule:

- use the shortest run where the candidate ranking stops changing much between adjacent milestones

If the top 2 swap repeatedly between 5M, 10M, and 20M tokens, the proxy is too short.

## Suggested First Pass For This Repo

### Phase A: coarse sweep

Run 5 candidates at `20M` tokens with:

- `schedule_type=cosine`
- `warmup_ratio=0.01`
- current batch and accumulation

### Phase B: refine around winner

If the winner is, for example, `0.006`, run:

- `0.005`
- `0.006`
- `0.007`

and preserve the AdamW ratio.

### Phase C: transfer check

Take the best 2 candidates to `200M` tokens.

If the same winner survives, lock LR and move to the next question:

- warmup ratio
- weight decay
- Muon momentum

## Research Questions After LR

- LR-002: does the best `adamw_lr / muon_lr` ratio differ from the current prior?
- LR-003: does the best LR depend on warmup ratio enough that LR and warmup must be tuned jointly?
- LR-004: does the best LR scale cleanly with larger effective batch size?

## Concrete Recommendation

If the goal is minimum time with useful transfer to long training, I would not do a huge hyperparameter search.

I would do:

1. 5-run coarse multiplicative sweep at `20M` tokens
2. 3-run local refinement at `20M` tokens
3. 2-run confirmation at `200M` tokens

That is enough to make an evidence-based LR choice before committing to a very long run.
