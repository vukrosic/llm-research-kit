# AI Research Roadmap

## Goal

- Train a stable ~1B dense GPT-style model with `OneBConfig`.
- Use it as the first step toward a GPT-3-class training stack.
- Optimize loss per unit compute, not loss per step.
- Keep experiments clean enough to attribute gains correctly.

## Core Principle

Separate optimization research from architecture research until the baseline is strong. If both change together, attribution is weak.

## Overall Program

Run two tracks:

1. Optimization track
2. Architecture track

The optimization track comes first. The architecture track starts only after the baseline recipe is reasonably stable.

## Optimization Track

### Objective

Find the training recipe with the best long-run validation loss at acceptable stability and throughput.

### Research Loop

For each knob:

1. Ask one narrow question.
2. Use the cheapest proxy that preserves long-run ranking.
3. Keep all other settings fixed.
4. Promote only the top 1-2 candidates to a longer confirmation run.
5. Record the decision and move to the next knob.

### Current Baseline Priors

From `OneBConfig`:

- `muon_lr = 0.008`
- `adamw_lr = 0.0015`
- `muon_momentum = 0.95`
- `warmup_ratio = 0.01`
- `schedule_type = cosine`
- `weight_decay = 0.1`
- `dropout = 0.0`
- `grad_clip = 1.0`

Current optimizer split:

- Muon for most 2D weight matrices
- AdamW for embeddings, norms, and remaining parameters

This split is defined in `training/trainer.py`.

## Optimization Priority Order

Tune in this order:

1. learning rate
2. warmup ratio
3. AdamW-to-Muon LR ratio
4. weight decay
5. Muon momentum
6. effective batch size scaling
7. schedule type
8. grad clip
9. dropout

The rule is simple:

- optimization dynamics first
- regularization second
- low-leverage cleanup last

## Learning Rate Research

### Question

Which `muon_lr` and `adamw_lr` pair gives the best long-run loss reduction per unit time without causing instability?

### Why It Comes First

- LR is the highest-leverage hyperparameter.
- A bad LR can waste a full long run.
- Most other settings interact with LR.

### Main Risk

The best LR in the first few hundred steps is often too high for long training. Short sweeps should preserve ranking, not just maximize early speed.

### Proxy Design

Use a two-stage process:

1. screening run: `200_000_000` to `500_000_000` tokens
2. longer transfer check: `2_000_000_000` to `5_000_000_000` tokens

Do not use tiny sweeps like `1M` or `20M` tokens if the goal is long-run transfer on a 1B model. They mostly measure very early acceleration.

### What Must Stay Fixed

- model shape
- global batch size
- sequence length
- dataset mixture
- precision
- optimizer assignment
- warmup ratio
- scheduler shape

### Initial Sweep

Keep the current AdamW-to-Muon ratio near the present prior:

- `0.0015 / 0.008 = 0.1875`

Sweep:

- `muon_lr = 0.004`, `0.006`, `0.008`, `0.010`, `0.012`
- `adamw_lr = 0.1875 * muon_lr`

So the first five candidates are:

- `(0.004, 0.00075)`
- `(0.006, 0.001125)`
- `(0.008, 0.0015)`
- `(0.010, 0.001875)`
- `(0.012, 0.00225)`

Then:

1. take the best 1-2 candidates
2. refine locally around the winner
3. confirm at `200M` tokens

### Interaction Risk

`muon_lr` and `adamw_lr / muon_lr` likely interact. Use one of these:

1. small 2D grid
2. sequential sweeps with a re-check of the LR winner after the ratio sweep

Preferred first pass:

- `muon_lr = 0.006`, `0.008`, `0.010`
- ratio = `0.10`, `0.1875`, `0.25`

This 3x3 grid costs about the same as separate coarse LR and ratio sweeps and covers the joint space better.

### How To Rank Candidates

Primary metric:

- validation loss trajectory over the run

Practical ranking:

- lowest area under the validation-loss curve after warmup
- then lowest final validation loss
- then fastest wall-clock if tied

Reject candidates that show:

- repeated loss spikes
- NaNs
- clear post-warmup instability
- heavy clipping almost every step

### Schedule Floor

For long cosine runs, define the minimum LR explicitly. The default prior is:

- `min_lr = 0.1 * peak_lr`

Treat this as part of the training recipe, not an implicit detail.

## Warmup Research

### Question

What `warmup_ratio` gives the best stable long-run training for the chosen LR?

### Sweep

After rough LR is chosen, test:

- `0.002`
- `0.005`
- `0.01`
- `0.02`

### Why It Matters

Warmup often determines whether an LR is merely fast early or usable for long training.

## AdamW-to-Muon LR Ratio Research

### Question

What ratio between `adamw_lr` and `muon_lr` works best for this optimizer split?

### Sweep

Hold `muon_lr` fixed near the chosen value and test ratio values such as:

- `0.10`
- `0.15`
- `0.20`
- `0.25`

### Why It Matters

The two optimizers update different parameter groups. This ratio still matters after the main LR is set.

## Weight Decay Research

### Question

What `weight_decay` gives the best medium- and long-run validation loss?

### Sweep

Test:

- `0.05`
- `0.1`
- `0.15`
- `0.2`

### Note

Weight decay often matters more later in training, so confirm it on a longer run.

## Muon Momentum Research

### Question

What `muon_momentum` is best once LR is reasonably tuned?

### Sweep

Test:

- `0.90`
- `0.95`
- `0.98`

### Note

Lower priority than LR and warmup, but still worth checking once the baseline is stable.

## Batch Size Scaling Research

### Question

How does the best training recipe change when effective batch size changes?

### Rule

If effective batch size changes, re-check:

- LR
- warmup
- possibly weight decay

Batch changes usually require retuning.

## Schedule Type Research

### Recommendation

Keep `cosine` as the default long-run prior unless evidence suggests otherwise.

`constant` may be fine for short probes, but its ranking may not transfer to a long cosine run.

### Research Tool

For screening and transfer experiments, also consider a warmup-stable-decay schedule. A stable middle phase makes different token budgets easier to compare because the schedule horizon changes less aggressively than with cosine.

Use WSD as a research tool if ranking transfer under cosine looks noisy. Keep cosine as the production default unless WSD clearly wins in long runs.

## Grad Clip Research

### Recommendation

Leave `grad_clip = 1.0` unless clipping happens very often.

If clipping is frequent, first suspect:

- LR too high
- warmup too short

Do not tune clip first.

## Dropout Research

### Recommendation

Keep `dropout = 0.0` as the initial prior for dense GPT pretraining at this scale.

Only tune dropout if validation clearly decouples from training loss.

## Optimization Decision Standard

A short run is useful only if it preserves candidate ranking into a longer run.

General rule:

- if top candidates keep swapping rank between nearby milestones, the proxy is too short

The production setting should usually be:

- slightly below the highest aggressive setting that still looks good in the short proxy

not:

- simply the most aggressive setting that wins very early

## Data Strategy

Keep dataset mixture fixed during optimization and architecture comparisons.

Data work is a separate track. At minimum, document:

- dataset sources
- mixture weights
- filtering rules
- dedup policy
- tokenizer version

Do not tune data mixture in the middle of optimizer or architecture sweeps.

## Architecture Track

### Objective

Research architectural changes that improve validation loss at matched compute and acceptable system cost.

### Main Principle

Architecture should be evaluated at fixed compute, not just fixed steps or fixed parameter count.

The real question is whether it improves loss per dollar or per FLOP.

### Requirements For A Credible Win

A new architecture should clear all three:

1. better validation loss at matched training compute
2. acceptable throughput and memory cost
3. not dramatically harder to tune

If it wins only after special handling or significantly worse throughput, it probably does not get you to GPT-3-class capability faster in practice.

## Architecture Research Loop

For each idea:

1. define one minimal mechanism
2. verify it trains stably
3. compare against the baseline at matched compute
4. only then expand the design space

Start with the smallest version that could plausibly work.

## Architecture Evaluation Ladder

Use a three-stage ladder:

1. small-model sanity check
2. medium proxy experiment
3. 1B confirmation run

### Stage 1: small-model sanity check

Question:

- does the idea train at all?

Check for:

- NaNs
- activation blowups
- memory regressions
- throughput collapse
- implementation bugs

### Stage 2: medium proxy

Question: does it beat the baseline at matched compute by enough margin to justify promotion?

Promotion threshold:

- at least ~0.5% validation-loss improvement at matched FLOPs
- sustained through the last quarter of the run
- without unacceptable throughput or memory regression

### Stage 3: 1B confirmation

Question: does the gain survive at the target scale?

Only promote the top 1-2 ideas.

## Cross-Layer Attention Research

### Core Question

Does connecting attention across layers improve loss-vs-compute at fixed training recipe?

### Important Constraint

"Connecting attention across layers" is too vague to test directly. Pick one minimal variant.

Possible variants:

- attend to previous-layer KV states
- gated skip from earlier attention outputs
- cross-layer residual mixing
- shared recurrent memory across layers

### Recommended First Variant

Start with one simple gated cross-layer path:

- small parameter increase
- limited to every N layers, not every layer
- easy to ablate
- easy to disable

This is better than dense all-to-all cross-layer attention, which is costlier and harder to interpret.

### First Questions To Answer

- does it train stably?
- does it improve validation loss at matched compute?
- how much throughput does it cost?
- does it require retuning to look good?

### What Must Be Matched Against Baseline

- train tokens
- optimizer recipe
- batch size
- sequence length
- dataset mix
- evaluation set
- parameter count or training FLOPs

If the variant gets extra tuning budget or extra compute without accounting for it, the comparison is not clean.

## Evaluation Beyond Validation Loss

Validation loss is the primary screening metric.

For 1B confirmation runs of architecture variants, also check:

- downstream or few-shot evaluation
- long-context behavior if relevant
- throughput and memory at deployment-relevant settings

Two models can match on perplexity and still differ in downstream behavior.

## Bridge Beyond 1B

The 1B baseline is the first target, not the end state.

After a stable 1B recipe exists, add scale-transfer checks:

- train a smaller version such as `125M`
- train an intermediate version such as `350M`
- compare how optimal LR scales with width/depth
- check whether the Muon/AdamW split still makes sense at larger matrix sizes
- check whether architecture gains grow, shrink, or disappear with scale

The goal is to avoid overfitting the recipe to exactly one model size.

## Cross-Layer Attention Research Questions

- ARCH-001: At fixed compute, does cross-layer attention beat the baseline validation-loss curve?
- ARCH-002: Is the gain still present after fair LR and warmup tuning for both baseline and variant?
- ARCH-003: What is the minimal cross-layer mechanism that preserves most of the gain?
- ARCH-004: Does the gain come from easier optimization early or stronger representation later?
- ARCH-005: Does it help only short-context pretraining loss, or also long-context behavior?

## Research Sequence

Recommended sequence:

1. stabilize the 1B baseline
2. tune LR
3. tune warmup
4. tune LR ratio or run a small LR/ratio joint sweep
5. tune weight decay
6. confirm the baseline over a non-trivial run
7. run small scale-transfer checks such as `125M` and `350M`
8. introduce one minimal architecture change
9. test it on a smaller proxy first
10. confirm only top architecture ideas on the 1B model

## Practical Rules

- Never change multiple major knobs at once unless the experiment is explicitly about interaction effects.
- Use short runs for screening and longer runs for confirmation.
- Favor loss at matched compute over loss at matched steps.
- Favor ideas that preserve throughput and stability.
- Record failures, not just wins.

## Immediate Next Steps

1. Use `OneBConfig` as the baseline recipe.
2. Run either a 3x3 LR/ratio grid or a longer coarse LR sweep at `200M` to `500M` tokens.
3. Refine around the winner.
4. Confirm the top 1-2 candidates at `2B` to `5B` tokens.
5. Tune warmup and weight decay next.
6. Add small scale-transfer checks.
7. Only after that, implement one minimal cross-layer attention variant and test it on a smaller proxy.
