# Scaling Research Rules

## Core Principles

1. **Use a fixed small seed set for the main benchmark.** Current default: `42` and `137`.
2. **Val loss is the only metric.** Lower val loss wins.
3. **Eager mode only.** `torch.compile` adds overhead and changes short-run behavior.
4. **Time-based stopping.** Every run is fixed to `5s`, `10s`, or `20s`.
5. **Active scope is LR scaling only.** Do not branch into schedules, weight decay, momentum, batch size, or architecture search in this phase.

## Tier Structure

| Tier | Duration | Purpose | Max experiments | Promote top N |
|------|----------|---------|-----------------|---------------|
| T1 | 5s | Broad LR screening | 12 | Top 2-3 |
| T2 | 10s | Intermediate transfer check | 12 | Top 2-3 |
| T3 | 20s | Target tier to predict | 12 | Winner |

## T1: 5s Screening

**Goal:** Find the short-run LR ordering.

**What to vary:**
- `muon_lr` only
- Keep `adamw_lr = muon_lr / 4`
- Keep all other settings fixed

**Rules:**
- Use the same LR grid as every other tier
- Compare ranks, top-2 containment, and regret rather than only the winner

## T2: 10s Transfer Check

**Goal:** Check whether the `5s` ranking persists and whether it helps predict `20s`.

**What to run:**
- The same fixed LR grid as `5s`
- The same seed set as `5s`

**Rules:**
- Compare `5s -> 10s` rank changes
- Test whether `10s` is the minimum reliable selection tier

## T3: 20s Target Tier

**Goal:** Identify the actual winner and judge whether short runs predicted it.

**What to run:**
- The same fixed LR grid as `5s` and `10s`
- The same seed set as `5s` and `10s`

**Rules:**
- Build the `5s/10s/20s` transfer table
- Evaluate exact winner match, top-2 containment, and regret
- Decide whether `5s` alone was enough, or whether `10s` materially improved the prediction

## Tie-Break Seeds

Only run extra seeds beyond the main two-seed benchmark if:
- the top candidates differ by less than about `0.01` val_loss at the same duration
- or the result is intended as a publishable claim and the winner is too close to call

## Leaderboard Rules

1. **One leaderboard per tier.** Never compare losses across durations directly.
2. **Deduplication:** If the same config is run twice at the same tier and seed, keep only the best result.
3. **Leaderboard size:** Show the relevant contenders, not a padded list.
4. **Required columns:** Rank, exp_id, val_loss, config changes.
5. **Baseline:** The default config must appear in every tier.

## Scaling Table Rules

The transfer table is the most important output.

| Config | 5s | 10s | 20s | Rank Trend | Prediction Outcome |
|--------|----|-----|-----|------------|--------------------|
| ... | val_loss | val_loss | val_loss | T1->T2->T3 | predicted winner or not |

**Key metrics:**
- **Rank trend:** Track the ordering across tiers.
- **Prediction outcome:** Did the short-run ranking identify the true `20s` winner?

## What Not To Do

1. **Do not broaden scope.** No new hyperparameter categories until LR transfer is answered.
2. **Do not skip tiers.** Going straight from `5s` to `20s` weakens the transfer story.
3. **Do not change multiple HPs at once.** Only LR moves in this phase.
4. **Do not overclaim.** A close single-seed result is a hint, not a theorem.

## Metrics To Report

- Exact winner match at `20s`
- Top-2 containment of the `20s` winner
- Regret at `20s`
- Rank correlation: `5s` vs `20s`
- Rank correlation: `10s` vs `20s`

## When To Stop

- Stop the benchmark when the fixed-grid, two-seed sweep is complete
- Stop this phase when you can answer the five protocol questions with a clean transfer table
