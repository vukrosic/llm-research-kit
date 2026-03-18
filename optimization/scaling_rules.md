# Scaling Research Rules

## Core Principles

1. **Single seed (42) for all experiments.** Same seed = same data order = fair comparison. Any val_loss difference is real signal, not noise.
2. **Val loss is the only metric.** Lower val_loss = better. No subjective judgments.
3. **Eager mode only** (no torch.compile). Compilation overhead (~46s) ruins short experiments and changes throughput characteristics.
4. **Time-based stopping.** Each tier has a fixed wall-clock budget. The model trains for exactly that many seconds.

## Tier Structure

| Tier | Duration | Purpose | Max experiments | Promote top N |
|------|----------|---------|-----------------|---------------|
| T1   | 5s       | Wide exploration, eliminate bad configs | 20-30 | Top 8 |
| T2   | 10s      | Narrow the field, first scaling signal | 8-12 | Top 5 |
| T3   | 20s      | Confirm scaling trends | 5-8 | Top 3 |
| T4   | 30s+     | Final validation | 3-5 | Winner |

## T1: Wide Exploration (5s)

**Goal:** Explore the hyperparameter space broadly. Eliminate clearly bad configs.

**What to vary:**
- batch_size: [1, 2, 3, 4, 8]
- muon_lr: [0.006, 0.008, 0.010, 0.012, 0.016, 0.024]
- weight_decay: [0.0, 0.05, 0.1, 0.2]
- grad_clip: [0.5, 1.0, 2.0]
- Any architectural or schedule changes

**Rules:**
- Run 20-30 experiments in systematic batches of 5
- Always include the current best config as a control/baseline
- Log every result, even failures and OOMs
- After all T1 experiments, rank by val_loss and take top 8

## T2: Scaling Signal (10s)

**Goal:** See which T1 winners actually scale. First real ranking.

**What to run:**
- All top 8 from T1 (exact same configs, just 10s instead of 5s)
- Plus 2-4 "diversity picks" — configs that were mediocre at T1 but explore a different region of HP space (e.g., very small batch size, unusual LR)

**Rules:**
- 8-12 experiments total
- Compare T1→T2 rank changes to identify scaling trends
- Configs that DROP 3+ ranks from T1→T2 are "non-scalers" — eliminate them
- Configs that RISE 2+ ranks are "scalers" — these are the real candidates
- Promote top 5 to T3

## T3: Confirmation (20s)

**Goal:** Confirm scaling trends hold. Identify the real winner.

**What to run:**
- All top 5 from T2
- Plus 1-2 new configs inspired by scaling patterns (e.g., if smaller batch keeps winning, try even smaller)
- Plus 1 "extrapolation" config: take the T1→T2 scaling trend and extrapolate the optimal HP for 20s

**Rules:**
- 5-8 experiments total
- Build the full scaling table: each config's val_loss at 5s, 10s, 20s
- Calculate scaling rate: (val_loss_5s - val_loss_20s) / 15 seconds = loss improvement per second
- The config with the best scaling rate AND good absolute loss wins
- Promote top 3 to T4

## T4: Final Validation (30s+)

**Goal:** Confirm the winner at longer training. This is what actually matters.

**Rules:**
- Run top 3 from T3 at 30s
- The winner at 30s is the recommended production config
- Optional: run winner at 60s, 120s for extrapolation confidence
- Optional: run winner with 3 different seeds (42, 137, 256) to confirm it's not seed-dependent

## Leaderboard Rules

1. **One leaderboard per tier.** Never compare across tiers (5s loss vs 10s loss is meaningless).
2. **Deduplication:** If the same config is run twice at the same tier, keep only the best result.
3. **Leaderboard size:** Show top 8 per tier.
4. **Required columns:** Rank, exp_id, val_loss, steps, tokens/sec, delta vs best, config changes.
5. **Baseline:** The default config (no changes) must appear in every tier as the reference point.

## Scaling Table Rules

The scaling table is the most important output. It shows how each config performs across tiers.

| Config | 5s | 10s | 20s | 30s | Scaling Rate | Rank Trend |
|--------|----|----|-----|-----|-------------|------------|
| ... | val_loss | val_loss | val_loss | val_loss | Δloss/Δtime | T1→T2→T3→T4 |

**Key metrics:**
- **Scaling rate:** `(loss_T1 - loss_T3) / (20 - 5)` = loss improvement per second. Higher magnitude = better scaler.
- **Rank trend:** Track rank at each tier. Rising = scaler. Falling = non-scaler.
- **Consistency:** A config that's top 3 at ALL tiers is more trustworthy than one that's #1 at one tier but #5 at another.

## What NOT to Do

1. **Don't over-invest in T1.** 5s experiments are cheap but misleading. Use them for elimination, not selection.
2. **Don't skip tiers.** Going straight from 5s to 20s misses the scaling inflection points.
3. **Don't change multiple HPs at once** in the same batch. Change one variable at a time in systematic sweeps so you can attribute the effect.
4. **Don't ignore "boring" configs.** The default config or a simple change often beats complex multi-HP combinations.
5. **Don't chase noise.** If two configs differ by <0.01 in val_loss, they're effectively tied. Look at scaling rate to break ties.

## Experiment Naming Convention

Format: `{duration}s_{batch_size_info}_{lr_info}_{other_changes}`

Examples:
- `5s_bs4_lr0.010` — 5s, batch_size=4, muon_lr=0.010
- `10s_bs3_lr0.010_wd0.1` — 10s, batch_size=3, muon_lr=0.010, weight_decay=0.1
- `20s_default` — 20s with all default hyperparameters

## When to Stop

- **Stop exploring a HP** when 3+ experiments show it has <0.01 effect on val_loss at T2+.
- **Stop a tier** when you've run all planned experiments and the top 3 are clear.
- **Stop the whole research** when the T4 winner is identified and confirmed with multi-seed.

## Recording Results

Every experiment must produce a JSON file in `results/{batch_name}/` with:
- `exp_id`, `changes` (dict of HP diffs from default), `val_loss`, `train_loss`
- `steps`, `tokens_per_second`, `training_time`, `train_seconds`, `seed`
- `status` ("done", "failed", "oom")

The queue file (`optimization/queue.json`) tracks planned, running, and completed experiments.
The dashboard reads from `results/` and `optimization/` to show live status.
