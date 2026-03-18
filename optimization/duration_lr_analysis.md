# Archived Duration -> LR Scaling Analysis

This note is kept as an archive of an earlier broader sweep. It is not the active basis for current conclusions because it uses `5s`, `15s`, `45s`, and `80s` rather than the active `5s`, `10s`, `20s` research scope.

## Status

Archived. Do not treat this as the current project conclusion.

## What It Contained

- A budgeted duration sweep across `5s`, `15s`, `45s`, `80s`
- Quadratic fits in `log(muon_lr)` per duration
- A weak duration-dependent power-law fit for the inferred optimum LR

## Why Archived

- The duration grid does not match the active tiers
- The power-law fit used only three valid duration-level fits
- The result conflicts with the current `5s`, `10s`, `20s` leaderboard signal
- The current project is intentionally narrower: determine whether short runs predict the `20s` LR winner

## Use

Use this only as background context for a prior exploratory sweep, not as a live scaling rule.

## Archived Artifacts

- Raw grouped statistics: [`duration_lr_grouped.csv`](/root/llm-research-kit/optimization/duration_lr_analysis/duration_lr_grouped.csv)
- Per-duration fit table: [`duration_lr_fits.csv`](/root/llm-research-kit/optimization/duration_lr_analysis/duration_lr_fits.csv)
- Summary text: [`summary.txt`](/root/llm-research-kit/optimization/duration_lr_analysis/summary.txt)
- Plots:
  - [`parabolas.png`](/root/llm-research-kit/optimization/duration_lr_analysis/parabolas.png)
  - [`power_law.png`](/root/llm-research-kit/optimization/duration_lr_analysis/power_law.png)
  - [`residuals.png`](/root/llm-research-kit/optimization/duration_lr_analysis/residuals.png)
  - [`heatmap.png`](/root/llm-research-kit/optimization/duration_lr_analysis/heatmap.png)
