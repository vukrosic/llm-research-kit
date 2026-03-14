# CLAUDE.md — AI Research Agent Rules

This file governs how Claude Code operates in this repository. Read it completely before doing anything experiment-related.

---

## 1. What This Project Is

Small-scale LLM architecture research. We train ~88M-parameter transformer models for short token budgets (6M tokens) to rapidly evaluate architectural decisions. The goal is to find combinations of architecture choices, optimizer settings, and training tricks that consistently improve validation loss.

---

## 2. Folder Structure

```
CLAUDE.md                    ← you are here (AI rules)
research/
  inbox/                     ← user drops papers/ideas as .md files here
  processed/                 ← AI moves files here after extracting hypotheses
  hypotheses.md              ← AI-maintained distilled idea list
experiments/
  history.json               ← ALL experiments ever run (source of truth)
  queue.json                 ← pending experiments (AI reads/writes this)
  leaderboard.md             ← top performers (AI keeps updated)
ablation_results/
  6000000tok/                ← raw 6M token results
configs/
  ablation_configs.py        ← where experiment configs live
run_ablations.py             ← training runner
```

---

## 3. Before Designing Any Experiment

**Always do these steps in order:**

1. **Read `experiments/history.json`** — search for similar configs. If an experiment with identical or nearly-identical flags already exists, do NOT re-run it. Instead, look for a neighboring variation that hasn't been tried.

2. **Read `research/inbox/`** — process any unread papers/ideas (see Section 6).

3. **Read `experiments/queue.json`** — check what's already queued. Do not duplicate queued experiments.

---

## 4. Exploration vs. Exploitation

**Default ratio: 70% exploitation, 30% exploration.**

### Exploitation (70%)
Refine around known winners. If an experiment scored > +0.5% improvement:
- Try ±1 step variations of its key hyperparameter
- Try combining it with the second-best winner
- Try combining two winning experiments to see if they stack

### Exploration (30%)
Completely new mechanisms not yet tried. Source from:
- `research/hypotheses.md` (ideas from papers)
- Novel combinations of two neutral experiments (neutral + neutral can sometimes = winner)
- Mechanisms that failed on old baseline but haven't been tried on new g2 baseline
- Completely new ideas and features from scratch

### Override conditions (shift to more exploration):
- If the last 15 exploitation experiments all returned "neutral" or worse → shift to 50/50
- If a new paper is in `research/inbox/` → process it and add at least 2 exploration experiments from it
- If the best val_loss hasn't improved in 30 experiments → shift to 60% exploration

### Override conditions (shift to more exploitation):
- If the last 3 experiments all returned "winner" → push to 90% exploitation to maximize the vein

---

## 5. Experiment Design Rules

### Naming convention
```
{category}_{description}
e.g.: attn_rope_scaled, ffn_bilinear_wide, opt_muon_lr_0.018, norm_rms_gate
```

### Queue entry format (`experiments/queue.json`)
```json
{
  "exp_id": "attn_rope_scaled",
  "hypothesis": "Scaled RoPE base improves long-range generalization",
  "source": "exploitation",
  "parent_exp": "attn_qk_layernorm",
  "expected_delta": "+0.3% to +0.8%",
  "priority": 2,
  "token_budget": 6000000,
  "flags_override": {"rope_base": 500000},
  "status": "pending",
  "added_by": "claude",
  "tags": ["attention", "positional-encoding"]
}
```

`priority`: 1 = run next, 2 = high, 3 = normal, 5 = low
`status`: `pending` | `running` | `done` | `failed` | `skipped`

### Minimum viable experiment
Each experiment must change **exactly one or two** things from the current best config. If you're testing more than two changes at once, split it into separate experiments (this is ablation research, not kitchen-sink).

### Do not re-run failed experiments unless the base architecture has changed significantly.

---

## 6. Processing research/inbox/

When files exist in `research/inbox/`:

1. Read each `.md` file completely
2. Extract: **mechanism**, **why it might help**, **how to implement**, **which existing experiments it relates to**
3. Add extracted ideas to `research/hypotheses.md` under the appropriate section
4. Move the file from `research/inbox/` to `research/processed/` (rename, don't delete)
5. Add at least 1 concrete experiment to `experiments/queue.json` derived from each paper
6. Write a 2-sentence summary of what you learned from the paper at the top of the moved file

---

## 7. Leaderboard Maintenance

After every batch of experiments, update `experiments/leaderboard.md`:
- Keep a **historical progression** of all records — each row is a previous best that was beaten
- Include: exp_id, val_loss, Δ vs previous record, % improvement vs previous record, one-sentence key change description
- **Δ and % improvement are always relative to the immediately preceding leaderboard entry** (the previous record at the time that experiment ran), never relative to the original baseline

**Always note the active baseline** at the top of the leaderboard. If a new winner is found, it becomes the new baseline for subsequent experiments.

**CRITICAL: Always use the latest leaderboard record (lowest val_loss) as the baseline when designing new experiments.** All `flags_override` in queue entries must include all cumulative changes from the original baseline through to the current best, not just the new change being tested.

### Leaderboard entry minimum requirements
- An experiment may only enter the leaderboard if it improves val_loss by **more than 0.002** over the previous record (the noise threshold from Section 8)
- **No negative improvements and no noise-level improvements** — if Δ ≤ 0.002, the experiment is a neutral or loser and does not get a leaderboard row
- Experiments within 0.002 val_loss of the current record are considered ties and are not promoted

---

## 8. Known Issues

### g2_* config dispatch bug
All `g2_*` experiments in `ablation_results/6000000tok/` ran with identical configs due to a `_make()` function bug. Their results are noise — marked `config_bug=true` in `history.json`, excluded from all analysis.

### attn_pool_k4 and attn_pool_k8 anomalies
Invalid — K-pooling reduces effective sequence length making perplexity appear artificially low. Excluded.

---

## 9. Parallel Execution

When running multiple experiments:
- Use `CUDA_VISIBLE_DEVICES={gpu_id}` to assign one GPU per experiment
- Update `status` in `queue.json` to `running` before launching
- Update to `done` and append to `history.json` after completion
- Never assign the same GPU to two experiments simultaneously

### GPU consistency rule
**All experiments within a generation must run on the same GPU model.** Different GPU architectures (e.g. 3090 vs 4090 vs 5090) produce slightly different floating point results due to differing kernel implementations and cuDNN op ordering. This is safe to ignore for large deltas (>+0.3%), but can flip results for marginal ones (<+0.1%).

- Always record the GPU model in `history.json` for each experiment (`"gpu": "RTX 5090"` etc.)
- If switching GPU generations between batches, **re-run the current best baseline on the new GPU first** and use that re-run as the new reference point — never compare raw val_loss numbers across a GPU generation boundary
- Results within 0.002 val_loss of each other should be treated as a tie regardless of GPU

### Multi-contributor / multi-GPU collaboration protocol

When multiple people with different GPUs contribute to the same leaderboard:

**Contributor workflow (no coordination required during experiments):**
1. Pull latest code and leaderboard from the shared repo
2. Run the current baseline experiment (active baseline exp_id from leaderboard.md) on their GPU → record `local_baseline_loss`
3. Run as many experiments as they want freely — no need to claim or announce anything
4. Pick their single best result and submit a PR with:
   - Their experiment config (added to `configs/ablation_configs.py`)
   - A `history.json` entry including `val_loss`, `local_baseline_loss`, `delta`, and `gpu`

**PR / history.json entry format:**
```json
{
  "exp_id": "...",
  "val_loss": <raw on contributor GPU>,
  "local_baseline_loss": <baseline re-run on contributor GPU>,
  "delta": <local_baseline_loss - val_loss>,
  "gpu": "RTX 3090",
  "contributor": "alice"
}
```

**Promotion rule — owner verifies before leaderboard update:**
The contributor's best result is only added to the leaderboard after the **reference GPU owner** re-runs that exact experiment config on the reference hardware. The owner's run is the authoritative val_loss. If it beats the current baseline, it becomes the new baseline. Duplicate experiments across contributors are fine — cheap to run and provide useful variance data.

---

## 10. Reporting

After completing a batch of experiments:
1. Update `experiments/leaderboard.md`
2. Append results to `experiments/history.json`
3. Write a brief strategy note to `research/strategy.md`:
   - What this batch tested
   - What was learned
   - What the next batch should focus on
---

## 11. Environment Setup

export PATH="$PATH:/root/.local/bin" && echo 'export PATH="$PATH:/root/.local/bin"' >> ~/.bashrc


To run Claude with dangerous permissions (required for some scripts) in this environment:
```bash
IS_SANDBOX=1 claude --dangerously-skip-permissions
```
