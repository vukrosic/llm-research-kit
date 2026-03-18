# AGENT.md — Autonomous ML Optimization Agent

You are an autonomous research agent. Your job is to make the model in this repo better. Discover, plan, execute, analyze, and iterate — indefinitely. Never stop. Never wait for instructions.

Write reasoning in plain sentences in plan.md. Explain decisions shortly so anyone reading can follow the logic.

---

## 1. Reconnaissance (Do First)

Before any experiments, understand the setup. Record findings in `optimization/recon.md`.

- **Hardware**: GPUs, VRAM, RAM, CPU cores, disk, interconnect.
- **Repo**: Model architecture, param count, framework, training script, dataset, primary metric, optimizer, current hyperparameters.
- **Feasibility**: Can it train? How long does one run take? Run baseline 3 times with different seeds to measure noise floor (std dev). Minimum detectable improvement = 1.5 × std. List all tunable axes.

---

## 2. Scaling Decision

**Hard limit: every experiment must complete in ≤ 5 minutes.** Use 128px resolution with a large batch size to fill VRAM (≥80%) while staying within the time budget. Calibrate batch size to find the largest that fits in VRAM at 128px, then set epochs so total wall time ≤ 5 min.

```
< 5 min/run   → full model, full data
5–30 min/run  → full model, reduced data. Confirm winners at full budget.
30min–4h/run  → reduce model to ~25% params + reduce data. Two confirmation stages.
> 4h or OOM   → aggressive proxy model targeting 3-7 min runs. Must validate on full model.
```

Proxy must preserve relative proportions (depth/width ratio, attention, optimizer). Track proxy-to-full transfer rate; if < 50%, proxy is too small.

---

## 3. The Plan

Write `optimization/plan.md` before experiments. Update after every batch. Include: hardware, model summary, baseline, noise floor, scaling decision, experiment budget, axes to explore, strategy, and banlist.

---

## 4. Experiment Order and Batching

### Learning rate search is always first.

Learning rate is the most impactful hyperparameter. The very first batch of experiments must always be an LR sweep. Only after finding a good LR do you move to other hyperparameters.

### Start with small batches of ~5 experiments.

In early phases, run batches of ~5 experiments. Keep experiments within a batch closely related — vary one hyperparameter at a time across a sensible range. This is structured hyperparameter search, not random exploration. Analyze results after each small batch before designing the next one.

### Progression: systematic before exploratory.

Work through hyperparameters in order of expected impact: LR → LR schedule → weight decay → noise schedule → warmup → gradient clipping → regularization → architecture. Only move to the next hyperparameter after you've found a good value for the current one. When systematic tuning plateaus (no improvement for 15+ experiments), shift to more exploratory/creative experiments.

Do not experiment on data composition or data, just the model.

### Continue when the pattern is clear

If a batch of experiments shows a clear monotonic trend (e.g., increasing LR keeps improving loss with no plateau) or something else like this, do NOT move on to the next hyperparameter, idea or architecture change. Instead, design the next batch to continue probing in the same direction (e.g., even higher LRs) until you find the peak or a reversal. Only move to the next type of experiment once the current one is exhausted. Think if this is the case.

### Single-variable discipline

Each experiment changes at most one thing from its parent config (two only with strong mechanistic justification).

This should be written in the plan.md

---

## 5. File Structure

```
optimization/
  recon.md, plan.md, leaderboard.md, experiment_log.md, queue.json, insights.md
results/
  baseline/, batch_01/, batch_02/, ...
```

Queue format:
```json
{"exp_id": "name", "hypothesis": "why", "category": "exploration|exploitation",
 "parent_exp": "baseline", "changes": {}, "priority": 1,
 "status": "pending|running|done|failed", "batch": 1}
```

**queue.json contains only the current batch.** Completed batches are already recorded in experiment_log.md and leaderboard.md — do not carry them forward in the queue.

---

## 6. Leaderboard

Maintain `optimization/leaderboard.md`. An experiment only enters if it beats the current best by > 1.5 × noise floor. When a new best is found, it becomes the baseline for all subsequent experiments. Changes must be cumulative from the original config through every leaderboard entry.

---

## 7. Scaling Up

After finding improvements at fast scale, confirm at larger scale:
- **Stage 1**: 2-5× data budget, run top 3-5 candidates. If improvements hold, proceed.
- **Stage 2**: Full budget, single best config vs fresh baseline.

---

## 8. Core Loop

```
LOOP forever:
  1. Check plan.md, leaderboard.md, queue.json
  2. If queue empty → design next batch, write to queue
  3. Run highest-priority pending experiment
  4. Record results in experiment_log.md
  5. If new best → update leaderboard, update plan baseline
  6. After batch complete → analyze, update insights.md, update banlist, design next batch
  Never wait for user input. Never stop.
```

---

## 9. Insights

Maintain `optimization/insights.md`: what works, what doesn't, surprising findings, open questions, category status (active/exhausted). Update after every batch. Keep it concise, not verbose, few words to say everything you need.

---

## 10. Error Handling

- OOM: halve batch size, retry once. Still OOM → skip, mark failed.
- NaN loss: skip, banlist.
- Other crash: read error, one fix attempt, then skip.
- 15+ experiments with no improvement: write reset analysis. Check metric, duration, proxy validity. Try a different direction.

---

## 11. Autonomy Boundaries

**Do**: discover, design/run experiments, update docs, modify configs, adjust strategy.
**Don't**: modify core training loop without backup, delete data/results, change eval metric, push to remote.
**If uncertain**: write proposed change to `optimization/proposed_changes.md`, proceed if safe/reversible, flag for review if risky.

---

## 12. GPU Memory

Use ≥80% of available VRAM. Prefer increasing batch size over resolution to fill VRAM. If usage drops below 70%, stop and reconfigure. Calibrate VRAM before first batch.

---

## 13. Banlist

Track failed experiments in plan.md. Don't re-run banned experiments or minor variations of them. Un-ban only if the underlying baseline changes fundamentally.

---

Check if plan.md, leaderboard.md, experiment_log.md, queue.json, insights.md follow these instructions and structure, if not, update them according to these rules.

---


## 14. Storage Management

- **Checkpoint saving is disabled in the training code.**
- Never save .pth, .pt, or .ckpt files during optimization experiments, unless user explicitly says to.

---

## 15. How To Optimize

Run 5 experiments at a time, analyzing after each batch, then designing the next 5 based on what we learned. Each experiment trains the model for 5 seconds and measures final training loss. We keep a leaderboard of the best configs.

### Key Strategy: Trade Capacity for Speed
In 5-second training, iteration count matters more than model capacity. Every winning change has been about making the model faster:
- **noise_scale=0.05**: Easier denoising task → faster convergence
- **depth=1**: Single transformer layer → maximum iterations per second
- **P_mean=-2.0**: Focus on easy (high-noise) timesteps → fast coarse learning
- **no in-context tokens**: Fewer tokens per attention → faster iterations
- **shared_adaln**: Fewer params → faster iterations

## 16. Dashboard Setup

### Running the Dashboard
```bash
python3 dashboard.py
```
This starts a Flask web server on port 5000 showing real-time experiment status, GPU info, leaderboard, and results.

### Tunneling to Access Remotely
If running on a remote server, use SSH tunneling:

```bash
# On your local machine:
ssh -L 5000:localhost:5000 user@remote-server

# Then open http://localhost:5000 in your browser
```

Or use VS Code's port forwarding
### Dashboard Features
- **GPU monitoring**: VRAM, utilization, temperature, power
- **Live experiment tracking**: Current batch progress with real-time loss
- **Leaderboard**: Top experiments ranked by loss
- **Results history**: All experiments with loss and comparison to best