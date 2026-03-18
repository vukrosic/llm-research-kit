# AI Research Roadmap

## Goal

Build the best possible 1B dense GPT model — matching or beating published 1B baselines (TinyLlama, Pythia-1B, OLMo-1B). Use it to prove the auto-research tool works. Every experiment = a post. Every result = content.

**Hard deadlines:**
- March 25: one publishable result for professor cold-emails
- May 1: visa expires — need concrete research portfolio by then

## Compute

| Resource | Value |
|----------|-------|
| Balance | $100 → ~357 hours at $0.28/hr |
| GPU | 1× L40S 48GB |
| 88M throughput | ~60k tokens/sec (measured) |
| 1B throughput | **unknown — must measure first** |
| 1B memory fit | **unknown — must verify first** |

## Current Model Architecture

| Component | Implementation |
|-----------|---------------|
| Attention | Merged QKVO projection, GQA (4 KV heads) |
| QK-norm | ✅ RMSNorm on Q and K (already present) |
| Positional | RoPE (base=10000) |
| FFN | Squared ReLU (Primer-style) |
| Normalization | Pre-norm RMSNorm |
| Embeddings | Tied input/output, scaled by √d_model |

## Existing LR Data (88M model)

| Duration | Tokens | Best LR | Val Loss | Ranking Stable? |
|----------|--------|---------|----------|-----------------|
| 5s | ~300K | 0.008 | 6.764 | — |
| 10s | ~620K | 0.007 | 6.513 | no (shifted) |
| 20s | ~1.2M | 0.006 | 6.263 | no (shifted) |
| 80s | ~5M | 0.012 | 5.177 | no (shifted again) |

All val losses at 80s are within ~0.08 of each other. Rankings have not stabilized. Need longer runs.

---

## Decision Process

**Now:** design experiments, pick variables, set token budgets, define "stable" concretely.
**After data:** pick winners, lock values, move to next variable.

**"Ranking stabilized" means:** the same top-2 candidates win at both N tokens and 2N tokens. If top candidates swap between adjacent budgets, the proxy is too short.

---

## Experiment Tracking

- `experiments.jsonl` — one line per run, append-only
- `decisions.md` — one entry per completed sweep: winner + reasoning + next step
- Evaluation: lowest area-under-val-loss (after warmup) → lowest final val loss → fastest wall-clock
- Reject if: loss spikes, NaNs, instability, constant clipping

---

# Experiments To Run Now

## EXP-0: 1B Feasibility Check (do this first, ~2 min)

**Question:** Can 1B (OneBConfig) train on L40S 48GB? What's the tokens/sec?

```bash
python train_llm.py \
  --config_class configs.llm_config.OneBConfig \
  --train_tokens 500000 \
  --dataset_path ./processed_data/pretrain_1B
```

**Why first:** if 1B doesn't fit, everything changes. If it's 5x slower than expected, compute budget changes. Takes 1-2 minutes.

**Record:** tokens/sec, memory usage, whether it completes without OOM.

## EXP-1: LR Ranking Stability (88M, 8M tokens, ~2-3 min each)

**Question:** Does the LR ranking stabilize when we go from 5M tokens (80s) to 8M tokens?

**Candidates:** top 4 from 80s data: 0.008, 0.012, 0.018, 0.024 (the default LLMConfig LR)

```bash
# Run each with 2 seeds
python train_llm.py --muon_lr 0.008 --adamw_lr 0.002 --train_tokens 8000000 --seed 42 --output_dir results/exp1_lr_8M
python train_llm.py --muon_lr 0.012 --adamw_lr 0.003 --train_tokens 8000000 --seed 42 --output_dir results/exp1_lr_8M
python train_llm.py --muon_lr 0.018 --adamw_lr 0.0045 --train_tokens 8000000 --seed 42 --output_dir results/exp1_lr_8M
python train_llm.py --muon_lr 0.024 --adamw_lr 0.006 --train_tokens 8000000 --seed 42 --output_dir results/exp1_lr_8M
```

**Decision rule:** if top-2 at 8M match top-2 at 5M (80s), ranking is stable → lock LR.
If ranking shifts → run top 3 at 20M tokens.

## EXP-2: LR Confirmation at 20M tokens (only if EXP-1 ranking shifts)

**Candidates:** top 3 from EXP-1.
**Token budget:** 20M tokens (~5-6 min each).
**Decision rule:** same — do top-2 match EXP-1? If yes, lock LR.

## EXP-3: Quick Transfer Check (88M vs 1B ranking)

**Question:** does the 88M LR winner also win at 1B?

**Method:** run the 88M winner AND runner-up at 1B for 500K-1M tokens (minimum to get a val loss reading). If rankings match, we trust the proxy for future experiments.

**Why early:** this validates or invalidates the entire "screen at 88M, confirm at 1B" strategy. Do it NOW, not in week 3-4.

## EXP-4: Warmup Sweep (after LR is locked)

**Variable:** `warmup_ratio` at locked LR
**Values:** 0.0, 0.005, 0.01, 0.02
**Token budget:** 8M tokens each at 88M
**Note:** current 88M default is warmup_ratio=0.0. The OneBConfig default is 0.01. There's a gap here — we need to know which is better.

## EXP-5: Weight Decay Sweep (after warmup is locked)

**Variable:** `weight_decay`
**Values:** 0.05, 0.1, 0.15, 0.2
**Token budget:** 8M tokens each at 88M, confirm winner at 20M

---

# Experiments To Design Later (after optimization is locked)

## Architecture experiments

Only test changes with published evidence AND that differ from current architecture:

| Experiment | Change | Current → New | Lines of code |
|-----------|--------|---------------|---------------|
| ARCH-001 | SwiGLU FFN | Squared ReLU → SwiGLU | ~15 lines |
| ARCH-002 | Residual attention | No cross-layer → residual attn weights | ~20 lines |
| ARCH-003 | Embedding scale | `× √d_model` → remove or replace | ~2 lines |
| ARCH-004 | Residual scaling | Uniform → `1/√n_layers` | ~5 lines |

**Removed from previous roadmap:**
- ~~QK-norm~~ (already in the model)
- ~~Alternative attention patterns~~ (too vague, no clear single change to test)
- ~~Norm placement~~ (already using pre-norm RMSNorm, the modern standard)

## Scaling experiments

These happen naturally as part of optimization:
- SCALE-001: 88M→1B transfer = EXP-3 above (moved earlier)
- SCALE-002: optimal LR vs token budget = already visible in existing 5s/10s/20s/80s data

---

# Content Strategy

| When | Post | Expected reach |
|------|------|---------------|
| Today | "LR ranking shifts with training duration — 5s vs 80s results" | X: 2k-5k |
| After EXP-1 | "Does the LR winner stabilize at 8M tokens?" + auto-research tool mention | X: 3k-8k |
| After EXP-3 | "Can 88M experiments predict 1B results?" (the scaling question) | X: 5k-15k |
| After optimization | "Best training recipe for a 1B model on $X of compute" | X: 10k-30k |
| After architecture | "SwiGLU vs Squared ReLU at 1B — which FFN wins?" | X: 5k-15k |

View estimates based on actual X analytics (2k-15k daily, viral spikes to 50-84k). Research updates are niche content — realistic range is 2k-8k per post, with occasional 15k+ breakouts on scaling-law or cost-efficiency angles.

**Novita ask for $1,000:** after 15-20 posts with mentions, cumulative 50k-100k views across platforms. Send engagement report.

---

# Practical Rules

1. Never change multiple knobs at once.
2. 88M for screening, 1B for confirmation.
3. Ranking = "stable" when top-2 match at N and 2N tokens.
4. Record everything in experiments.jsonl. Decisions in decisions.md.
5. Every experiment = a post. Failures are content.
6. Use the auto-research tool to run batches when possible — it's the product.

---

# Execution Order

1. ☐ **EXP-0:** 1B feasibility check on L40S (2 min)
2. ☐ **EXP-1:** LR at 8M tokens, 4 candidates (10-15 min)
3. ☐ **Post:** LR ranking shift results (today)
4. ☐ **EXP-2:** LR at 20M tokens if needed (15-20 min)
5. ☐ **EXP-3:** Quick 88M→1B transfer check (5-15 min)
6. ☐ **Post:** "Can 88M predict 1B?" results
7. ☐ **EXP-4:** Warmup sweep at locked LR (10-15 min)
8. ☐ **EXP-5:** Weight decay sweep (10-15 min)
9. ☐ Lock full optimization recipe
10. ☐ Implement ARCH-001 (SwiGLU) behind config flag
11. ☐ Test at 88M, confirm at 1B if it wins
12. ☐ Full 1B training with best recipe
