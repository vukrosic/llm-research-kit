# Experiment tasks — paper implementations

Open research tasks for lab contributors. Each file in `papers/` is one self-contained task from a
recent paper: what to read, what to implement, what to train, and what an accepted result looks like.

**Rules (all tasks):**
1. **Baseline first.** Your first contribution is always reproducing the pinned baseline config and
   reporting loss curve + final val loss + wall-clock + GPU. Every task compares against it.
2. Read the actual paper PDF before running anything.
3. Equal token/wall-clock budgets between arms, control run present, config diff + logs + **at least
   one figure** in the PR — no figure, not accepted.
4. Different tokenizers or corpora → report **bits-per-byte**, never per-token loss.
5. Accepted PRs get named credit on the published report.

| # | Task | Paper venue | Suggested preset | ~GPU cost |
|---|---|---|---|---|
| [P1](papers/P1-proxy-lr-ranking.md) | Do small-model verdicts survive? (proxy-LR check) | ICLR 2026 | 25M ×4 runs | $5–8 |
| [P2](papers/P2-mixture-scaling-law.md) | Fit a mixture scaling law, extrapolate the mix | ICML 2026 ×2 | 25M grid + 50M | $15–25 |
| [P3](papers/P3-quality-filter-threshold.md) | What does the quality filter actually do? | ICML 2026 | 25M ×2 | $4 |
| [P4](papers/P4-repeat-vs-mix.md) | Repeat the good data, or add more kinds? | ICML 2026 | 50M ×2 | $8–12 |
| [P5](papers/P5-decay-phase-data.md) | Put the best data where the model can still learn | ICML 2026 spotlight | 50M ×3 | $12–18 |
| [P6](papers/P6-vocab-sweep.md) | Is a 49k vocab wrong for a small model? | ICML 2026 spotlight | 25M ×4–5 | $10 |
| [P7](papers/P7-late-to-early.md) | Late-to-Early Training (read-and-gate) | preprint | 25M ×2 | $4 |
| [P8](papers/P8-synthetic-swap.md) | Swap in free public synthetic data | preprint + public corpus | 25M ×1 | $2 |

More papers to browse and propose from: [PAPER-SUGGESTIONS.md](PAPER-SUGGESTIONS.md).
