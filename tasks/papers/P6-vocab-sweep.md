# P6 — Is a 49k vocab wrong for a small model?

**Papers:**
- TokSuite: Measuring the Impact of Tokenizer Choice — https://openreview.net/forum?id=vIZz7LvObC (**ICML 2026 spotlight**)
- Over-Tokenized Transformer (ICML 2025) — https://arxiv.org/abs/2501.16975 — input vocab helps at
  ALL sizes, which complicates the naive "small model → small vocab" bet
- Embedding-layer learning rate under transfer — https://arxiv.org/abs/2605.21486 (preprint)

**Plain:** a small model spends a huge share of its parameters on a 49,152-word vocabulary. A
smaller vocabulary frees parameters for actual layers — win or loss?

**Implement:** train 8k/16k/32k BPE tokenizers on FineWeb-Edu; parameter-matched configs (freed
embedding params → more depth). Optional 4th arm: decoupled input/output vocab.

**Runs:** 4–5 at 25M preset.
**Metric trap:** per-token loss is incomparable across tokenizers — **bits-per-byte only**.

**Accept:** any vocab beats 49k at matched params beyond run-to-run noise. Config diffs + curves + figure, PR.
