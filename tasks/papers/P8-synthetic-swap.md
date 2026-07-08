# P8 — Swap in free public synthetic data (one cell)

**Papers:** BeyondWeb — https://arxiv.org/abs/2508.10975 (preprint, the motivation);
Cosmopedia-v2 — https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus (the public corpus we use).

**Plain:** companies pay to have LLMs rewrite web text into textbook-style data. The rewritten
corpora are already public and free — does adding them help at our scale?

**Implement:** one corpus variant — 25% of FineWeb-Edu tokens replaced by Cosmopedia-v2,
token-matched against the pure-Edu baseline.

**Runs:** 1 at 25M preset (the baseline is your control). Cheapest task on the board.

**Accept:** held-out bits-per-byte vs baseline + a one-paragraph verdict on whether public
synthetic data earns a slot in the mix. Config diff + curve + figure, PR.
