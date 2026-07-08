# P2 — Fit a mixture scaling law, extrapolate the data mix

**Papers (ICML 2026):**
- Explaining Data Mixing Scaling Laws — https://openreview.net/forum?id=joReaAnwnH
- Capacity-Aware Mixture Law — https://openreview.net/forum?id=DDtL4VOUcT
- Background (2024 preprint): AutoScale — https://arxiv.org/abs/2407.20177

**Plain:** the best recipe of web/math/code data for a small model is NOT the best recipe for a
bigger one. Instead of guessing, fit a formula from cheap small runs and let it predict the mix.

**Implement:** domain-tagged corpus (FineWeb-Edu + FineMath + Stack-Edu slices), mixture weights
in the data prep, per-domain loss logging, a small script that fits the law and extrapolates.

**Runs:** 6–9 mixes at 25M (one mix = one claimable sub-task) → fit → 1 confirm run at 50M:
extrapolated mix vs the naive small-scale winner.

**Accept (one cell):** one mix trained, per-domain + held-out bits-per-byte reported, config diff + figure.
**Accept (full study):** extrapolated mix ≥ naive winner at 50M, beyond run-to-run noise.
