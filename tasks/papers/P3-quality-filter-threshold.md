# P3 — What does the quality filter actually do? (threshold arm)

**Paper:** Removing Noise, not Finding Gold: Quality Filtering for Large-Scale Pretraining —
https://openreview.net/forum?id=2taaKYQR7h (**ICML 2026**)

**Plain:** everyone filters web data by "educational quality." This paper argues the win comes from
throwing out junk, not from keeping only the very best. If true, pushing the filter harder stops
helping. Cheap to test.

**Implement:** prepare FineWeb-Edu shards at two classifier thresholds (score ≥2 vs ≥3),
token-matched. Before running, write down your prediction: plateau or no plateau?

**Runs:** 2 at 25M preset.

**Accept:** threshold verdict (helps / plateaus / hurts) on shared held-out bits-per-byte + one
paragraph interpreting it through the paper's noise-removal framing. Config diffs + curves + figure, PR.
