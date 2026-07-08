# P4 — Out of data: repeat the good stuff, or add more kinds?

**Paper:** InfoLaw: Information Scaling Laws with Quality-Weighted Mixture Data and Repetition —
https://openreview.net/forum?id=fQaVptMRCY (**ICML 2026**)

**Plain:** small models are often trained on many times more tokens than "optimal." At that point,
is it better to repeat the best data twice, or add math/code you haven't used yet?

**Implement:** two matched-token corpora — (A) FineWeb-Edu repeated ~2 epochs, (B) single-epoch
FineWeb-Edu + FineMath + Stack-Edu union.

**Runs:** 2 at 50M preset, shared held-out bits-per-byte.

**Accept:** repeat-vs-add verdict beyond run-to-run noise + a comparison against what the paper's
law predicts. Config diffs + curves + figure, PR.
