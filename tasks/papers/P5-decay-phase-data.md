# P5 — Put the best data where the model can still learn it

**Papers:**
- Midtraining Bridges Pretraining and Posttraining Distributions — https://openreview.net/forum?id=5PfEQzE9bf (**ICML 2026 spotlight**)
- AC-ODM: Actor–Critic Online Data Mixing — https://openreview.net/forum?id=2bKKamEtto (**ICML 2026**)
- The warning: How Learning Rate Decay Wastes Your Best Data — https://arxiv.org/abs/2511.18903 (preprint)

**Plain:** labs save their best math/code data for the end of training. But by then the learning
rate is tiny, so the model can barely absorb it. Schedule the data and the learning rate *together*.

**Implement:** two-stage data loader (stable-phase mix → decay-phase mix with math/code
concentrated), with the decay fraction exposed as a flag.

**Runs:** 3 at 50M preset — uniform mix / decay-concentrated / decay-concentrated + longer decay.

**Accept:** does concentration beat uniform beyond run-to-run noise on shared held-out
bits-per-byte? Config diffs + curves + figure, PR. Stretch (own task, only if this wins):
AC-ODM's online reweighting.
