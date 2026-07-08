# P1 — Do small-model verdicts survive? (proxy-LR check)

**Paper:** Can Small Training Runs Reliably Guide Data Curation? — https://arxiv.org/abs/2512.24503 (**ICLR 2026**)

**Plain:** labs pick training data using small cheap models. This paper says those picks are often
wrong at full size — unless the small model trains with a *lower learning rate*. We test that claim
on our own setup.

**Implement:** nothing new — run a corpus A/B (plain FineWeb vs FineWeb-Edu, same token budget,
25M preset) at three peak LRs: default, 0.5×, 0.25×. Six runs, or split with another contributor.

**Accept:** table of Edu-vs-plain ranking at each LR + one paragraph: does the verdict flip or
strengthen as LR drops? Config diffs + curves + one figure, PR.

**Why it matters:** if reduced LR changes the verdict, every future cheap screening run here
adopts it. Highest leverage-per-dollar task on the board.
