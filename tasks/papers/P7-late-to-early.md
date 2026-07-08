# P7 — Late-to-Early Training (read-and-gate task)

**Paper:** LET: LLMs Learn Earlier, So Faster and Better — https://arxiv.org/abs/2602.05393
(2026 **preprint, not peer-reviewed**)

**Plain:** a fresh preprint claims a trainer tweak makes models learn useful structure earlier.
Unvetted — step one is deciding whether it's even real. "This paper isn't worth a run" is a fully
accepted outcome.

**Step 1 (the actual task):** read the PDF critically. Do the baselines look fair? Are budgets
matched? Write a short GO / NO-GO note with your reasons.

**Step 2 (only on GO):** implement the change behind a flag, 1 run + control at 25M preset.

**Accept:** the gate note (either verdict), or the A/B result on GO. Config diff + curves + figure
if you ran it, PR either way.
