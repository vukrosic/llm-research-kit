# Research Inbox

Drop papers, ideas, or notes here as `.md` files.

The AI will:
1. Read each file and extract testable hypotheses
2. Add ideas to `research/hypotheses.md`
3. Add concrete experiments to `experiments/queue.json`
4. Move the file to `research/processed/` with a summary prepended

## Format suggestions

You can paste anything — a paper abstract, your own notes, a link + summary, a half-formed idea.
The more context the better, but even a one-liner works:

```markdown
# Idea: Grouped Query Attention with asymmetric heads
Tried in Mistral-7B. Theory: fewer KV heads = less memory, similar quality.
Key paper: https://arxiv.org/abs/2305.13245
Hypothesis: GQA with 2 KV heads might work better than 4 for our 512-dim model.
```

## What makes a good inbox entry

- **Paper notes**: paste the abstract + which section is relevant + your interpretation
- **Ideas**: describe the mechanism and why you think it'll help
- **Failure analysis**: if something didn't work, describe why and under what conditions it might

## What NOT to put here

- Completed experiment results (those go in `experiments/history.json`)
- Bug reports (open a GitHub issue or fix it directly)
