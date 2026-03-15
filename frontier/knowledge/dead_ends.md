# Dead Ends — What Didn't Work and Why

This file prevents the research loop from repeating failed approaches. Every entry must explain **WHY** it failed, not just that it did.

---

## From Ablation System (Transferred Knowledge)

These mechanisms were tested within the transformer and found to be losers. They may or may not transfer to other architecture families.

### Catastrophic Losers (>2% worse)
- **Muon cautious**: -2.6% to -3.2%. Conservative updates hurt at small scale where aggressive exploration of loss landscape is needed.
- **Muon update_clip**: -12% to -13%. Clipping Muon updates destroys the orthogonalized gradient direction.
- **Muon frob_scale**: -19%. Frobenius scaling collapses effective learning rate.
- **Post-norm architecture**: Catastrophic. Gradient flow through residual stream is critical; post-norm blocks it.
- **No QK norm**: Catastrophic. At 88M scale, attention logits explode without normalization.
- **Tied weights=False**: Catastrophic. Untying wastes parameters at 88M scale.
- **Parallel block (PaLM-style)**: -0.5% to -1%. Mixing attention and FFN outputs adds noise at this scale.

### Consistent Losers (0.2% to 2% worse)
- **Gated residual on row+rms baseline**: -0.28% to -0.82%. Conflicts with Muon gradient normalization.
- **Value norm**: Normalizing value vectors removes useful magnitude information.
- **Depth-scaled init**: Over-dampens initial gradient flow at 22 layers.
- **GPT-2 init**: Output projection scaling too aggressive for 22 layers.

---

## From Frontier Experiments

*Entries will be added here as frontier experiments are run.*
