# Goal

**Discover scaling laws and optimal hyperparameter configurations through fast experiments (5-20 seconds).**

Focus areas:
1. How do optimal hyperparameters change as training duration increases? (5s → 10s → 20s)
2. Which hyperparameters transfer across scales and which don't?
3. What is the relationship between model size, learning rate, and optimal training duration?
4. Can we predict longer-run performance from short-run metrics?

Metric: validation loss (cross-entropy)
Dataset: pretrain_1B (1B tokens pre-processed, used as data source for all experiments)
Architecture: 88M MinimalLLM transformer (baseline)

This is a research project about understanding scaling behavior, not about achieving a single best loss number.
