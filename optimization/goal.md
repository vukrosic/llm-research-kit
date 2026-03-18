# Goal

**Discover whether short LLM runs can predict the 20s learning-rate winner through fast experiments at `5s`, `10s`, and `20s`.**

Focus areas:
1. How does the optimal Muon learning rate change across `5s -> 10s -> 20s`?
2. Is `5s` good for elimination but not final selection?
3. Is `10s` the minimum reliable selection tier for the `20s` winner?
4. Is top-2 prediction much easier than exact winner prediction?
5. What is the cheapest LR protocol that predicts the `20s` winner with low regret?

Metric: validation loss (cross-entropy)
Dataset: pretrain_1B (1B tokens pre-processed, used as data source for all experiments)
Architecture: 88M MinimalLLM transformer (baseline)

This is a focused research project about LR transfer across short durations, not a broad hyperparameter search.
