# Sparse Attention 5M Run

Matched first pass against the existing 5M dense baseline.

## Dense Baseline

- Metrics: `plots/metrics_8000000_20260527_123338.json`
- Attention: dense
- Parameters: `6,652,800`
- Train token budget: `8,000,000`
- Actual tokens: `8,011,776`
- Sequence length: `2048`
- Batch size: `8`
- Dataset: `processed_data/pretrain_1B`
- Seed: `42`
- Final train loss: `5.2120`
- Final val loss: `5.4153`
- Final val accuracy: `0.1997`

## MiniMax Sparse

- Metrics: `plots/metrics_8000000_sparse_minimax_20260527_135944.json`
- Attention: `minimax_sparse`
- Sparse config: `block_size=16`, `top_k=8`, `index_dim=None`, `pooling=max`, `router_source=separate`
- Parameters: `6,685,824`
- Train token budget: `8,000,000`
- Actual tokens: `8,011,776`
- Sequence length: `2048`
- Batch size: `8`
- Dataset: `processed_data/pretrain_1B`
- Seed: `42`
- Final train loss: `5.6625`
- Final val loss: `5.8466`
- Final val accuracy: `0.1616`

## Result

This sparse setting underperformed the dense 5M baseline at the same token budget.

- Val loss delta: `+0.4313` sparse minus dense
- Val accuracy delta: `-0.0381` sparse minus dense
- Active training time: `81.60s` sparse

## Next Question

The first diagnostic is not whether sparse attention beats dense immediately. It is whether the router is too restrictive. The next cheapest ablation is a 5M run with a larger sparse window, for example `top_k=16`, while keeping the rest of the setup fixed.
