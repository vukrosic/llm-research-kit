# Runs

## 2026-05-27 Tiny Real-Data Comparison

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python experiments/train_tiny_real_compare.py \
  --token_budget 500000 \
  --seq_len 256 \
  --batch_size 8 \
  --eval_every 25 \
  --device auto \
  --run_root runs/tiny_real_compare
```

Dataset:

- Source: `vukrosic/blueberry-1B-pretrain`, split `train[:20000]`
- Local path: `processed_data/speedrun_40M`
- Tokenizer: `HuggingFaceTB/SmolLM2-135M`
- Run slice loaded: 602,112 tokens
- Training budget: 500,000 tokens

Model:

- Dense: 5,309,856 parameters
- MiniMax sparse: 5,337,696 parameters
- Shared shape: `d_model=96`, `n_layers=6`, `n_heads=6`, `n_kv_heads=2`, `d_ff=384`, `seq_len=256`
- Sparse routing: `block_size=16`, `top_k=2`, `index_dim=16`

Results:

| Model | Device | Tokens | Seconds | Final val loss | Val ppl | Val acc |
|---|---:|---:|---:|---:|---:|---:|
| dense | mps | 501,760 | 88.49 | 7.4787 | 1769.88 | 0.0843 |
| minimax_sparse | mps | 501,760 | 69.11 | 7.4860 | 1782.89 | 0.0905 |

Artifacts:

- Run directory: `runs/tiny_real_compare/20260527_153627`
- Plot: `runs/tiny_real_compare/20260527_153627/comparison.png`
- Dense metrics: `runs/tiny_real_compare/20260527_153627/dense/metrics.jsonl`
- Sparse metrics: `runs/tiny_real_compare/20260527_153627/minimax_sparse/metrics.jsonl`

Interpretation:

This is a smoke run, not a claim about quality. Both models learn on the real 40M dataset slice and land almost tied after 500k tokens. The sparse model is slightly worse on final validation loss but finished faster in this unfused MPS run. The next serious run should repeat across 3 seeds and at least one longer token budget before treating the delta as signal.

Follow-up check:

- Added a seed-controlled training `DataLoader` generator after this run. The original comparison reset the model seed for each model, but did not isolate the shuffle RNG from architecture-specific initialization. That did not make the models identical, but it was a fairness bug.
- Added `experiments/attention_diagnostics.py`. On `seq_len=256`, the untrained dense and sparse models have mean absolute logit difference about `0.221`, and sparse attends to about `37.3%` as many key/value tokens as full dense causal attention in the first sparse layer.
- A short seeded-loader check at `runs/tiny_real_compare_seeded_check/20260527_154440` still produced nearly overlapping curves after 100k tokens: dense `val_loss=9.2845`, sparse `val_loss=9.2821`.

## 2026-05-27 Longer 1M-token comparison

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python experiments/train_tiny_real_compare.py \
  --token_budget 1000000 \
  --seq_len 256 \
  --batch_size 8 \
  --eval_every 50 \
  --device auto \
  --run_tag 1m_long \
  --run_root runs/tiny_real_compare_long
```

Results:

| Model | Device | Tokens | Seconds | Final val loss | Val ppl | Val acc | Tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | mps | 1,001,472 | 92.00 | 6.8665 | 959.54 | 0.1140 | 10,885.95 |
| minimax_sparse | mps | 1,001,472 | 127.24 | 6.8866 | 979.05 | 0.1083 | 7,870.67 |

Artifacts:

- Run directory: `runs/tiny_real_compare_long/20260527_155106_1m_long`
- Plot: `runs/tiny_real_compare_long/20260527_155106_1m_long/comparison_1m_long.png`
- Plot tags: `runs/tiny_real_compare_long/20260527_155106_1m_long/plot_tags.json`

Interpretation:

At 1M tokens, dense is still slightly ahead on validation loss, and the gap is visible in the delta panel, but the models remain very close. The throughput gap on MPS is larger than the loss gap: dense was faster in this longer run, while sparse remained competitive in quality. This argues for ablations that change routing behavior, not just longer next-token training, if we want a clearer architectural signal.

## 2026-05-27 Scale, context, and architecture sweep

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python experiments/run_sparse_attention_study.py \
  --train_token_budget 80000 \
  --dataset_token_budget 300000 \
  --batch_size 4 \
  --device auto \
  --eval_batches 4
```

Artifacts:

- Run directory: `runs/sparse_attention_study/20260527_161935`
- Plot: `runs/sparse_attention_study/20260527_161935/study_summary.png`
- Results JSON: `runs/sparse_attention_study/20260527_161935/study_results.json`
- Results CSV: `runs/sparse_attention_study/20260527_161935/study_results.csv`

Interpretation:

Three scale settings were trained at 256-token context and then evaluated at 256, 512, and 1024 tokens. Dense stayed slightly ahead of sparse at every scale, but the gap remained modest and the longer-context eval did not collapse.

The mid-scale router ablation showed that pooling style and shared-router projection were mostly degenerate at this budget, while `top_k` and block size gave the clearest architectural signal. `top_k=4` improved the loss a bit at the cost of throughput and higher routing density, while smaller blocks were faster and kept the loss close to the baseline.

## 2026-05-27 Mid-scale architecture ablation

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python experiments/run_mid_arch_ablation.py \
  --train_token_budget 80000 \
  --dataset_token_budget 300000 \
  --batch_size 4 \
  --device auto \
  --eval_batches 4
```

Artifacts:

- Run directory: `runs/mid_arch_ablation/20260527_162441`
- Plot: `runs/mid_arch_ablation/20260527_162441/arch_summary.png`
- Results JSON: `runs/mid_arch_ablation/20260527_162441/study_results.json`
- Results CSV: `runs/mid_arch_ablation/20260527_162441/study_results.csv`

Interpretation:

The probe density from a held-out real batch moved in the expected direction with `top_k` and block size: `top_k=1` kept about 23% of dense causal tokens, `top_k=2` about 37%, and `top_k=4` about 59%. The best validation loss in this quick pass came from `top_k=4`, but it was also the slowest. Block size 8 was the fastest of the block-size variants and stayed close to the baseline loss, while block size 32 pushed density up and slowed the model down.
