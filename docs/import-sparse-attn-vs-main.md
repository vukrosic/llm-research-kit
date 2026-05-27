# Sparse Attention Import vs `main`

Branch: `codex/import-sparse-attn`

This branch imports the `sparse-attn` research repo as a nested subtree under `sparse-attn/`.
The diff against `main` is almost entirely additive.

## What Was Added

- `sparse-attn/models/minimax_sparse_attention.py`
- `sparse-attn/configs/minimax_sparse_config.py`
- `sparse-attn/training/tiny_trainer.py`
- `sparse-attn/experiments/attention_diagnostics.py`
- `sparse-attn/experiments/block_retrieval_probe.py`
- `sparse-attn/experiments/minimax_topk_sweep.py`
- `sparse-attn/experiments/train_tiny_real_compare.py`
- `sparse-attn/experiments/run_sparse_attention_study.py`
- `sparse-attn/experiments/run_mid_arch_ablation.py`
- `sparse-attn/tests/test_minimax_sparse_attention.py`
- `sparse-attn/data/real_dataset.py`
- `sparse-attn/data/download_speedrun_40m.py`
- `sparse-attn/docs/EXPERIMENT_PLAN.md`
- `sparse-attn/docs/RUNS.md`
- `sparse-attn/docs/sparse_attention_research_notes.md`
- `sparse-attn/docs/sparse_attention_short_paper.tex`
- `sparse-attn/sparse_attention_short_paper.pdf`
- `sparse-attn/README.md`
- `sparse-attn/CONTRIBUTING.md`
- `sparse-attn/requirements.txt`

## What Was Not Imported

- `sparse-attn/processed_data/`
- `sparse-attn/runs/`
- LaTeX build byproducts such as `*.aux`, `*.log`, and `*.out`

Those remain ignored in `sparse-attn/.gitignore`.

## What This Means

`main` still contains the dense baseline repo and the 5M/25M/50M baseline artifacts.
This branch adds a separate sparse-attention research lane without rewriting the dense baseline.

That is the right shape if the goal is to compare:

- dense baseline training behavior
- sparse attention routing behavior
- quality / throughput tradeoffs at matched token budgets

## Recommended Next Experiment

Run a matched comparison between:

- dense baseline: the existing 5M model in `llm-research-kit-scaling`
- sparse model: `sparse-attn` with `MiniMaxSparseAttention`

Keep these fixed:

- token budget
- sequence length
- batch size
- evaluation cadence
- random seed
- dataset slice

Start with the smallest cheap comparison first:

1. `5M dense` vs `5M sparse`
2. one short token budget
3. one seed
4. one plot with train loss, val loss, and throughput

Only after that should we scale to:

- multiple seeds
- longer token budgets
- a `top_k` sweep
- a block-size sweep

## Why This Order

The repo already has a lot of sparse-attention variants, but the first question is still whether the architecture is worth keeping at all.
If the 5M comparison is not clearly promising, there is no reason to spend time on a larger sweep.
If it is promising, the imported study scripts already give you the next ablation ladder.
