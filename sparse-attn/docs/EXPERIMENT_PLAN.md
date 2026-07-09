# MiniMax Sparse Attention Experiment Plan

## Current Baseline

The dense/GQA baseline lives in the separate `vukrosic/llm-research-kit` repository. Keep that checkout unchanged so dense training and existing metrics remain a clean reference.

## Implemented Variant

`models/minimax_sparse_attention.py` implements the diagrammed MiniMax sparse block:

- index query: one query per GQA group
- index key: one shared index key stream
- routing: causal index attention, block max pool, top-k selected blocks
- sparse branch: normal Q heads attend only to selected KV blocks from their GQA group
- causal protection: future tokens inside the selected current block are masked

## First Experiments

1. **Correctness smoke**
   Run `python -m unittest discover -s tests`.
   This checks shape, gradients, causality, top-k block selection, and LLM integration.

2. **Router sanity**
   Run `python experiments/block_retrieval_probe.py`.
   This tests whether block max pooling recovers a planted matching block before we train anything.

3. **Prefill microbench**
   Run `python experiments/minimax_topk_sweep.py --seq_lens 128 256 512 1024`.
   This gives early latency and theoretical sparsity curves across `top_k` and `block_size`.

4. **Training comparison**
   Add a `train_llm.py` adapter that matches the baseline data loader and trains:
   `dense`, `minimax_sparse top_k=2`, `minimax_sparse top_k=4`, and `minimax_sparse top_k=8`.
   Use identical seeds, token budgets, batch sizes, and eval milestones.

5. **Long-context retrieval**
   Before spending GPU time on pretraining, use a synthetic copy/needle task where the answer depends on a far earlier block.
   This directly tests whether the sparse router keeps the block that matters.

## Main Risk

This implementation is intentionally unfused PyTorch. It will not reproduce the large speedups from the image yet. The first research question is quality and routing behavior; kernel work comes only after the selection policy is worth keeping.
