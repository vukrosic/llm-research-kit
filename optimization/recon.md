# Reconnaissance

## Hardware
- GPU: NVIDIA L40S, 46 GB VRAM, CUDA 13.0
- CPU: 28 cores
- RAM: 1 TB (749 GB available)
- Disk: 60 GB total, 26 GB free

## Model
- Architecture: MinimalLLM (dense transformer)
- Params: 88M (d_model=512, n_layers=22, n_heads=8, d_ff=2048)
- Features: GQA (4 KV heads), RoPE, RMSNorm, Squared ReLU, weight tying, Q/K norm
- Merged QKVO projection for efficiency

## Training Setup
- Optimizer: Hybrid Muon (2D tensors) + AdamW (norms/embeddings)
- Baseline Muon LR: 0.024
- Baseline AdamW LR: 0.006
- Weight decay: 0.2
- Schedule: constant
- AMP: BF16
- Batch size: 4
- Seq length: 2048
- Grad clip: 1.0
- Dropout: 0.0
- torch.compile: off for short experiments; eager mode is the active setup

## Data
- Dataset: `processed_data/pretrain_1B/` (pre-tokenized, 33 arrow files)
- Tokenizer: HuggingFaceTB/SmolLM2-135M (`vocab_size=49152`)
- Validation split: 10%

## Active Axes
1. Muon learning rate
2. Derived AdamW learning rate via `adamw_lr = muon_lr / 4`
3. Training duration: `5s`, `10s`, `20s`

Other tunable axes exist in the repo, but they are out of scope for the current research pass.

## Feasibility
- Throughput: about 50K tokens/sec in eager mode
- 5s run: about 250K tokens
- 10s run: about 500K tokens
- 20s run: about 1M tokens
- Primary evaluation strategy: use seed `42` for all ranking runs
- Tie-break rule: only run extra seeds if the top candidates at the same duration are within about `0.01` val_loss and the winner is too close to call
