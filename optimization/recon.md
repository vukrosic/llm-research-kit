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
- Muon LR: 0.024, momentum: 0.95
- AdamW LR: 0.006, weight_decay: 0.2
- Schedule: constant (no warmup, no decay)
- AMP: BF16
- Batch size: 8, seq_len: 2048
- Grad clip: 1.0, dropout: 0.0
- torch.compile: True

## Data
- Dataset: processed_data/pretrain_1B/ (pre-tokenized, 33 arrow files)
- Tokenizer: HuggingFaceTB/SmolLM2-135M (vocab_size=49152)
- 10% validation split

## Tunable Axes
1. Learning rates (muon_lr, adamw_lr)
2. LR schedule (constant, cosine, linear) + warmup_ratio
3. Weight decay
4. Muon momentum
5. Gradient clipping
6. Batch size
7. Model dimensions (d_model, n_layers, d_ff, n_heads)
8. Gradient accumulation steps
9. Dropout

## Feasibility
- tokens_per_second: TBD (calibrating)
- 5s experiment: TBD tokens
- Noise floor: TBD (3 baseline runs with different seeds)
