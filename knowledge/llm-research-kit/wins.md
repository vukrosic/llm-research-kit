# Wins

## Current Durable Beliefs

- The codebase implements a modular transformer with GQA, RoPE, and RMSNorm
- Muon optimizer is claimed to outperform AdamW
- 88M parameter default model (22 layers, 512 d_model)
- Supports torch.compile and mixed-precision (BF16) training

## Evidence

- README.md documents architecture and optimizer choices
- Training script (`train_llm.py`) is the main entrypoint
- Benchmark scripts exist for evaluation (ARC, GSM8K, Hellaswag)