# Wins

## Current Durable Beliefs

- The codebase implements a modular transformer with GQA, RoPE, and RMSNorm
- Muon optimizer is claimed to outperform AdamW
- 88M parameter default model (22 layers, 512 d_model)
- Supports torch.compile and mixed-precision (BF16) training

## Activation Function Findings (2026-04-04)

**Experiment:** activation-discovery, 5 activations, 88M model, 2M tokens, TitanX Pascal 12GB

**Result ranking (final val_loss, lower = better):**
1. relu:         6.0598
2. gelu:         6.0713  (+0.011 vs relu)
3. silu:         6.0777  (+0.018)
4. swiglu:       6.0995  (+0.040)
5. squared_relu: 6.1084  (+0.049)  ← current default

**Key verdict:** Default `squared_relu` is worst. Plain `relu` wins, with `gelu` and `silu` close.

**Mechanism breakdown:**
- Smoothness (ReLU vs GELU): negligible benefit (-0.011 in GELU's direction)
- Quadratic (SquaredReLU vs ReLU): hurts by 0.049 — quadratic amplification is counter-productive here
- Gating (SwiGLU vs SiLU): gating hurts by 0.022 at this token budget

**Warning:** All differences are small (~0.05 range). Needs validation at higher token count (8M+).

**Micro-scale debug (debug_activations.py, tiny model, random data):**
- relu > squared_relu > gelu > silu > swiglu — same ranking at top, confirms relu.
- Debug results on random data are consistent direction but not magnitude.

## Architecture Notes

- **Model**: 88M parameters default (configurable)
- **Layers**: 22 Transformer blocks
- **Hidden Dimension**: 512
- **Feed-Forward Dimension**: 2048
- **Attention**: 8 query heads, 4 KV heads (Grouped Query Attention)
- **Positional Encoding**: Rotary (RoPE) — self-contained implementation (no torchtune dep)
- **Normalization**: Pre-norm RMSNorm
- **Activation**: Squared ReLU (Primer-style) — **recommended to change to relu or gelu**
- **Vocab Size**: 49,152
- **Sequence Length**: 2048 tokens

## Key Components

- Weight tying between token embeddings and LM head
- Fused QKVO projection
- QK-Normalization for training stability
- Muon optimizer support (orthogonal updates)

## Environment Notes (TITAN X Pascal, 2026-04-04)

- GPU: NVIDIA TITAN X (Pascal), 12 GB VRAM, CC 6.1
- PyTorch: 2.4.0+cu118 (downgraded from 2.11+cu130 to support CC 6.1)
- torchtune/torchao incompatible with torch 2.4 — replaced with self-contained RoPE
- BF16 AMP: works but no hardware BF16, software emulation
- torch.compile: disabled (CC 6.1 not in precompiled kernel list)
- Flash attention: not available on Pascal — SDPA uses math kernel (high VRAM per batch)
- Effective batch: 2 (seq=2048) × 4 grad accumulation = effective batch 8
- Throughput: ~3800 tokens/s at steady state after eval-step warmup
- Per-run time: ~10 min at 2M tokens, ~42 min at 8M tokens
