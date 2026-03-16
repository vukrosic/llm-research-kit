# Frontier Architecture Insights

AI-maintained log of cumulative insights from frontier experiments.

---

## Core Discovery: Progressive Conv→Attn Architecture

**The breakthrough architecture**: Multi-head causal convolutions in early layers (75%), windowed causal attention in late layers (25%). Beats transformer by 3-4% at 12M tokens.

### Why it works:
1. **Conv layers are excellent local feature extractors** — exponentially-spaced kernels (3,5,9,17,33,65) capture patterns at multiple scales efficiently
2. **Attention layers provide global context** — even with small windows (128-256), they capture long-range dependencies that conv cannot
3. **Progressive ordering matters** — conv-first → attn-last is better than interleaved (3.59 vs 3.62). The model benefits from extracting local features before attending globally.
4. **The combination overcomes both weaknesses**: pure conv plateaus at 12M tokens, pure attention is slower to start. Together, they're better than either alone.

### Design space exploration (Batch 11):
| Variant | val_loss | Key difference |
|---------|----------|----------------|
| ProgressiveMoE | 3.5697 | + MoE FFN (1.6x params) |
| ProgressiveDeep (20L) | 3.5876 | deeper, narrower FFN |
| ProgressiveWide (w=256) | 3.5890 | wider attention window |
| ProgressiveHalf (50/50) | 3.5899 | more attention layers |
| Progressive (75/25, w=128) | 3.5921 | original design |
| Interleaved | 3.6186 | alternating conv/attn |

### Key insights from exploitation:
- **Conv/attn ratio barely matters**: 50/50 vs 75/25 is within noise (0.002)
- **Window size barely matters**: 128 vs 256 is within noise
- **Depth helps**: 20L with narrower FFN matches 16L with wider FFN
- **MoE helps but unfairly**: 1.6x params for 0.02 improvement
- **Interleaved is strictly worse**: progressive ordering > interleaving

## Scaling Behavior

- **At 6M tokens**: Conv architectures win (4.04 vs 4.09 for transformer)
- **At 12M tokens**: Pure conv ties transformer (3.71 ≈ 3.71)
- **At 12M tokens**: Progressive conv→attn beats both (3.59 vs 3.71)
- **Implication**: Conv has better inductive bias (wins early) but attention has better capacity (catches up). Combining them captures both advantages.

## Multi-Head Causal Conv (MHConv)

The core convolution primitive used across all winning architectures:
- Per-head depthwise causal convolutions
- Exponentially-spaced kernel sizes: 3, 5, 9, 17, 33, 65
- SiLU activation per head, gated output projection
- ~300ms/step (faster than transformer at 356ms)
- O(n) complexity

## Batch 15-17: Exploitation & Refinement

### Width scaling (Batches 14-17):
- **12L x 640d >> 16L x 512d**: Width consistently beats depth at same param count
- **12L x 704d (3.5397)**: New best, but 124M params (vs 107M for 640d models)
- At equal params (~107M), 14L x 640d matches 12L x 704d within noise

### QK-norm (Batch 17):
- Adding RMSNorm to Q and K before attention: 3.5424 (vs 3.5524 without)
- ~0.01 improvement from QK-norm alone — stabilizes attention patterns

### Token shift (Batch 16):
- RWKV-inspired: mix current token with previous via learned weight
- Helps conv layers: 3.5478 (vs ~3.55 without) — marginal but consistent

### Kitchen sink ≠ best (Batch 17):
- Combining ALL improvements (token shift + GQA + growing windows): 3.5501
- Worse than individual improvements. Possible interference between mechanisms.

### GatedLinearAttention (Batch 17):
- Failed: dimension mismatch d_head vs d_head_state in einsum
- Concept sound but needs careful implementation with matching dimensions

## Design Space Saturation

The progressive conv→attn paradigm appears to be approaching a plateau around 3.54 at 12M tokens:
- Best improvements now come from scaling width (which costs more params)
- QK-norm is the last clear structural improvement found
- Next breakthrough likely needs a fundamentally different mechanism, not more optimization

## What To Try Next

1. **Differential attention**: Two attention patterns subtracted to cancel noise
2. **Decay masking**: Soft exponential boundary instead of hard window cutoff
3. **Value residual connections**: Raw embedding fed into value projections (gradient highway)
4. **Per-token routing**: Soft continuous routing between conv and attn per token
5. **Scale test**: Run best at 24M, 48M tokens — does advantage grow or shrink?
6. **O(n) global mixing**: Replace windowed attention with linear attention that actually works
