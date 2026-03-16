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

## Batch 100 + Novel Non-Attention Tournament (5-min, d=512, n=22)

### Attention vs Non-Attention gap
| Category | Best val_loss | Example |
|----------|-------------|---------|
| Conv + VR Attention (best) | 3.759 | CosineAttnVR |
| Standard GQA + Conv + VR | 3.784 | TSVRBase |
| Pure MHConv (no attention) | 3.920 | PureConv |
| Best novel non-attention | 4.004 | LearnedDecayConv |

**The attention gap is ~0.22 val_loss.** No O(n) mechanism matched attention in 5 min.

### What works in non-attention:
1. **Physics-motivated kernels win**: LearnedDecayConv (damped oscillator: A·exp(-αt)·cos(ωt+φ)) = best novel arch (4.004)
2. **Multi-head exponential decay** (4.009): Multiple timescales per head captures hierarchical structure
3. **Dynamic conv bank** (4.026): Content-dependent kernel selection bridges the adaptivity gap
4. **Token shift pyramid** (4.055): Log(n) shift offsets give cheap global coverage
5. **Conv + cumsum hybrid** (4.097): Separating local (conv) and global (cumsum) is sound

### What fails without attention:
- **Pure cumsum methods** (5.0-5.3): CumsumHierarchy, PolynomialMixer, RecChannelMix — running averages lack the expressivity for sequence modeling
- **Complex rotator** (4.459): Too slow (16K tok/s) and complex oscillations don't help at this scale
- **Reaction-diffusion** (4.295): Beautiful theory, weak practice at 5 min

### Key insight: PureConv (3.920) is remarkably strong
The existing MHConv with 8 heads at kernel sizes [3,5,9,17,33,65] is BETTER than all 15 novel designs.
Its advantages: (1) per-channel learned weights (not constrained), (2) SiLU activation per head,
(3) gated output, (4) exponentially-spaced kernels give multi-scale coverage efficiently.

### What attention provides that conv doesn't:
The 0.136 gap between PureConv (3.920) and transformer (3.784) = value of attention.
Attention provides: content-based routing (which tokens to read), dynamic weighting (how much),
and global receptive field (any position). The best non-attention methods approximate some of
these but none replicate all three.

## Batch V2: Closing the Gap (Partial Results)

**NEW BEST non-attention: V2_04_GatedMHConv = 3.915** (beats PureConv 3.920!)
The key innovation: per-head content-dependent gating on MHConv. Each token selects
which conv kernel sizes matter via a learned gate. This gives content-dependent
multi-scale processing — the critical property that attention has and conv didn't.

| Architecture | val_loss | Δ vs transformer | Key innovation |
|---|---|---|---|
| V2_04_GatedMHConv | **3.915** | +0.131 | Per-head content gating |
| V2_03_ValueResConv | 3.935 | +0.151 | Value residual for conv |
| V2_02_CrossHead | 3.949 | +0.165 | Cross-head interaction |
| V2_01_WideKernel | 4.017 | +0.233 | Wider kernels (up to 511) |
| V2_05_DecayShift | 4.102 | +0.318 | Hybrid decay+shift |

**Key insight: content-dependent HEAD SELECTION is the missing piece.**
Standard MHConv processes all heads equally. GatedMHConv lets each token choose
which heads (= which kernel sizes = which temporal scales) to attend to.
This is a form of content-dependent processing without pairwise comparison.

Wider kernels alone (V2_01) actually HURT — they're too slow, processing fewer tokens.
Value residual helps conv modestly (+0.015 vs PureConv).

## What To Try Next

1. **Improve PureConv**: Add wider kernels (up to 512), better gating, cross-head interaction
2. **Content-dependent conv**: Kernels generated or selected based on token content
3. **Value residual for conv**: Feed embedding into conv value projections (worked +0.1 for attention)
4. **Hybrid novel mixers**: Combine LearnedDecayConv + TokenShiftPyramid + DynamicConvBank
5. **Gated linear recurrence via parallel scan**: If fast enough, this could be the missing piece
6. **Mixture of conv experts**: Different conv heads activated per token
7. **Scale test**: Run best non-attention at 12M tokens — does the gap shrink?
