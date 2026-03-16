# Dead Ends — What Didn't Work and Why

This file prevents the research loop from repeating failed approaches. Every entry must explain **WHY** it failed, not just that it did.

---

## From Ablation System (Transferred Knowledge)

These mechanisms were tested within the transformer and found to be losers. They may or may not transfer to other architecture families.

### Catastrophic Losers (>2% worse)
- **Muon cautious**: -2.6% to -3.2%. Conservative updates hurt at small scale where aggressive exploration of loss landscape is needed.
- **Muon update_clip**: -12% to -13%. Clipping Muon updates destroys the orthogonalized gradient direction.
- **Muon frob_scale**: -19%. Frobenius scaling collapses effective learning rate.
- **Post-norm architecture**: Catastrophic. Gradient flow through residual stream is critical; post-norm blocks it.
- **No QK norm**: Catastrophic. At 88M scale, attention logits explode without normalization.
- **Tied weights=False**: Catastrophic. Untying wastes parameters at 88M scale.
- **Parallel block (PaLM-style)**: -0.5% to -1%. Mixing attention and FFN outputs adds noise at this scale.

### Consistent Losers (0.2% to 2% worse)
- **Gated residual on row+rms baseline**: -0.28% to -0.82%. Conflicts with Muon gradient normalization.
- **Value norm**: Normalizing value vectors removes useful magnitude information.
- **Depth-scaled init**: Over-dampens initial gradient flow at 22 layers.
- **GPT-2 init**: Output projection scaling too aggressive for 22 layers.

---

## From Frontier Experiments

### Fundamentally Non-Causal (cannot be fixed)
- **FFT/Spectral mixing**: FFT is inherently bidirectional. Even with causal conv post-processing, the FFT step sees the future. SpectralGateV2 got 0.02 val_loss — pure data leakage.
- **Haar wavelet decomposition**: Even/odd split lets position 0 see position 1. Any wavelet that splits into even/odd subsequences is non-causal.
- **Downsample→upsample pooling**: Downsampling pools adjacent tokens (future leak), upsampling via repeat_interleave spreads information backwards.

### Numerically Unstable
- **GravityMixer**: `exp(log_decay * position)` overflows at position 2048. Fundamental issue with cumsum+exp approach for position-dependent decay.
- **EMAMixer**: Same exp(position) overflow. Any architecture needing exp of unbounded position values is doomed.
- **GatedConvResidual**: Dense skip connections between layers destabilized training — loss initially dropped then exploded upward at ~400 steps.

### Too Slow / OOM
- **CausalLinearAttention**: kv cumsum creates (B, H, L, d_head, d_head) tensor — OOM at 2048 seq_len. Would need chunking or custom kernel.
- **GatedRetention**: Sequential per-chunk loop with state accumulation. 12x slower than transformer. Needs CUDA kernel.
- **CellularAutomata**: Even after depthwise conv fix, marginal and slow (730ms → fixed to ~400ms but still poor val_loss).

### Architecture Patterns That Don't Work
- **Pure conv at 12M tokens**: MHConvPool gets 3.71 at 12M — ties transformer. Conv is data-efficient at 6M but plateaus. Needs attention for continued improvement.
- **Interleaved conv/attn < Progressive conv→attn**: Alternating every layer (3.62) is worse than conv-first then attn-last (3.59). The model benefits from local processing first, global processing last.
- **ConvCascade (sequential cascading)**: val_loss 4.64 — sequential composition within a layer loses to parallel multi-width approach.
- **Cross-head MLP mixing**: Marginal — adds overhead without benefit vs baseline conv (3.73 vs 3.70 transformer).
- **Global gate via causal running average**: Doesn't help scaling (3.74 vs 3.70 transformer).

### Batch 13 Failures
- **SlidingStateAttention**: 3905ms (11x too slow). Python loops over chunks with per-element causal masking. Would need a fused kernel.
- **Selective scan (Mamba-like)**: OOM. Sequential scan with (B, L, D, N) intermediates where N=16 state dim. d_state=16 × d_model=512 creates too much intermediate memory at L=2048.
- **RecurrentGateConv (GRU on conv features)**: OOM during backward. GRU gates require 3x d_model linear projections with d_model*2 input → too memory-heavy for 512d.
- **MultiResConv (pyramid downsample→upsample)**: NON-CAUSAL (val_loss 1.51 = data leakage). Strided conv downsample + repeat_interleave upsample still leaks future through the stride grouping. Same fundamental issue as HierPoolMixer.
- **Deep narrow (24L x 384d = 73M params)**: 3.6192 — worse than 88M models. Under-parameterized at this scale.

### V2 Failures
- **V2_08_GatedLinRecurrence** (CRASH): In-place tensor modification `h[:, t] = ...` in sequential loop incompatible with autograd. Sequential scan patterns need `torch.cat` or list-based accumulation, not in-place assignment.
- **V2_09_ConvPlusScan** (CRASH): Same in-place modification issue as V2_08.
- **V2_10_MultiGrainConv** (val_loss=0.646 = DATA LEAKAGE): Strided downsampling + repeat_interleave upsample is non-causal. Same issue as MultiResConv — strided views leak future info. ANY architecture that changes temporal resolution is suspect.

### Novel Non-Attention Failures (Batch NoAttn-1)
- **PolynomialMixer (5.056)**: Quadratic interactions via element-wise product + cumsum. Cumsum of products is not expressive enough — just captures running second moments without content-based routing.
- **RecurrentChannelMix (5.110)**: Separated temporal (cumsum) + channel (MLP) mixing. Too weak — the temporal aggregation is just a running mean, and position-dependent channel MLP can't compensate.
- **CumsumHierarchy (5.313)**: Multi-level cumsums (mean of means). Running averages of running averages just gives increasingly smoothed signals — no sharp feature extraction.
- **ComplexRotator (4.459, 16K tok/s)**: Damped oscillatory states. Two problems: (1) very slow due to per-head conv with complex rotation, (2) periodic patterns in text are rare at token level.
- **OscillatoryRecurrence (8.401, 727 tok/s)**: Evolution of ComplexRotator using input-dependent frequencies. Catastrophic slowdown (40x slower than transformer) and failed to converge in 5-min window. Complex-valued rotations are too compute-heavy for current GPU kernels without custom implementation.
- **ReactionDiffusion (4.295)**: Activator-inhibitor system. Cool theory but diffusion-based mixing is too slow to propagate information. 3 steps of k=5 conv = receptive field of 15, which is too narrow.
- **CausalDiffusion (4.220)**: Unrolled heat equation. Same issue — iterative nearest-neighbor mixing doesn't reach far enough. Would need many more steps (expensive).

### V3 Failures (Physics/Neuroscience-Inspired)
- **RetentionMixer, WavePropagation, CompressiveMemory, NeuralODE, TopKMemConv, Hopfield, GatedPoolConv, ConvResNet** (ALL OOM): Physics-inspired mechanisms that create O(L²) or large intermediate tensors crash at batch_size=4, L=2048. Retention needs O(L²) decay matrix; others have excessive per-layer state.
- **InfoBottleneckMixer (4.303)**: Information bottleneck via projection to small dim + back. Too lossy — 18K tok/s but the bottleneck destroys information that can't be recovered.
- **DenseConvNet (4.412)**: Dense connections between conv layers (DenseNet-style). Too many skip connections add noise at 22 layers, plus slow (18K tok/s).
- **StochDepthConv (4.600)**: Stochastic depth (random layer dropping during training). Too aggressive at 22 layers — dropping layers randomly destroys learned representations.
- **AdaptiveSpanConv (5.071)**: Learned adaptive span per head. Extremely slow (8K tok/s) — the adaptive masking is too expensive.
- **KalmanFilter (5.309)**: Kalman filter state estimation. State transition model is too simple (linear) and slow (29K tok/s but terrible loss).
- **Key lesson**: Physics-inspired approaches at 512d/22L mostly OOM or are too slow. The mechanisms that work (conv, gating) are simple and parallelizable.

### Batch 14-17 Failures
- **Depth-scaled residuals (1/sqrt(2i+1))**: ConvGQADepthGate got 3.6118 — over-dampens later layers. Residual scaling hurts more than it helps at this depth.
- **Removing RMSNorm (norm-free)**: ConvGQANormFree → NaN at step ~100. Normalization is non-negotiable.
- **Dual residual streams (split capacity)**: ConvAttnDualRes got 3.91 — splitting the residual stream halves effective capacity per path.
- **Kitchen sink (all ideas combined)**: 3.5501, worse than individual improvements. Combining token shift + growing windows + GQA doesn't sum — possible interference between mechanisms.
- **GatedLinearAttention (Batch 17)**: Dimension mismatch: d_head=80 (from 640/8) vs d_head_state=32. Need to match dimensions or use separate projections.
- **ProgressiveGQADeep (16L x 640d)**: 3.5990 — deeper at same width worse than shallower+wider. Confirms width > depth.
- **ConvHeavyAttnLight (12L, 9 conv + 3 attn)**: 3.6064 — too few attention layers. 50/50 split is the sweet spot.
