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

### V5: Radical Alternatives (All Worse Than GatedMHConv)
- **Hash routing (5.44)**: Content-dependent bucket assignment via learned hashing. Completely fails — soft bucket membership doesn't create meaningful content-based mixing. The einsum over bucket features loses positional information.
- **Token sorting (5.13)**: Sort by learned key, conv in sorted space. Fails because sorting disrupts sequential structure and doesn't preserve causality. Position information is lost even with position bias.
- **EMA-based approaches (NaN, NaN, 7.27)**: Three variants all fail. The cumsum trick for parallel EMA (cumsum of log-decays → exp) is numerically unstable in bf16. Log of small decays → large negative numbers → cumsum → overflow/underflow. Would need fp32 or custom kernels.
- **Frequency-domain gating (4.46)**: Causal windowed band gating. Very slow (13.8K tok/s due to unfold) and band-pass filtering of sliding windows isn't expressive enough.
- **PID control (4.09)**: Proportional + integral (cumsum) + derivative (diff). The integral term (cumulative mean) is too smooth and the derivative term (diff) is too noisy. Neither captures content-dependent temporal patterns.
- **Sparse global taps (4.07)**: Fixed exponential-position taps (1,2,4,...,1024) with learned weights. Too sparse — the few tap positions don't align with relevant context positions. And slow (26K tok/s) due to multiple shifted copies.
- **Lateral inhibition (4.06)**: Competitive head suppression via softmax over head energies. Doesn't help — the competition is too rigid and heads already specialize via learned kernels.
- **Key lesson**: GatedMHConv's formula (per-head depthwise causal conv + SiLU + content-dependent sigmoid gating + output projection) is a LOCAL OPTIMUM in non-attention architecture space. No radical alternative from signal processing, neuroscience, control theory, or hash-based routing comes close. The 0.131 gap to transformer requires content-content comparison, not a better convolution variant.

### V4: GatedMHConv Exploitation Dead Ends
- **All 15 GatedMHConv variants within 0.12 of each other**: The architecture is at a local optimum. No modification of gating mechanism (sigmoid, softmax, per-channel), number of heads (8 vs 16), activation function (SiLU vs GELU), normalization (head norm), residual connections, token shift, value residual, or dual value paths breaks through ~3.92. The plateau is structural, not parametric.
- **More heads = worse**: V4_01 (16 heads, 3.970) worse than 8 heads (3.915). Smaller per-head dim = less expressive per head.
- **Softmax competitive gating**: V4_02 (3.978) — forcing heads to compete via softmax is too rigid vs independent sigmoid gates.
- **Cumsum for global context**: V4_11 (4.038) — running averages are too weak for global mixing. Confirms: cumsum ≠ attention.
- **Deep conv (2 stacked per head)**: V4_12 (3.983) — stacking convs within a head doesn't help. Receptive field already covered.
- **Cross-head mixing**: V4_06 (3.999) — mixing head outputs after gating adds overhead without benefit.
- **Kitchen sink**: V4_15 (3.938) combining token shift + value residual is WORSE than base. Mechanisms interfere.

### Batch 14-17 Failures
- **Depth-scaled residuals (1/sqrt(2i+1))**: ConvGQADepthGate got 3.6118 — over-dampens later layers. Residual scaling hurts more than it helps at this depth.
- **Removing RMSNorm (norm-free)**: ConvGQANormFree → NaN at step ~100. Normalization is non-negotiable.
- **Dual residual streams (split capacity)**: ConvAttnDualRes got 3.91 — splitting the residual stream halves effective capacity per path.
- **Kitchen sink (all ideas combined)**: 3.5501, worse than individual improvements. Combining token shift + growing windows + GQA doesn't sum — possible interference between mechanisms.
- **GatedLinearAttention (Batch 17)**: Dimension mismatch: d_head=80 (from 640/8) vs d_head_state=32. Need to match dimensions or use separate projections.
- **ProgressiveGQADeep (16L x 640d)**: 3.5990 — deeper at same width worse than shallower+wider. Confirms width > depth.
- **ConvHeavyAttnLight (12L, 9 conv + 3 attn)**: 3.6064 — too few attention layers. 50/50 split is the sweet spot.

### V6: Minimal Attention Hybrid Dead Ends
- **Low-rank attention (d_head=16)**: V6_05 = 3.927, barely better than pure conv (3.909). Q/K projections need enough capacity (d_head≥64) to form meaningful attention patterns. d_head=16 compresses too much.
- **Chunked linear attention (ELU+1 feature map)**: V6_08 = 4.651 — catastrophically bad. The inter-chunk scaling (0.1) is ad-hoc, and ELU+1 normalization is wrong. Linear attention needs careful implementation.
- **Pure transformer with conv optimizer settings**: V6_10 = 4.296. The Muon lr=0.024 / AdamW lr=0.006 settings are tuned for conv models, NOT for pure transformers. The real baseline (3.784) uses different training infrastructure.
- **PolyConv + attention**: V6_11 = 3.972, WORSE than GatedMHConv + attention (3.854). PolyConv is both slower (29K vs 35K tok/s) and less expressive. GatedMHConv is strictly superior as the conv primitive.
- **Interleaved conv/attn (confirmed again)**: V6_04 = 3.873 vs V6_03 progressive = 3.844, same 4 attn layers. Progressive (conv-first, attn-last) always wins.
- **More multi-head attention layers ≠ better**: V6_12 (6 attn, 3.846) barely better than V6_03 (4 attn, 3.844). Adding attention layers has sharply diminishing returns after 2-4.
- **8-head standard attention < 1 single-head attention**: V6_01 (8-head, 3.854) vs V6_06 (single-head, 3.768). Splitting into 8 narrow heads loses cross-feature information. With 21 conv layers having already built rich representations, one full-width pass is more effective.

### V7: SingleHead Exploitation Dead Ends
- **Attention at beginning**: V7_02 = 3.929 (worse than pure conv 3.909). Attention on raw embeddings is useless — needs conv-processed features.
- **d_qk=32**: V7_05 = 3.836. Q/K projections too small for meaningful content routing. Minimum d_qk≈64.
- **d_qk=256 vs d_qk=128**: V7_13 (3.760) ≈ V7_06 (3.758). No benefit from Q/K wider than 128.
- **ScaledRes conv + attention**: V7_07 = 3.790 vs control 3.777. V4's per-dimension scaling doesn't help when attention is present.
- **2 SingleHead at END (both at end)**: V7_14 = 3.793. Two attention layers at the same position waste capacity. Better to spread them.
- **4-head attention (d_v=128)**: V7_10 = 3.823. Still worse than 1-head (3.777). Multi-head always loses at this setup.
- **3 SingleHead layers**: V7_04 = 3.779. Three layers is slower (33K tok/s vs 35K) and not significantly better than 1 well-placed layer (3.744).
- **Kitchen sink (ScaledRes + VR)**: V7_15 = 3.719 vs VR alone V7_08 = 3.713. Combining ScaledRes with value residual slightly hurts — mechanisms may interfere.

### V8: VR Exploitation Dead Ends
- **VR alpha=0.3**: V8_03 = 3.741 — too little embedding. The attention V-projection needs substantial raw token identity (alpha≥0.5).
- **VR from mid-layer state**: V8_10 = 3.723, worse than VR from embedding (3.698). Processed intermediate representations lose the raw token identity that VR provides.
- **VR at layer 7 (too early)**: V8_11 = 3.722, worse than layer 11 (3.698). Only 7 conv layers haven't built enough features for attention to route.
- **d_qk=128 with VR**: V8_08/V8_14 ≈ V8_01. d_qk=128 adds nothing when VR is present — VR provides the information that larger Q/K capacity was compensating for.
- **3 VR layers**: V8_12 = 3.699 ≈ V8_01 = 3.698 but 5% slower. Three attention layers don't sum.
- **Learned per-dim alpha**: V8_09 = 3.704 ≈ fixed alpha=0.7. Not worth the extra parameters.

### V9: Width + Placement Dead Ends
- **d=704 14L**: V9_03/V9_04 = 3.71. Awkward middle ground — not wide enough to compensate for fewer layers.
- **d=640 18L**: V9_07/V9_08 = 3.73-3.74. Too deep — 32.5K tok/s vs 35.8K for 16L. Throughput loss > depth gain.
- **d=640 20L narrow FFN**: V9_12 = 3.720. Even deeper is even worse.
- **Conv VR on ALL layers**: V9_09 = 3.695, worse than VR on last 5 only (V9_10 = 3.678). Early layers don't benefit from embedding blending — they need to learn fresh representations.
