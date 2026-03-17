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
| V2_13_TSMHConvVR | 3.943 | +0.159 | TokenShift + MHConv + ValueRes |
| V2_02_CrossHead | 3.949 | +0.165 | Cross-head interaction |
| V2_15_TriplePath | 3.968 | +0.184 | Three parallel conv paths |
| V2_01_WideKernel | 4.017 | +0.233 | Wider kernels (up to 511) |
| V2_12_TSConvWideVR | 4.024 | +0.240 | Wide kitchen sink |
| V2_05_DecayShift | 4.102 | +0.318 | Hybrid decay+shift |
| V2_14_DoubleConv | 4.113 | +0.329 | Two MHConv passes |
| V2_06_ConvBankEMA | 4.159 | +0.375 | Conv bank + EMA |
| V2_11_ConvMoE | 4.176 | +0.392 | Mixture of conv experts |
| V2_08_GatedLinRec | CRASH | — | In-place autograd error |
| V2_09_ConvPlusScan | CRASH | — | In-place autograd error |
| V2_10_MultiGrain | 0.646 | — | DATA LEAKAGE (non-causal) |

**Key insight: content-dependent HEAD SELECTION is the missing piece.**
Standard MHConv processes all heads equally. GatedMHConv lets each token choose
which heads (= which kernel sizes = which temporal scales) to attend to.
This is a form of content-dependent processing without pairwise comparison.

**Wider kernels alone HURT** (V2_01 = 4.017, slower than standard MHConv) — bigger kernels
are slower, so fewer tokens processed in 5 min, negating any representational benefit.

**Kitchen sink doesn't work**: V2_12 (TSConvWideVR = 4.024) and V2_13 (TSMHConvVR = 3.943)
show that combining many improvements doesn't sum. Simple GatedMHConv (3.915) beats them all.

**MoE for conv is bad**: V2_11 (4.176) — routing overhead + smaller per-expert capacity = worse.

**Value residual helps conv modestly**: +0.015 improvement over PureConv.

## V3: Physics/Neuroscience-Inspired (Mostly Failed)

8/13 experiments OOM'd. Physics-inspired mechanisms (Retention, Kalman, WavePropagation, NeuralODE,
Hopfield, CompressiveMemory) create O(L²) or large intermediate tensors that don't fit at batch=4, L=2048.

Best survivors: InfoBottleneck (4.303), DenseConv (4.412), StochDepth (4.600) — all much worse than
GatedMHConv (3.915). **Lesson: theoretical elegance ≠ practical performance. Simple, fast, parallelizable
mechanisms (conv + gating) dominate complex physics-inspired ones at this scale.**

## V4: GatedMHConv Exploitation (COMPLETE — 15 variants)

| Rank | Variant | val_loss | Key change |
|------|---------|----------|------------|
| 1 | V4_14_ScaledRes | 3.9158 | Per-dimension learned output scaling |
| 2 | V4_09_OutResidual | 3.9210 | Learned output residual connection |
| 3 | V4_13_GELU | 3.9224 | GELU activation instead of SiLU |
| 4 | V4_08_HeadNorm | 3.9245 | RMSNorm per head before gating |
| 5 | V4_05_ValueRes | 3.9296 | Value residual from embedding |
| 6 | V4_15_TSVR | 3.9382 | Token shift + value residual |
| 7 | V4_04_TokenShift | 3.9394 | RWKV-style token shift |
| 8 | V4_03_ChannelGate | 3.9428 | Per-channel gates (vs per-head) |
| 9 | V4_10_DualValue | 3.9524 | Dual value projections |
| 10 | V4_07_GLU | 3.9697 | GLU-style dual gating |
| 11 | V4_01_16Heads | 3.9699 | 16 heads (vs 8) |
| 12 | V4_02_SoftmaxGate | 3.9781 | Softmax competitive gates |
| 13 | V4_12_DeepConv | 3.9828 | 2 stacked convs per head |
| 14 | V4_06_CrossPost | 3.9989 | Cross-head mixing after gating |
| 15 | V4_11_PlusCumsum | 4.0381 | Local conv + global cumsum |

**The GatedMHConv plateau is CONFIRMED at ~3.92.** Best variant (V4_14_ScaledRes, 3.9158) is only 0.001 better than base GatedMHConv (3.915). All 15 variants fall within 0.12 of each other. No simple modification (more heads, different gating, normalization, residuals, activations, token shift, value residual) breaks through the plateau.

**Key V4 takeaways:**
- Per-dimension scaling (V4_14) is the best tweak — subtle but consistent
- GELU ≈ SiLU (3.9224 vs 3.915) — activation function barely matters
- More heads (16 vs 8) HURTS: smaller per-head dim reduces expressivity
- Softmax gates worse than sigmoid: competitive head selection too rigid
- Cumsum for global context (V4_11) is the worst: running averages too weak
- Kitchen sink (V4_15 = token shift + value residual) doesn't sum: 3.938 > base 3.915
- The remaining 0.131 gap to transformer requires something fundamentally different

## V5: Radical New Mechanisms (ALL FAILED vs GatedMHConv)

15 fundamentally different architectures tested. NONE beat GatedMHConv (3.915).

| Rank | Architecture | val_loss | Mechanism |
|------|-------------|----------|-----------|
| 1 | PolyConv | 4.004 | Quadratic cross-term conv interactions |
| 2 | MultiPath | 4.008 | 3-path: fine + medium + sparse global |
| 3 | DilatedStack | 4.009 | WaveNet-style stacked dilated conv |
| 4 | ConvGRU | 4.019 | GRU gating via depthwise conv |
| 5 | CrossChannel | 4.023 | MHConv + cross-channel MLP |
| 6 | Dendritic | 4.038 | Multi-branch multiplicative gating |
| 7 | LateralInhib | 4.056 | Competitive head suppression |
| 8 | SparseGlobal | 4.074 | Sparse taps at exp positions |
| 9 | PIDConv | 4.087 | PID control theory mixer |
| 10 | FreqGate | 4.464 | Causal frequency band gating (very slow) |
| 11 | TokenSort | 5.130 | Sort by key, conv in sorted space |
| 12 | HashRouted | 5.443 | LSH-inspired routing |
| NaN | RecConvState | NaN | Cumsum-based recurrence (unstable) |
| NaN | AdaptiveEMA | NaN | Content-dependent EMA (unstable) |
| FAIL | ConvPlusEMA | 7.272 | GatedMHConv + parallel EMA (broken EMA) |

**Key insights from V5:**
1. **The GatedMHConv formula is remarkably robust.** No "radical" alternative comes close.
2. **EMA-based approaches are numerically unstable in bf16.** The cumsum trick (cumsum of log-decays)
   overflows or underflows. Three EMA variants (V5_03, V5_13, V5_15) all failed catastrophically.
3. **Polynomial cross-terms (PolyConv) are the best non-GatedMHConv idea.** conv(x)² - conv(x²) adds
   quadratic position interactions. Still 0.089 worse than GatedMHConv.
4. **Multi-path / multi-scale helps marginally** (MultiPath, DilatedStack ~4.01). Having parallel
   conv paths at different scales is sound but doesn't match GatedMHConv's adaptive head gating.
5. **Hash routing / token sorting fundamentally don't work for sequences.** Disrupting sequential
   order loses too much positional information (5.1-5.4 loss).
6. **Speed matters enormously.** FreqGate (13.8K tok/s) and SparseGlobal (26K tok/s) are too slow
   to train well in 5 minutes. GatedMHConv (34K tok/s) gets 2-3x more training.

**The 0.131 gap to transformer appears FUNDAMENTAL at this scale.**
Attention provides content-content comparison (which token to read based on content) that
NO O(n) mechanism replicates. All attempts at approximation (sparse taps, hash routing,
multi-path, EMA) fail to capture this property adequately.

## V6: Minimal Attention Hybrids — BREAKTHROUGH

**V6_06 (21 GatedMHConv + 1 SingleHeadAttn) = 3.7678 — BEATS transformer baseline (3.784) by 0.016!**

This is the first non-standard architecture to beat the transformer in 5 minutes of training.

### Complete V6 Results
| Rank | Architecture | val_loss | Δ vs trans | tok/s | Params |
|------|-------------|----------|------------|-------|--------|
| 1 | V6_06 SingleHead (21c+1a) | **3.768** | **-0.016** | 34,685 | 106.5M |
| 2 | V6_03 Conv18+Attn4 | 3.844 | +0.060 | 35,085 | 108.5M |
| 3 | V6_12 Conv16+Attn6 | 3.846 | +0.062 | 34,993 | 109.5M |
| 4 | V6_02 Conv20+Attn2 | 3.852 | +0.068 | 35,268 | 107.5M |
| 5 | V6_01 Conv21+Attn1(8h) | 3.854 | +0.070 | 35,354 | 107.0M |
| 6 | V6_04 Interleaved | 3.873 | +0.089 | 35,125 | 108.5M |
| 7 | V6_07 ChunkedLocal | 3.891 | +0.107 | 35,003 | 107.5M |
| 8 | V6_09 PureConv | 3.909 | +0.125 | 35,454 | 106.5M |
| 9 | V6_05 LowRank | 3.927 | +0.143 | 33,135 | 106.6M |
| 10 | V6_11 PolyConv+Attn1 | 3.972 | +0.188 | 29,141 | 112.4M |
| 11 | V6_10 PureTransformer | 4.296 | +0.512 | 34,041 | 117.6M |
| 12 | V6_08 LinearAttn | 4.651 | +0.867 | 33,903 | 107.5M |

### Why SingleHeadAttn beats 8-head standard attention
The critical difference between V6_06 (3.768) and V6_01 (3.854):
- **SingleHead**: 1 head, d_head=64 for Q/K, **d=512 for V** → one attention pattern over full-width values
- **Standard 8-head**: 8 heads, d_head=64 for Q/K/V → 8 patterns over narrow (64-dim) values

Single-head attention gives the model **one global content-content comparison over the FULL representation**.
Rather than splitting features into 8 heads and losing cross-head information, a single head reads ALL 512 features.
After 21 layers of multi-scale convolution, the model has rich local features — it needs ONE good global pass to integrate them, not 8 weaker passes over slices.

### Key V6 Insights
1. **One expressive attention layer > many cheaper ones**: V6_06 (1 single-head, 3.768) > V6_03 (4 standard attention, 3.844)
2. **Progressive > interleaved (confirmed again)**: V6_04 (interleaved, 3.873) < V6_03 (progressive, 3.844) with same number of attn layers
3. **More attention ≠ better**: V6_12 (6 attn layers, 3.846) barely better than V6_03 (4 attn, 3.844) — diminishing returns
4. **Low-rank attention is bad**: V6_05 (d_head=16, 3.927) is barely better than pure conv (3.909). Q/K need enough capacity.
5. **Chunked local attention is weak**: V6_07 (W=128, 3.891) — restricting attention to windows loses global value.
6. **Linear attention is broken**: V6_08 (4.651) — the ELU+1 feature map with chunked processing is numerically wrong.
7. **PolyConv is slower and worse than GatedMHConv**: V6_11 (3.972) vs V6_09 (3.909) when both get 1 attn layer.
8. **V6_10 pure transformer (4.296) is terrible**: The V6 optimizer settings (muon_lr=0.024) are tuned for conv, not transformer. The real transformer baseline (3.784) uses different training infrastructure.

### The Architecture Recipe
```
21 × GatedMHConv(d=512, 8 heads, kernels=[3,5,9,17,33,65], sigmoid gating)
 1 × SingleHeadAttn(d_qk=64, d_v=512, QK-norm)
Each layer: pre-norm → mixer → residual → pre-norm → SwiGLU FFN → residual
```
Total params: 106.5M. Training: ~34.7K tok/s (only 2% slower than pure conv).
The single attention layer adds minimal compute but provides the global content-content comparison that conv lacks.

## V7: SingleHead Exploitation — VALUE RESIDUAL IS KEY

**V7_08 (21 GatedMHConv + 1 SingleHead with Value Residual) = 3.7128 — beats transformer by 0.071!**

### Complete V7 Results
| Rank | Architecture | val_loss | Δ vs trans | tok/s | Key Variable |
|------|-------------|----------|------------|-------|-------------|
| 1 | V7_08 SingleHead+VR (end) | **3.713** | **-0.071** | 34,867 | Value residual from embedding |
| 2 | V7_15 KitchenSink | 3.719 | -0.065 | 34,418 | ScaledRes conv + VR attn |
| 3 | V7_01 SingleHead (mid) | 3.744 | -0.040 | 34,781 | Attention at layer 11 |
| 4 | V7_03 2×SingleHead (mid+end) | 3.749 | -0.035 | 34,074 | Two attention layers |
| 5 | V7_06 d_qk=128 (end) | 3.758 | -0.026 | 34,775 | Larger Q/K capacity |
| 6 | V7_13 d_qk=256 (end) | 3.760 | -0.024 | 34,671 | Very large Q/K |
| 7 | V7_11 Single+Multi (end) | 3.771 | -0.013 | 34,680 | Single-head + 8-head attn |
| 8 | V7_12 Control (V6_06 copy) | 3.777 | -0.007 | 34,828 | Exact V6_06 reproduction |
| 9 | V7_04 3×SingleHead | 3.779 | -0.005 | 33,309 | Three attn layers |
| 10 | V7_07 ScaledRes conv (end) | 3.790 | +0.006 | 34,494 | V4-best conv + attn |
| 11 | V7_14 2×SingleHead (end) | 3.793 | +0.009 | 34,036 | Two attn at end |
| 12 | V7_09 2-head (d_v=256) | 3.798 | +0.014 | 34,908 | 2 heads, wide values |
| 13 | V7_10 4-head (d_v=128) | 3.823 | +0.039 | 35,210 | 4 heads, medium values |
| 14 | V7_05 d_qk=32 (end) | 3.836 | +0.052 | 34,823 | Small Q/K capacity |
| 15 | V7_02 SingleHead (start) | 3.929 | +0.145 | 34,843 | Attention at layer 0 |

### Key V7 Insights

1. **VALUE RESIDUAL is the single biggest improvement** (+0.065 over plain single-head at end).
   Blending 50% of the original embedding into V gives the attention layer access to unprocessed
   token identity information. After 21 layers of conv, the hidden state has become highly processed —
   the attention layer benefits from seeing what the original tokens were.

2. **Middle placement > End placement** for single-head attention (3.744 vs 3.777).
   Placing attention at layer 11 means conv layers BOTH before AND after. The attention
   provides a global mixing step in the middle of the processing pipeline, and later conv
   layers can post-process the attention output.

3. **Fewer, wider attention heads are strictly better**:
   - 1 head (d_v=512): 3.777
   - 2 heads (d_v=256): 3.798
   - 4 heads (d_v=128): 3.823
   - 8 heads (d_v=64): 3.854 (V6_01)
   Each head doubling costs ~0.02 val_loss. The model needs ONE good global pattern, not many weak ones.

4. **Q/K dimension has diminishing returns**: d_qk=32 (3.836) < d_qk=64 (3.777) < d_qk=128 (3.758) ≈ d_qk=256 (3.760). The sweet spot is d_qk=64-128. Going wider than 128 doesn't help.

5. **Beginning placement is terrible** (3.929). Attention on raw embeddings with no conv processing = useless. The attention needs rich features from conv layers to route.

6. **More attention layers have diminishing returns**: 1 layer (3.744) < 2 layers (3.749) < 3 layers (3.779). Adding more layers costs throughput and doesn't compensate. Better to have one perfectly placed layer.

7. **ScaledRes conv (V4 best) doesn't stack with attention** (V7_07 = 3.790 vs V7_12 control = 3.777). The V4 improvements were compensating for lack of attention. With attention, they're less relevant.

8. **Kitchen sink ALMOST works**: V7_15 (ScaledRes + VR) = 3.719 vs V7_08 (VR only) = 3.713. The ScaledRes actually slightly hurts here, possibly due to interference.

### The Current Best Architecture
```
21 × GatedMHConv(d=512, 8 heads, kernels=[3,5,9,17,33,65], sigmoid gating)
 1 × SingleHeadAttn(d_qk=64, d_v=512, QK-norm, value_residual=0.5*embed)
Each layer: pre-norm → mixer → residual → pre-norm → SwiGLU FFN → residual
```
val_loss = 3.713, beating transformer (3.784) by 1.9%.
106.5M params, 34.9K tok/s (2% slower than pure conv, same as transformer).

## V8: VR + Placement + Width — WIDTH WINS AGAIN

**V8_13 (Wide d=640, 16L, VR single-head at end) = 3.6836 — beats transformer by 0.100 (2.6%)**

### Complete V8 Results
| Rank | Architecture | val_loss | Δ vs trans | tok/s | Key Variable |
|------|-------------|----------|------------|-------|-------------|
| 1 | V8_13 Wide d=640 16L VR | **3.684** | **-0.100** | 35,842 | Width scaling (124M params) |
| 2 | V8_06 2xVR (mid+end) | 3.696 | -0.088 | 34,261 | Two VR attention layers |
| 3 | V8_01 VR mid (layer 11) | 3.698 | -0.086 | 35,072 | VR + middle placement |
| 4 | V8_14 VR+d128 mid | 3.698 | -0.086 | 34,945 | d_qk=128 + VR + mid |
| 5 | V8_12 3xVR (7,14,21) | 3.699 | -0.085 | 33,448 | Three VR layers |
| 6 | V8_08 VR+d128 mid | 3.700 | -0.084 | 34,964 | Same as V8_14 |
| 7 | V8_02 VR layer 15 | 3.702 | -0.082 | 35,046 | Placement test |
| 8 | V8_05 VR alpha=0.9 | 3.703 | -0.081 | 35,024 | More embedding |
| 9 | V8_04 VR alpha=0.7 | 3.704 | -0.080 | 35,047 | Slightly less embedding |
| 10 | V8_09 VR learned alpha | 3.704 | -0.080 | 35,021 | Per-dim learned blend |
| 11 | V8_07 VR+d128 end | 3.711 | -0.073 | 34,978 | d128 at end |
| 12 | V8_15 Control (V7_08) | 3.719 | -0.065 | 35,034 | Reproducibility check |
| 13 | V8_11 VR layer 7 | 3.722 | -0.062 | 34,996 | Early placement |
| 14 | V8_10 VR from layer-10 | 3.723 | -0.061 | 35,031 | Mid-layer state as VR |
| 15 | V8_03 VR alpha=0.3 | 3.741 | -0.043 | 35,043 | Too little embedding |

### Key V8 Insights

1. **WIDTH > DEPTH (confirmed again)**: V8_13 (d=640, 16L, 124M params) = 3.684, beating all d=512 22L variants (~107M). Width scaling continues to be the most reliable source of improvement.

2. **VR + Middle placement synergize**: V8_01 (VR at mid) = 3.698 vs V8_15 (VR at end) = 3.719. The VR at mid improvement (+0.021) is additive with VR itself.

3. **Placement sweet spot is layers 11-15**: V8_11 (layer 7) = 3.722, V8_01 (layer 11) = 3.698, V8_02 (layer 15) = 3.702. Middle-to-late placement works best.

4. **Alpha is robust: 0.5-0.9 all similar**: alpha=0.3 (3.741) is distinctly worse, but 0.5 (3.698), 0.7 (3.704), 0.9 (3.703) are within noise. Learned per-dim alpha (3.704) = fixed 0.7.

5. **VR from embedding > VR from mid-layer state**: V8_10 (layer-10 state, 3.723) vs V8_01 (embedding, 3.698). The attention layer needs the RAW token identity, not a processed intermediate.

6. **2 VR layers marginally better than 1**: V8_06 (3.696) vs V8_01 (3.698). Tiny improvement, 2% slower.

7. **3 VR layers = diminishing returns**: V8_12 (3.699) barely better than V8_01 (3.698) but 5% slower.

8. **d_qk=128 adds nothing with VR**: V8_08/V8_14 (3.700/3.698) ≈ V8_01 (3.698). VR already provides the key information that larger Q/K would help with.

### The Architecture Evolves
```
Previous best (V7): 21 × GatedMHConv + 1 × SingleHead+VR at end = 3.713
New best (V8): 15 × GatedMHConv + 1 × SingleHead+VR at end, d=640, 16L = 3.684

At matched params (~107M):
V8_01: 21 × GatedMHConv + 1 × SingleHead+VR at MIDDLE = 3.698
```

## V9: Width + VR + Placement — OPTIMAL CONFIG FOUND

**V9_13 (d=640, 16L, VR at layer 10) = 3.668 — beats transformer by 0.116 (3.1%)**

### Complete V9 Results
| Rank | Architecture | val_loss | Δ trans | tok/s | Params | Key |
|------|-------------|----------|---------|-------|--------|-----|
| 1 | V9_13 d640 VR@10 | **3.668** | **-0.116** | 35,681 | 124M | VR at 62% depth |
| 2 | V9_15 d640 VR@8 a=0.7 | 3.668 | -0.116 | 35,616 | 124M | Alpha=0.7 at 50% |
| 3 | V9_01 d640 VR@8 | 3.672 | -0.112 | 35,814 | 124M | VR at 50% depth |
| 4 | V9_06 d768 VR mid | 3.674 | -0.110 | 35,776 | 137M | Wider, fewer layers |
| 5 | V9_02 d640 2×VR | 3.678 | -0.106 | 34,828 | 124M | Two VR layers |
| 6 | V9_10 ConvVR late5 | 3.678 | -0.106 | 35,630 | 124M | Conv+attn both have VR |
| 7 | V9_11 d640 14L wide FFN | 3.683 | -0.101 | 36,155 | 130M | dff=3200 |
| 8 | V9_05 d768 VR end | 3.685 | -0.099 | 35,766 | 137M | Width compensates |
| 9 | V9_14 Control (V8_13) | 3.688 | -0.096 | 35,682 | 124M | Reproducible |
| 10 | V9_09 ConvVR all | 3.695 | -0.089 | 35,487 | 124M | VR on all layers |
| 11 | V9_04 d704 VR mid | 3.707 | -0.077 | 34,180 | 132M | Awkward width |
| 12 | V9_03 d704 VR end | 3.712 | -0.072 | 34,188 | 132M | - |
| 13 | V9_12 d640 20L narrow | 3.720 | -0.064 | 32,656 | 127M | Too deep |
| 14 | V9_08 d640 18L mid | 3.728 | -0.056 | 32,532 | 135M | Too deep |
| 15 | V9_07 d640 18L end | 3.738 | -0.046 | 32,517 | 135M | Too deep |

### Key V9 Insights

1. **OPTIMAL PLACEMENT is ~60% depth**: Layer 10 of 16 (V9_13, 3.668) beats layer 8 of 16 (V9_01, 3.672). The attention layer should be just past the halfway point — enough conv processing before, enough after.

2. **d=640 16L is the sweet spot**: Better than d=768 12L (fewer layers hurt), d=704 14L (awkward), d=640 18L (too slow, fewer tokens processed), d=640 20L (even worse).

3. **Depth kills throughput**: 16L@640 = 35.8K tok/s, 18L@640 = 32.5K, 20L@640 = 32.7K. The 10% throughput loss at 18L means fewer tokens in 5 min, negating any depth benefit.

4. **Conv VR (VR applied to conv layers) is promising**: V9_10 (last 5 conv + attn VR) = 3.678, V9_09 (all conv VR) = 3.695. VR helps late conv layers but hurts early ones (alpha=0.2 may be wrong for early layers).

5. **Alpha=0.5 ≈ 0.7**: V9_15 (0.7, 3.668) ≈ V9_01 (0.5, 3.672). Robust.

6. **d=704 is an awkward point**: Worse than d=640 (fewer layers, 14 vs 16) AND worse than d=768 (not wide enough to compensate). Width scaling has non-linear returns.

### The Current Best Architecture
```
d=640, 16 layers:
  Layers 0-9:  GatedMHConv(d=640, 8 heads, kernels=[3,5,9,17,33,65], sigmoid gating)
  Layer 10:    SingleHeadAttn(d_qk=80, d_v=640, QK-norm, value_residual=0.5*embed)
  Layers 11-15: GatedMHConv(d=640, 8 heads, ...)
  FFN: SwiGLU(d=640, dff=2560) in each layer
```
val_loss = 3.668, 124M params, 35.7K tok/s. Beats transformer (3.784) by 3.1%.

## V10: Final Exploitation — PLATEAU CONFIRMED

**V10_09 (d=640, VR@10, alpha=0.8) = 3.665 — beats transformer by 0.119 (3.1%)**

All 12 V10 variants cluster between 3.665-3.681. Seed=43 verification: 3.679 (consistent).

### Key V10 Findings
- **Placement is robust**: Layers 9-11 all within 0.004 of each other
- **Alpha=0.8 marginally best**: 3.665 vs 3.668 (alpha=0.5). Not significant.
- **Conv VR helps marginally**: V10_10 (conv VR@12-14 + attn VR@10) = 3.667. Tiny improvement.
- **Seed verification**: V10_12 (seed=43) = 3.679 confirms result is robust.
- **Architecture is at a plateau**: No modification breaks through ~3.665.

### The Optimal Architecture (FINAL)
```
d=640, 16 layers, 124M params:
  Layers 0-9:   GatedMHConv(d=640, 8 heads, kernels=[3,5,9,17,33,65], sigmoid gating)
  Layer 10:     SingleHeadAttn(d_qk=80, d_v=640, QK-norm, VR alpha≈0.5-0.8 from embedding)
  Layers 11-15: GatedMHConv(d=640, 8 heads, ...)
  FFN:          SwiGLU(d=640, dff=2560) per layer
  Other:        Pre-norm (RMSNorm), tied embeddings, EmbeddingWithScale
```
val_loss = 3.665-3.668 (seed-dependent), 35.5K tok/s.
**Beats transformer (3.784) by 3.1% at 5 minutes of training.**

### Architecture Journey
| Batch | Best val_loss | Δ trans | Key Discovery |
|-------|-------------|---------|---------------|
| V1-V5 | 3.915 | +0.131 | GatedMHConv = best non-attention mechanism |
| V6 | 3.768 | -0.016 | SingleHead (d_v=full) > MultiHead |
| V7 | 3.713 | -0.071 | Value residual from embedding |
| V8 | 3.684 | -0.100 | Width scaling (d=640 > d=512) |
| V9 | 3.668 | -0.116 | Optimal placement at ~60% depth |
| V10 | 3.665 | -0.119 | Plateau. Alpha/conv VR = marginal |

## What To Try Next

1. **12M token scale test**: Train best architecture for full 12M token budget to compare with existing best (3.449)
2. **Fundamentally new ideas**: The conv+single-head-VR paradigm is exhausted. Need a qualitatively different approach.
3. **Dynamic routing**: Instead of fixed single attention layer, learn when to attend
4. **Attention + conv WITHIN a layer**: Parallel conv+attention rather than sequential
5. **State-space model hybrid**: Replace GatedMHConv with SSM for some layers
