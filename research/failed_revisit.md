# Failed Experiments — Revisit Candidates

Experiments that failed but may work under different conditions. The AI checks this file before each new batch to see if retry conditions are now met.

---

## Catastrophic Failures (Training Divergence)

### post_norm
- **Val loss**: 7.6537 (+60.3% regression vs 4.7733 baseline)
- **Tested at**: 10M tokens
- **Failure hypothesis**: Post-LN destabilizes training at this scale without careful initialization. Known issue from original BERT/GPT-1 — requires specific warmup.
- **Retry condition**: After implementing proper learning rate warmup (≥5% warmup ratio) + smaller init scale. OR if depth-scaled initialization is confirmed working.
- **Retry status**: Not yet

### layer_norm_post / swiglu_post_norm
- **Val loss**: 7.66 (+60.5% regression)
- **Tested at**: 10M tokens
- **Failure hypothesis**: Same as post_norm. Post-normalization is extremely sensitive to initialization at small scale.
- **Retry condition**: Same as post_norm — needs warmup + careful init. Consider trying with `init_scheme: depth_scaled` first.
- **Retry status**: Not yet

### gpt2_style
- **Val loss**: 7.6599 (+60.5% regression)
- **Tested at**: 10M tokens
- **Failure hypothesis**: GPT-2 style combines post-LN + specific init that we didn't replicate fully. GPT-2's actual training used a scaled init (`0.02 / sqrt(2*n_layers)`).
- **Retry condition**: After implementing the correct GPT-2 init scheme (residual projection scaled by `1/sqrt(2*n_layers)`). This is a different question from just "post_norm".
- **Retry status**: Not yet

### no_final_norm
- **Val loss**: 12.47 (+161% regression)
- **Tested at**: 10M tokens
- **Failure hypothesis**: Final normalization is load-bearing for output scale. Removing it causes logit explosion.
- **Retry condition**: Only retry if experimenting with output scale clamping (e.g., softcapping). Even then, low priority.
- **Retry status**: Low priority

### no_norm
- **Val loss**: NaN (divergence)
- **Tested at**: 10M tokens
- **Failure hypothesis**: No normalization at all causes immediate training instability.
- **Retry condition**: Only if implementing a normalization-free architecture (e.g., NFNet-style with careful weight standardization). Different research direction entirely.
- **Retry status**: Different research track, not current priority

---

## Moderate Failures (Meaningful Regression)

### no_qk_norm
- **Val loss**: 4.8660 (+1.94% regression)
- **Tested at**: 10M tokens (pre-norm baseline)
- **Failure hypothesis**: QK-norm is critical for training stability in this architecture. Removing it causes attention score explosion.
- **Retry condition**: Already part of the new g2 baseline. The `attn_no_qk_norm` experiment at 6M confirms this remains a regression.
- **Retry status**: Confirmed important — do not remove

### no_embed_scale
- **Val loss**: 4.9173 (+3.02% regression)
- **Tested at**: 10M tokens
- **Failure hypothesis**: Embed scale (`sqrt(d_model)`) is needed to keep token embeddings in the right range relative to positional encodings.
- **Retry condition**: Only if switching to a completely different embedding scheme (e.g., learned positional embeddings that absorb the scale).
- **Retry status**: Unlikely to retry — confirmed load-bearing

### polar_express_2 (muon_ns_steps=2)
- **Val loss**: 4.9891 (+4.52% regression)
- **Tested at**: 10M tokens
- **Failure hypothesis**: Fewer Nesterov-Schulz steps produces worse orthogonalization of Muon gradients, hurting convergence.
- **Retry condition**: Confirmed — ns_steps=5 is optimal. Retry only at ns_steps=8 or 10 to see if more steps help.
- **Retry status**: `muon_ns_10` tested — neutral result. This direction closed.

### no_weight_tying
- **Val loss**: 5.0484 (+5.76% regression)
- **Tested at**: 10M tokens
- **Failure hypothesis**: Without weight tying, lm_head has independent parameters that the optimizer struggles to train well at this token budget.
- **Retry condition**: At 100M+ tokens where the model has enough data to train independent embeddings/LM head.
- **Retry status**: Not yet — scale-dependent

### dropout_01 (dropout=0.1)
- **Val loss**: 4.8535 (+1.68% regression at 10M), 5.1426 at 6M
- **Tested at**: 10M tokens
- **Failure hypothesis**: At 10M tokens this model is underfitting, not overfitting. Dropout actively hurts by reducing effective capacity during training.
- **Retry condition**: At 100M+ tokens where the model sees more data and overfitting becomes a real concern. Consider stochastic depth (softer regularization) instead.
- **Retry status**: `g2_sdrop_*` experiments queued as gentler alternative

### swiglu_swiglu (stacked SwiGLU)
- **Val loss**: 4.9342 (+3.37% regression)
- **Tested at**: 10M tokens
- **Failure hypothesis**: Two sequential SwiGLU sub-layers per block doubles the FFN depth but not the width — creates optimization difficulties without enough capacity gain.
- **Retry condition**: After depth-scaled initialization is confirmed working. Also worth trying with residual connections between the sub-layers.
- **Retry status**: Not yet

### swiglu_deep (2 reduced sequential sub-layers)
- **Val loss**: 4.8475 (+1.55% regression)
- **Tested at**: 10M tokens
- **Failure hypothesis**: Same as swiglu_swiglu — sequential stacking without proper width compensation.
- **Retry condition**: Try with `residual_scale: 0.5` between sub-layers, or with depth-scaled init.
- **Retry status**: Not yet

### glu_ffn
- **Val loss**: 4.8479 (+1.56% regression)
- **Tested at**: 10M tokens
- **Failure hypothesis**: Standard GLU (sigmoid gate) underperforms SiLU/GELU gating. Gate saturation issue.
- **Retry condition**: Not worth retrying — better gating functions clearly superior.
- **Retry status**: Closed

### muon_no_momentum (momentum=0.0)
- **Val loss**: 4.8951 (+2.55% regression)
- **Tested at**: 10M tokens
- **Failure hypothesis**: Muon optimizer loses most of its benefit without momentum — momentum is what makes the Nesterov-Schulz iteration effective.
- **Retry condition**: None — this is a fundamental property of the optimizer.
- **Retry status**: Closed

---

## Soft Failures (Neutral but Suspicious)

### scispace_sigmoidglu / scispace_pregatelu / scispace_prenormdown
- **Val loss**: ~4.86-4.87 (1.7-2.1% regression)
- **Tested at**: 10M tokens (pre-norm baseline)
- **Failure hypothesis**: Sigmoid gating saturates, preGate/prenorm variants introduce redundant normalization that conflicts with existing QK-norm.
- **Retry condition**: Test on g2 baseline (sandwich norm) — the interaction with internal norms may be different.
- **Retry status**: Queued for g2 baseline testing

---

*Last updated: 2026-03-13*
*Total failed: 18 | Closed (won't retry): 5 | Retry conditions not yet met: 10 | Queued for retry: 3*
