# Frontier Architecture Hypotheses

Distilled ideas for beyond-transformer architecture research.

---

## High-Confidence Hypotheses

### H1: Hybrid SSM-Attention will beat pure transformers at 88M scale
**Basis**: Jamba (AI21), Mamba-2, Griffin (Google) all show hybrids outperforming pure architectures at larger scales. At 88M, the SSM layers can handle local context cheaply while a few attention layers handle retrieval.
**Test**: hybrid_ssm_attn_alt, hybrid_progressive, hybrid_ssm_heavy

### H2: Gated Linear Attention is the sweet spot between linear attention and transformers
**Basis**: GLA adds a learned decay gate to linear attention, recovering much of softmax attention's expressiveness. The gate allows input-dependent forgetting — critical for language modeling.
**Test**: gla_baseline, hybrid_gla_attn

### H3: Differential attention will improve signal-to-noise ratio
**Basis**: Microsoft's DiffTransformer showed consistent improvements. The dual-softmax-with-subtraction mechanism cancels out noise/common attention patterns and amplifies task-relevant patterns.
**Test**: diffattn_baseline

---

## Medium-Confidence Hypotheses

### H4: RetNet's multi-scale decay provides a natural hierarchy of temporal attention
**Basis**: Different heads attending at different time scales (from 3-token to 500-token effective range) may be more parameter-efficient than uniform attention.
**Test**: retnet_baseline

### H5: RWKV's token shift is a powerful yet underappreciated mechanism
**Basis**: Simple shift-and-mix of adjacent tokens provides local context without convolution cost. Could transfer to other architectures.
**Test**: rwkv_baseline, then cross-pollinate token shift to SSMs and hybrid models

### H6: Evolving state machines with full-rank state matrices may exceed SSM capacity
**Basis**: SSMs maintain a vector state; an evolving d×d matrix state provides d× more capacity per token. The question is whether we can efficiently update and read from this state.
**Test**: evolving_state_baseline

---

## Speculative Hypotheses

### H7: Frequency-domain mixing captures periodic patterns that attention misses
**Basis**: FNet showed that pure FFT mixing gets 70-90% of BERT quality. With *learned* spectral filters, we might close the gap further.
**Test**: freqmix_baseline

### H8: Polynomial attention may find a better accuracy-compute Pareto front
**Basis**: Quadratic kernel (1+QK^T)^2 captures second-order feature interactions. Between linear attention (degree 1) and softmax (effectively infinite degree), degree 2-3 might be the sweet spot.
**Test**: polyattn_deg2_baseline, then polyattn_deg3

### H9: Multi-resolution convolutions may be the simplest effective non-attention mixer
**Basis**: Parallel convolutions at 3/7/15/31 kernel sizes cover the most important context windows. No attention mechanism needed.
**Test**: multires_baseline

### H10: The winning architecture might be a novel combination we haven't thought of yet
**Basis**: History shows breakthroughs come from unexpected places. The research system should maintain 20% capacity for truly novel ideas.
**Test**: Ongoing exploration, evolving with discoveries

---

## Cross-Pollination Ideas (to test after baselines)

1. **SinGLU FFN in SSM blocks**: SinGLU was a big transformer winner (+0.97%). Test if sin-gated FFN also improves SSM and RWKV blocks.
2. **Token shift in transformer**: RWKV's shift mechanism is cheap. Add it to transformer attention as an additional local context signal.
3. **Multi-scale decay in GLA**: RetNet's per-head decay rates in GLA's gated attention — multiple forgetting time constants.
4. **Mamba's selective gating in convolution models**: Make Hyena's convolution kernels input-dependent, like Mamba makes SSM parameters input-dependent.
5. **Warm momentum for non-transformer architectures**: The Muon warm momentum trick might help SSMs and linear attention too.
