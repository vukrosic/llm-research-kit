# Latent Predictive Language Modeling (LPLM)

## Beyond Next-Token Prediction: Learning to Think in Abstractions

---

## 1. The Problem with Next-Token Prediction

Current language models are trained to predict the next token — a single subword unit chosen from a ~50k vocabulary. This objective has a fundamental flaw: **it allocates equal capacity to predicting "the" as to predicting the key insight in a reasoning chain.**

Consider the sentence: *"The capital of France is Paris."*

A next-token model must predict every token sequentially: `The → capital → of → France → is → Paris → .`

But the actual *information* here is a single fact: `capital(France) = Paris`. The tokens "The", "of", "is", "." are syntactic scaffolding — high-entropy noise from a semantic standpoint. Current models waste enormous capacity modeling the distribution over these filler tokens.

**Worse:** next-token prediction is fundamentally local. The model can never "skip ahead" to plan what it wants to say. It generates left-to-right, one token at a time, with no explicit mechanism for abstract thought before committing to surface forms.

### What humans actually do

Humans don't think one word at a time. We:
1. Form an **abstract intention** (a meaning, a concept, a plan)
2. Chunk that intention into **phrase-level units**
3. Only at the last moment do we serialize into words

We propose a model that mirrors this: **predict the next abstract representation, not the next token.**

---

## 2. Core Idea: Latent Predictive Language Modeling

LPLM replaces next-token prediction with **next-representation prediction in a learned latent space**.

```
Traditional LM:    tokens → [model] → next token distribution
LPLM:              tokens → [encoder] → latent representations → [predictor] → next latent → [decoder] → tokens (optional)
```

### The three components

| Component | Input | Output | Role |
|-----------|-------|--------|------|
| **Chunk Encoder** (E) | Variable-length token span | Fixed-size latent vector z | Compress tokens into semantic units |
| **Latent Predictor** (P) | Sequence of latent vectors z₁...zₜ | Predicted next latent ẑₜ₊₁ | Predict next abstraction |
| **Token Decoder** (D) | Latent vector z | Token sequence | Unpack latent back to surface form |

The key insight: **the predictor operates entirely in latent space.** It never sees tokens. It thinks in abstractions.

---

## 3. Architecture Detail

### 3.1 Chunk Encoder: Learning to Segment and Compress

The encoder must solve two problems simultaneously:
1. **Where to segment** — which token spans form a semantic unit?
2. **How to compress** — what fixed-size vector captures the meaning of that span?

#### Adaptive Segmentation via Boundary Prediction

A lightweight boundary network (1-2 transformer layers) processes the raw token embeddings and outputs a boundary score bᵢ ∈ [0,1] for each token position:

```
b = σ(W_b · TransformerBlock(token_embeddings))
```

During training, we use a **soft segmentation** — each token belongs to adjacent chunks with weight proportional to the boundary scores. During inference, we threshold to get hard boundaries.

The model learns to place boundaries at semantically meaningful points — roughly phrase or clause boundaries, but driven entirely by what makes prediction easiest (not by linguistic rules).

**Target chunk size:** 4-16 tokens on average. This is a 4-16x compression of the sequence, meaning the predictor processes 4-16x fewer "tokens" (now latent vectors) for the same context length. A 8k-token context becomes ~500-2000 latent vectors.

#### Compression via Cross-Attention Pooling

Within each chunk, we use a learned query vector q to cross-attend over the chunk's token representations:

```
z = CrossAttention(q, chunk_tokens, chunk_tokens)    # shape: [d_latent]
```

This produces a single latent vector z per chunk. The query q is learned and shared across all chunks (with positional modulation).

**Alternative: Information Bottleneck Pooling.** Instead of cross-attention, pass chunk tokens through a small encoder and take the output of a [CLS]-like token, with an explicit information bottleneck (β-VAE style KL penalty) to force compression:

```
z = μ + σ·ε,    KL(q(z|chunk) || p(z)) < β
```

This forces the encoder to discard unpredictable details and retain only the semantically essential content.

### 3.2 Latent Predictor: Thinking in Abstractions

The predictor is the core of the model. It takes a sequence of latent vectors and predicts the next one:

```
ẑₜ₊₁ = Predictor(z₁, z₂, ..., zₜ)
```

**This is where the architecture innovation matters most.** Since the predictor operates on compressed semantic units rather than tokens, it has fundamentally different requirements:

#### Why Attention Might Not Be Optimal Here

Attention was designed for token-level sequence modeling. In latent space:
- Sequences are 4-16x shorter → O(n²) is less of a problem, but...
- Each latent vector is **information-dense** — it represents a whole phrase, not a subword
- The relationships between latents are more **structured** — they represent semantic dependencies, not syntactic ones
- We need **compositional** operations — combining meanings, not copying tokens

#### Proposed: Relational State Network (RSN)

We propose a novel sequence mixer for the predictor, designed for information-dense latent sequences:

**State evolution.** The predictor maintains a state matrix S ∈ ℝ^{d×d} that evolves as each latent is processed:

```
Sₜ = f(Sₜ₋₁, zₜ)
```

where f is a learned state transition that performs a **low-rank update**:

```
Sₜ = (1 - gₜ) · Sₜ₋₁ + gₜ · (Aₜ · Sₜ₋₁ · Bₜ + Cₜ · zₜ · Dₜᵀ)
```

- gₜ ∈ [0,1] is an input-dependent gate (how much to update state)
- Aₜ, Bₜ are low-rank rotation matrices derived from zₜ (rotate existing knowledge)
- Cₜ, Dₜ are low-rank projection matrices (integrate new information)

This is inspired by:
- **Kalman filters** (state estimation under uncertainty)
- **Fast weights** (using activations to modulate a weight matrix)
- **Mamba's selective state spaces** (input-dependent state transitions)

But crucially different: the state S is a full **relational matrix** that captures pairwise interactions between latent dimensions, not just a state vector. This gives it O(d²) capacity per step — matching the information density of the latent inputs.

**Readout.** To predict the next latent:

```
ẑₜ₊₁ = ReadOut(Sₜ, zₜ) = LayerNorm(Wout · flatten(Sₜ) + MLP(zₜ))
```

#### Multi-Scale Predictor Stack

We stack predictors at multiple granularities:

```
Level 0 (phrase):     z₁, z₂, z₃, ...        → predict next phrase latent
Level 1 (sentence):   pool(z₁..z₄) = s₁, ... → predict next sentence latent
Level 2 (paragraph):  pool(s₁..s₃) = p₁, ... → predict next paragraph latent
```

Higher levels predict coarser representations further into the future. Level 2 might predict the gist of the next paragraph before Level 0 has processed any of its phrases.

**Top-down guidance:** Higher-level predictions condition lower-level predictions:

```
ẑₜ₊₁⁽⁰⁾ = Predictor⁰(z₁..zₜ, ŝ⁽¹⁾, p̂⁽²⁾)
```

This gives the model a form of **planning** — it knows where the discourse is going before it predicts the next phrase.

### 3.3 Token Decoder: Unpacking Latents to Surface Form

The decoder converts a latent vector back to tokens when needed (e.g., for text generation):

```
tokens = Decoder(z)
```

This is a small autoregressive model (2-4 transformer layers) that takes z as a conditioning vector and generates the corresponding token sequence. It handles only the "easy" part — syntactic realization of a semantic representation — so it can be small.

**Key insight:** The decoder is almost a **deterministic mapping.** Given the abstract meaning z, the surface form is mostly determined (with some stylistic variation). This is why current LMs waste so much capacity — they're using a massive model to do a job that's 90% lookup.

---

## 4. Training Objective

### 4.1 Primary: Latent Prediction Loss

We use a **JEPA-style asymmetric architecture** with an exponential moving average (EMA) target encoder:

```
Target:     z̄ₜ₊₁ = sg(EMA_Encoder(chunk_{t+1}))     # stop-gradient
Prediction: ẑₜ₊₁ = Predictor(Encoder(chunk_1), ..., Encoder(chunk_t))

Loss_pred = ||ẑₜ₊₁ - z̄ₜ₊₁||²    (or cosine similarity loss)
```

The EMA target encoder prevents **representation collapse** (where everything maps to the same vector). This is the same trick that makes BYOL, DINO, and I-JEPA work.

**Why this works better than token prediction:**
- The target z̄ₜ₊₁ captures the **predictable information** in the next chunk
- Unpredictable details (exact word choice, syntactic variation) are discarded by the encoder's bottleneck
- The model focuses its capacity on **semantic prediction** — the hard, important part

### 4.2 Secondary: Token Reconstruction Loss

To ensure the latent space is grounded and the decoder works:

```
Loss_recon = CrossEntropy(Decoder(z̄ₜ), tokens_t)
```

This is a **VQ-VAE-style reconstruction objective** but with continuous latents. It ensures the latent vectors actually contain enough information to reconstruct the original text.

### 4.3 Tertiary: Multi-Scale Consistency

Higher-level predictions should be consistent with lower-level predictions:

```
Loss_consistency = ||pool(ẑₜ₊₁⁽⁰⁾, ..., ẑₜ₊ₖ⁽⁰⁾) - ŝ⁽¹⁾||²
```

### 4.4 Total Loss

```
L = Loss_pred + λ₁·Loss_recon + λ₂·Loss_consistency + λ₃·KL_bottleneck
```

---

## 5. Inference: Two Modes

### Mode 1: "Thinking" (Latent-Only)

The model predicts future latent representations without generating tokens. This is useful for:
- **Planning:** What should I say next? (predict several latents ahead)
- **Reasoning:** What follows from these premises? (predict latent chain)
- **Evaluation:** Is this a good response? (predict latent, compare to candidate)

This is **much faster than token generation** — we're predicting 1 latent per ~8 tokens, and the predictor sequence is 4-16x shorter.

### Mode 2: "Speaking" (Full Generation)

1. Predict next latent: ẑₜ₊₁ = Predictor(z₁..zₜ)
2. Decode to tokens: tokens = Decoder(ẑₜ₊₁)
3. Encode generated tokens back: zₜ₊₁ = Encoder(tokens)
4. Append to context, repeat

The model can also do **speculative latent decoding**: predict several latents ahead in parallel, decode all of them, then verify consistency.

### Mode 3: "Deliberation" (Multi-Step Latent Reasoning)

Before generating any tokens, the model runs multiple predictor steps in latent space:

```
ẑ₁ = Predictor(context)          # first thought
ẑ₂ = Predictor(context, ẑ₁)      # second thought
ẑ₃ = Predictor(context, ẑ₁, ẑ₂)  # third thought
tokens = Decoder(ẑ₃)              # only now generate text
```

This is **implicit chain-of-thought without token overhead.** The model "thinks" in latent space before committing to words. Each latent reasoning step is ~100x cheaper than generating a chain-of-thought in tokens.

---

## 6. Why This Could Beat Attention-Based Token Prediction

### 6.1 Capacity Allocation

| | Token LM | LPLM |
|---|---|---|
| Predicts | Next subword (~50k classes) | Next semantic representation (continuous) |
| Capacity spent on function words | ~40% | ~0% (absorbed by encoder) |
| Capacity spent on reasoning | ~60% | ~100% of predictor |
| Context length (88M params, 8k tokens) | 8k tokens | 500-2000 latents = effective 8k tokens |

### 6.2 Computational Efficiency

- **Predictor operates on 4-16x shorter sequences** → quadratic attention (if used) is 16-256x cheaper
- **Latent prediction is one step per chunk** vs one step per token → 4-16x fewer forward passes during generation
- **Deliberation mode** provides chain-of-thought quality without chain-of-thought token cost

### 6.3 Theoretical Advantages

- **Information-theoretic:** The latent bottleneck forces the model to learn compressed, abstract representations — exactly what we want for reasoning
- **Compositional:** Predicting in latent space naturally encourages compositional structure (combining meaning units) rather than surface-level pattern matching
- **Hierarchical:** Multi-scale prediction enables genuine planning and discourse-level coherence — something flat token models fundamentally lack

### 6.4 What Could Go Wrong

| Risk | Mitigation |
|---|---|
| Representation collapse | EMA target encoder (proven in JEPA/DINO) |
| Lossy compression loses critical details | Reconstruction loss + tunable bottleneck β |
| Boundary prediction is hard to learn | Start with fixed-size chunks, anneal to adaptive |
| Latent space is hard to decode | Pre-train decoder, use reconstruction loss |
| Need more parameters for same "effective" capacity | Each component can be smaller since it has a focused job |

---

## 7. Comparison to Related Work

| Method | Predicts | Granularity | Latent Space | Planning |
|---|---|---|---|---|
| GPT/LLaMA | Next token | Subword | No | No |
| BERT/T5 | Masked tokens | Subword | No | No (bidirectional) |
| I-JEPA (vision) | Next patch repr | Patch | Yes | No |
| Mamba/SSMs | Next token | Subword | No | No |
| MEGABYTE | Next patch of bytes | Fixed byte patches | No | No |
| **LPLM (ours)** | **Next semantic repr** | **Adaptive phrases** | **Yes** | **Yes (multi-scale + deliberation)** |

### Key differences from MEGABYTE
MEGABYTE patches bytes into fixed-size groups — this is still token prediction, just with a two-level model. LPLM learns **semantically meaningful** variable-length chunks and predicts in a **continuous latent space**, not a discrete token space. The compression is semantic, not positional.

### Key differences from I-JEPA
I-JEPA predicts masked patches in vision. LPLM extends this to **autoregressive** prediction of sequential semantic units in language, with a **hierarchical** multi-scale structure and a **dedicated decoder** for token generation.

---

## 8. Experimental Plan

### Phase 1: Proof of Concept (88M params, 6M tokens)

**Fixed-chunk LPLM:**
- Chunk size: 8 tokens (fixed, no adaptive segmentation)
- Encoder: 4-layer transformer, d=512
- Predictor (RSN): 8 layers, state matrix 256×256
- Decoder: 2-layer transformer, d=512
- Total: ~88M parameters
- Compare to: transformer baseline, Mamba, best frontier architectures

**Metric:** Since we predict latents, not tokens, we can't directly compare perplexity. Instead:
- **Reconstruction perplexity**: Encode → predict latent → decode → measure token-level perplexity
- **Latent prediction MSE**: How well does the predictor predict the target encoder's output?
- **Downstream accuracy**: On simple QA/reasoning tasks, does latent deliberation help?

### Phase 2: Adaptive Chunking

- Train the boundary predictor
- Compare fixed vs adaptive chunks
- Analyze what the model learns to treat as semantic units

### Phase 3: Multi-Scale + Deliberation

- Add sentence-level and paragraph-level predictors
- Test deliberation mode (multiple latent steps before generating)
- Compare to explicit chain-of-thought

### Phase 4: Scale

If Phase 1-3 show promise:
- Scale to 1B parameters, 100B tokens
- Compare to LLaMA-scale transformers
- Test on reasoning benchmarks where latent deliberation should help most

---

## 9. The Big Bet

The hypothesis behind LPLM is that **next-token prediction is a bottleneck on intelligence, not just a training objective.**

By predicting tokens, current LMs are forced to think at the granularity of subwords. They can't skip ahead. They can't plan. They can't think abstractly without verbalizing every step.

LPLM breaks this constraint. The predictor thinks in abstract representations. It can reason about meaning without being tied to surface form. The decoder is just a "language renderer" — the last mile of turning thoughts into words.

If this works, it suggests that the path to more capable AI isn't bigger transformers predicting tokens — it's **models that learn to think in the right abstraction space, at the right granularity, with the right structure.**

---

## Appendix A: Mathematical Formulation

### Encoder

Given input tokens x = (x₁, ..., x_N), the encoder produces:
- Token embeddings: h = TransformerBlock(Embed(x)) ∈ ℝ^{N×d}
- Boundary scores: b = σ(W_b · h) ∈ ℝ^N
- Chunk boundaries: B = {i : bᵢ > τ} (during inference; soft during training)
- Chunk representations: zⱼ = Compress(h[Bⱼ:Bⱼ₊₁]) for j = 1..M where M = |B|

### Predictor (RSN)

State evolution for layer l:
```
gₜ = σ(W_g · [zₜ; vec(Sₜ₋₁)])
Aₜ = I + tanh(W_A · zₜ) · W_A'              # near-identity rotation
Bₜ = I + tanh(W_B · zₜ) · W_B'
Cₜ = W_C · zₜ,  Dₜ = W_D · zₜ
Sₜ = (1-gₜ)·Sₜ₋₁ + gₜ·(Aₜ·Sₜ₋₁·Bₜ + Cₜ·Dₜᵀ)
output_t = W_out · vec(Sₜ) + W_skip · zₜ
```

### Decoder

Standard autoregressive transformer conditioned on latent z:
```
p(x_{i+1} | x_{1:i}, z) = softmax(W_vocab · TransformerBlock(Embed(x_{1:i}), z))
```
where z is injected via cross-attention or adaptive layer norm (FiLM conditioning).

### EMA Target Encoder

```
θ_target ← α · θ_target + (1-α) · θ_encoder,    α = 0.996 → 1.0 (cosine schedule)
```
