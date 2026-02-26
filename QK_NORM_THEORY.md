# 🧠 QK Normalization: Theory & Ablations

QK Normalization is a relatively recent technique designed to improve the stability and scaling of Transformer models. Below is the theoretical breakdown of the ablations we are currently testing at the **100M token scale**.

---

## 1. The Core Problem: Attention Explosion
In a standard Transformer, Attention is calculated as:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
As models grow, the $Q$ and $K$ vectors naturally drift toward higher magnitudes. This "sharps" the softmax distribution, eventually leading to **entropy collapse** (where one token gets all the attention) and **gradient vanishing** (where the derivative of softmax becomes zero).

---

## 2. Theoretical Breakdown of Ablations

### A. Base QK Norm (`Norm -> RoPE`)
*   **The Theory**: Apply `RMSNorm` to $Q$ and $K$ *before* positional embeddings ($RoPE$).
*   **Why it works**: It forces the dot product to remain within a stable range, effectively capping the "sharpness" of the attention.
*   **Trade-off**: By normalizing before rotation, the positional signal added by RoPE might re-introduce some magnitude variance, though usually negligible.

### B. QK Bias (The 15M Winner)
*   **The Theory**: Add a learnable bias vector $\beta_q, \beta_k$ to $Q$ and $K$ *after* the normalization.
*   **Why it works**: Norm only controls **scale** (variance). Adding bias controls **shift** (mean). This allows the model to learn "global anchors"—certain heads can learn to default to specific token types or "sink" tokens even if their variances are normalized to 1.
*   **Verdict**: This performed best at 15M because it adds flexibility to a "too rigid" normalization.

### C. K-Only Normalization
*   **The Theory**: Normalize the Keys ($K$) but leave the Queries ($Q$) unnormalized.
*   **Why it works**: Often, the Query magnitude represents "urgency" (how much I want to look), while Key magnitude represents "attractiveness" (how much I should be looked at). Normalizing only $K$ prevents any single token from dominating the attention field globally while allowing individual Queries to stay "loud."
*   **Verdict**: Strong 2nd place in early tests; suggests asymmetric attention is better for representational capacity.

### D. Per-Head Scaling (Learned Temperature)
*   **The Theory**: Let each head learn its own $1/\sqrt{d_k}$ coefficient.
*   **Why it works**: Normalization fixes the variance to 1.0, effectively taking away the model's ability to sharpen its focus if it *needs* to. A per-head learnable scalar allows "Specialist Heads" (sharp focus) and "Global Heads" (soft focus) to coexist.
*   **Verdict**: Theoretically sound, but might require more tokens than 15M to show its worth.

### E. Shared Norm
*   **The Theory**: Use the same `RMSNorm` instance for both $Q$ and $K$.
*   **Why it works**: This forces $Q$ and $K$ into the exact same latent space. In a dot-product attention mechanism, this makes "matching" more intuitive as both vectors are measured against the same statistical ruler.
*   **Verdict**: Simplifies the model; usually performs similar to or slightly worse than dual-norm due to lack of head-specific variance.

---

## 🚫 Dropped Ablations (Why they failed)

### No QK Norm (Baseline)
*   **Failure Mode**: Suffered from the original "Explosion" problem. Validation loss was nearly 0.1 higher than the normalized versions, proving that even for small models (88M), the stability of QK Norm is measurable.

### RoPE then Norm
*   **Failure Mode**: When you apply `RMSNorm` *after* `RoPE`, the normalization actually tries to "erase" the positional information. RMSNorm is invariant to scaling; if RoPE rotates vectors in a way that creates magnitude shifts, the Norm will flatten them out, effectively "blinding" the model to specific relative positions.

---

## 📊 100M Token Hypothesis
At 100M tokens, we expect **QK Bias** and **Per-Head Scaling** to pull ahead. As the model sees more data, the ability to control the "shape" of attention dynamically (Scaling) and have "anchor points" (Bias) becomes more critical for fine-grained language modeling.
