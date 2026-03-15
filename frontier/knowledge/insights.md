# Frontier Architecture Insights

AI-maintained log of cumulative insights from frontier experiments.

---

## Starting Hypotheses (Pre-Experiment)

### Why non-transformer architectures might win at 88M scale:

1. **Inductive bias advantage**: At small scale, the right inductive bias matters more than raw expressiveness. SSMs and linear RNNs have strong sequential inductive biases that might help with limited data.

2. **Compute efficiency**: O(n) architectures get more effective training steps per token budget because their forward/backward passes are faster. At 6M tokens, this speed advantage could translate to better final loss.

3. **Gradient flow**: Recurrent architectures with gating (Mamba, RWKV, GLA) may have smoother gradient landscapes at 88M scale, reducing the need for careful LR scheduling.

### Why transformers might be hard to beat:

1. **Proven optimization**: The ablation system has 80+ generations of optimization on the transformer baseline. New architectures start from scratch.

2. **Hardware alignment**: Transformers are optimized for GPU matrix multiplication. SSMs and linear attention may not utilize hardware as efficiently.

3. **Short sequences**: At seq_len=2048, the O(n²) cost of attention is manageable. The efficiency advantage of O(n) architectures is less pronounced.

---

## Experiment Results

*Results will be logged here after each batch of experiments.*
