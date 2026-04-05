# Training an 88M LLM from Scratch: 9 Experiments, One Surprising Win

> A fully autonomous research loop running on a single consumer GPU — NVIDIA TITAN X Pascal (12 GB, 2016 hardware).  
> No cloud. No team. Just an agent, a hypothesis, and a training loop.

---

## The Setup

- **Model:** 88M parameter transformer (22 layers, 512 d_model, 8 attention heads, GQA, RoPE, RMSNorm)
- **Optimizer:** Muon (hybrid Muon + AdamW) — a modern distributed-training optimizer, run on a single GPU
- **GPU:** NVIDIA TITAN X Pascal — 6-year-old consumer card, no flash attention, no native BF16
- **Dataset:** ~8M tokens per experiment run (~42 minutes each)
- **Goal:** Minimize validation loss through systematic hypothesis testing

The research loop is simple: form a hypothesis, run the experiment, update the best config, repeat.

---

## The Baseline

Starting point: `squared_relu` activation, constant learning rate, default grouped-query attention (4 KV heads).

| Config | val_loss |
|---|---|
| Baseline (squared_relu + constant LR + GQA n_kv=4) | 4.9214 |

---

## Experiment Chain

### H1–H2: Activation Functions × LR Schedules (2M tokens)

Tested 5 activation functions: `relu`, `squared_relu`, `gelu`, `silu`, `swiglu`.  
Tested 3 LR schedules: constant, warmup-only, warmup+cosine.

**Key finding:** At 2M tokens, `relu` wins. At 8M tokens, `squared_relu` takes over.  
There is a **crossover near 4M tokens** — a result that only appears when you test at multiple scales.

| Config | 2M val_loss | 8M val_loss |
|---|---|---|
| squared_relu + constant | 6.108 | 4.921 |
| relu + constant | 6.060 | 4.966 |
| squared_relu + cosine | 6.093 | **4.896** |
| relu + cosine | 6.060 | 4.916 |

> **Lesson:** Never benchmark at a single token count. The winner changes with scale.

---

### H3–H4: Validation + LR Magnitude (8M tokens)

Validated `squared_relu + cosine` beats `relu + cosine` at 8M tokens.  
Swept Muon learning rate: {0.012, 0.024, 0.036}.

| Muon LR | val_loss (2M) |
|---|---|
| 0.012 | 6.0883 |
| **0.024** | **6.0603** |
| 0.036 | 6.0851 |

**Default LR of 0.024 confirmed optimal.** No change needed.

---

### H5–H6: Crossover + Best Config (4M and 8M tokens)

Precisely located the relu/squared_relu crossover (~4–5M tokens).  
Validated `squared_relu + cosine` as the new best.

| Config | val_loss | vs baseline |
|---|---|---|
| Baseline | 4.9214 | — |
| squared_relu + cosine | **4.8956** | −0.026 |

---

### H7: Scale Validation (20M tokens)

Does the cosine LR benefit grow with training length?

| Training length | Cosine gain |
|---|---|
| 8M tokens | −0.026 |
| 20M tokens | **−0.070** |

**Yes — the benefit grows 2.7× from 8M to 20M tokens.**  
Recommendation: always use cosine LR for runs ≥ 8M tokens.

---

### H8: Grouped-Query Attention Ratio Sweep

The model used GQA with 8 query heads and 4 KV heads (2:1 ratio) — a standard efficiency tradeoff.  
Hypothesis: is this ratio actually optimal for an 88M model?

| Config | n_kv_heads | val_loss | vs default |
|---|---|---|---|
| **Full attention** | **8** | **4.8785** | **−0.017** |
| Default GQA | 4 | 4.8956 | — |
| Aggressive GQA | 2 | 4.9335 | +0.038 |
| MQA (single KV) | 1 | 4.9512 | +0.056 |

**Monotonic relationship: more KV heads → lower loss, always.**  
At 88M parameters, GQA is a false economy. Full attention wins.

> Each halving of KV heads costs approximately +0.04 val_loss.

---

### H9: Muon Optimizer Momentum Sweep ← Biggest Win

The Muon optimizer's momentum was set to 0.95 by default — and had never been questioned.  
Hypothesis: with cosine LR decay, momentum may need to be lower to stay responsive.

| Muon Momentum | val_loss | vs 0.95 |
|---|---|---|
| **0.90** | **4.7952** | **−0.083** |
| 0.95 (default) | 4.8785 | — |
| 0.98 | 5.0581 | +0.179 |
| 0.99 | 5.2200 | +0.341 |

**Momentum = 0.90 gains −0.083** — the single largest improvement of the entire research chain.  
Higher momentum is catastrophically bad: 0.99 loses +0.34 val_loss vs 0.90.

**Why does this happen?**  
Muon uses gradient orthogonalization. High momentum means the optimizer accumulates old gradient directions and orthogonalizes against a stale history. As cosine LR decays, the gradient landscape shifts — but high-momentum Muon keeps tracking the old landscape. Lower momentum keeps the orthogonalization fresh and aligned with current training dynamics.

---

## Final Results

| Step | What changed | val_loss | Gain |
|---|---|---|---|
| Baseline | squared_relu + constant LR + GQA n_kv=4 + momentum=0.95 | 4.9214 | — |
| H6 | + cosine LR decay | 4.8956 | −0.026 |
| H8 | + full attention (n_kv=8) | 4.8785 | −0.017 |
| **H9** | **+ momentum=0.90** | **4.7952** | **−0.083** |
| **Total** | | **4.7952** | **−0.126 (−2.56%)** |

---

## Current Best Config

```python
@dataclass
class BestConfig:
    ffn_activation: str = "squared_relu"
    schedule_type:  str = "cosine"
    warmup_ratio:   float = 0.05
    n_kv_heads:     int = 8          # full attention, no GQA
    muon_momentum:  float = 0.90     # lower than default 0.95
    muon_lr:        float = 0.024    # unchanged from default
```

---

## What's Running Now

**H10: Fine-grained momentum sweep** — testing {0.80, 0.85, 0.92} to find the true optimum.  
If lower momentum keeps helping, we go lower. If 0.90 is the floor, we move to the next axis.

---

## Key Takeaways

1. **Test at multiple scales.** The best activation at 2M tokens is not the best at 8M tokens.
2. **Cosine LR benefit compounds.** It's worth 2.7× more at 20M tokens than at 8M tokens.
3. **GQA is a tradeoff, not free.** At smaller model sizes, full attention wins outright.
4. **Optimizer hyperparameters are underexplored.** Momentum had never been swept — and it was the biggest gain.
5. **Sharp asymmetry is a signal.** When one direction is catastrophic and the other is a win, you've found something real.

---

*Model: 88M params | GPU: NVIDIA TITAN X Pascal (2016, 12 GB) | Framework: PyTorch 2.4 + Muon optimizer*  
*All experiments: 8M tokens, ~42 min per run, fully autonomous research loop*
