# FRONTIER.md — Beyond-Transformer Architecture Research

This file governs the frontier architecture research system. The goal is to discover the **next paradigm shift** in sequence modeling — something as transformative as attention was to RNNs.

---

## 1. Mission

We are not tweaking transformers. We are searching for **fundamentally better sequence modeling primitives**. The transformer is our strong baseline to beat, not our design template.

Every architecture must be evaluated at the same 88M-parameter / 6M-token scale as the ablation system, using the same data pipeline and evaluation, so results are directly comparable.

---

## 2. System Structure

```
frontier/
├── FRONTIER.md                    ← you are here
├── architectures/                 ← modular architecture implementations
│   ├── registry.py               ← architecture registry & factory
│   ├── base.py                   ← base class all architectures implement
│   ├── state_space.py            ← SSM variants (Mamba-like)
│   ├── linear_attention.py       ← linear attention variants
│   ├── retention.py              ← RetNet-style retention
│   ├── rwkv.py                   ← RWKV-style linear RNN
│   ├── hybrid.py                 ← mix-and-match heterogeneous layers
│   ├── conv_mixer.py             ← convolution-based sequence mixing
│   └── experimental.py           ← truly novel / uncategorized ideas
├── knowledge/
│   ├── mechanism_graph.json      ← graph of all mechanisms, interactions, results
│   ├── insights.md               ← AI-maintained cumulative insights
│   └── dead_ends.md              ← what didn't work and WHY
├── evolution/
│   ├── architect.py              ← generates new architecture candidates from knowledge
│   └── crossover.py              ← combines winning components across families
├── experiments/
│   ├── queue.json                ← frontier experiment queue
│   ├── leaderboard.md            ← cross-architecture leaderboard
│   └── run_frontier.py           ← runner for any registered architecture
├── analysis/
│   ├── compare.py                ← cross-architecture comparison
│   └── efficiency.py             ← FLOPs, memory, throughput measurement
└── research/
    ├── inbox/                    ← drop papers/ideas here
    ├── processed/                ← processed papers
    └── hypotheses.md             ← architecture-level hypotheses
```

---

## 3. Architecture Families

### Tier 1 — Strong theoretical basis, proven at scale elsewhere
- **State Space Models** (Mamba/S4): O(n) sequence mixing via structured recurrence
- **Linear Attention**: Replace softmax with kernel-based linear attention (O(n) inference)
- **Retention** (RetNet): Recurrent + parallel dual-form with exponential decay
- **RWKV**: Linear RNN with channel-wise time mixing and token shift

### Tier 2 — Promising but less proven
- **Convolution-based**: Hyena hierarchy, long convolutions
- **Hybrid architectures**: Mix transformer layers with SSM/linear-attn layers
- **Mixture of Experts**: Sparse activation for capacity without compute

### Tier 3 — Speculative / novel
- **Polynomial attention**: Replace softmax with polynomial kernels
- **Differential attention**: Attention as difference of two softmax maps
- **Frequency-domain mixing**: FFT-based token interaction
- **Graph neural sequence models**: Treat sequence as dynamic graph
- **Totally new mechanisms**: Invented during research

---

## 4. Research Loop

### Phase 1: Establish baselines (one-time)
1. Run the current best transformer config through `run_frontier.py` to get a reference val_loss
2. Implement and run one canonical variant from each Tier 1 family
3. Record all results in the leaderboard

### Phase 2: Iterative discovery (continuous)
For each batch:

1. **Read knowledge** — Check `knowledge/mechanism_graph.json`, `insights.md`, `dead_ends.md`
2. **Generate candidates** — Use `evolution/architect.py` logic:
   - 40% **cross-pollination**: Take a winning mechanism from family A, test it in family B
   - 30% **hybrid construction**: Combine layers from different families
   - 20% **novel mechanisms**: Completely new ideas from papers or first principles
   - 10% **depth exploration**: Refine the current best non-transformer architecture
3. **Run experiments** — Batch of 10-20 experiments (smaller than ablation batches because architectures are more diverse)
4. **Update knowledge** — After every batch:
   - Update `mechanism_graph.json` with new nodes and edges
   - Update `insights.md` with what was learned
   - Update `dead_ends.md` for failures (with WHY)
   - Update leaderboard
5. **Evolve** — The architect reads the updated knowledge and proposes the next batch

### Phase 3: Breakthrough exploitation
When any non-transformer architecture beats the transformer baseline:
1. Shift to 80% exploitation of that architecture family
2. Apply all known winning tricks from the transformer ablation system
3. Run ablation sweeps within the winning family
4. Document the breakthrough in `knowledge/insights.md`

---

## 5. Experiment Design Rules

### Constraint: Same compute budget
All architectures must target ~88M parameters and train for 6M tokens. This ensures fair comparison. The parameter count can vary ±10% (80M-96M) to accommodate architectural constraints.

### Queue entry format
```json
{
  "exp_id": "ssm_mamba_basic",
  "arch_family": "state_space",
  "arch_class": "MambaLM",
  "hypothesis": "Selective SSM with input-dependent gating matches transformer quality at O(n) cost",
  "source": "tier1_baseline",
  "parent_exp": null,
  "expected_delta": "within 5% of transformer baseline",
  "priority": 1,
  "token_budget": 6000000,
  "arch_config": {
    "d_model": 512,
    "n_layers": 24,
    "d_state": 64,
    "d_conv": 4,
    "expand_factor": 2
  },
  "status": "pending",
  "tags": ["state-space", "tier1", "baseline"]
}
```

### Naming convention
```
{family}_{variant}_{detail}
e.g.: ssm_mamba_basic, linattn_hedgehog_norm, hybrid_ssm4_attn4, rwkv_v6_timemix
```

---

## 6. Knowledge Graph

`knowledge/mechanism_graph.json` tracks:

```json
{
  "mechanisms": {
    "selective_scan": {
      "family": "state_space",
      "description": "Input-dependent SSM parameters (Mamba)",
      "first_tested": "ssm_mamba_basic",
      "results": [
        {"exp_id": "ssm_mamba_basic", "val_loss": 5.12, "delta_vs_transformer": -1.2}
      ],
      "interactions": {
        "gated_ffn": "positive",
        "rope": "not_applicable"
      },
      "status": "promising|neutral|dead"
    }
  },
  "edges": [
    {
      "from": "selective_scan",
      "to": "gated_ffn",
      "type": "synergy|conflict|neutral",
      "evidence": "ssm_mamba_gated showed +0.3% over ssm_mamba_basic"
    }
  ]
}
```

---

## 7. Cross-Pollination Rules

The most promising research direction is **transferring winning mechanisms across architecture families**:

1. If mechanism X works in transformers AND family Y has no equivalent → test X in Y
2. If mechanism X works in family A but fails in family B → record in `dead_ends.md` with analysis of WHY
3. If two families each have a unique winning mechanism → build a hybrid with both
4. Track every cross-pollination attempt in the knowledge graph edges

### Examples of cross-pollination
- SinGLU (transformer winner) → test as FFN in SSM blocks
- Selective gating (Mamba winner) → test as attention gate in transformers
- Time-mixing (RWKV) → test as alternative to attention in hybrid
- Exponential decay (RetNet) → test as attention bias in transformers

---

## 8. When to Declare a Breakthrough

A result is a **breakthrough** (not just an improvement) when:
1. A non-transformer architecture beats the transformer baseline val_loss by >1%
2. The architecture has a clear theoretical advantage (O(n) vs O(n²), recurrent inference, etc.)
3. The result reproduces across 3 independent runs with different seeds

When a breakthrough is found:
- Move it to the top of the leaderboard
- Write a detailed analysis in `knowledge/insights.md`
- Design 50 exploitation experiments around it
- Begin scaling experiments (more tokens, more parameters)

---

## 9. Leaderboard Format

```markdown
| Rank | exp_id | Family | val_loss | Δ vs transformer | Params | Notes |
|------|--------|--------|----------|------------------|--------|-------|
| 1    | ...    | ...    | ...      | ...              | ...    | ...   |
```

The transformer baseline is always Row 1 initially. Any architecture that beats it takes Rank 1.

---

## 10. Implementation Guidelines

### All architectures must:
1. Inherit from `FrontierModel` base class
2. Register with the architecture registry
3. Accept a config dict and produce the correct parameter count
4. Output logits of shape `(batch, seq_len, vocab_size)` — same as transformer
5. Work with the same data pipeline (input_ids → logits)
6. Support `torch.compile` (or gracefully fall back)
7. Support bf16 mixed precision

### The runner must:
1. Use the same training loop, optimizer setup, and evaluation as the ablation system
2. Record architecture family and config in metrics.json
3. Measure wall-clock time, peak memory, and throughput (tokens/sec)

---

## 11. Integration with Ablation System

This frontier system sits alongside the existing ablation system, not replacing it:
- Ablation system: optimizes within the transformer paradigm
- Frontier system: searches across paradigms

When the frontier system finds a winner, its mechanisms get ported back to the ablation system for detailed optimization. When the ablation system finds a winning mechanism, it gets tested across frontier architectures.
