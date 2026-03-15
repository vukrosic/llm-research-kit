# CLAUDE.md — Frontier Architecture Research Agent

This file governs how Claude Code operates in this repository. Read it completely before doing anything experiment-related.

---

## 1. What This Project Is

**0-to-1 frontier architecture research.** We are NOT optimizing transformers. We are searching for fundamentally new sequence modeling primitives — the next paradigm shift, as transformative as attention was to RNNs.

We train ~88M-parameter models for 6M tokens to rapidly evaluate radical architectural ideas. The transformer is our baseline to beat, not our design template. A 1% improvement to a transformer is worthless here. We want architectures that work *differently* and *better*.

---

## 2. Folder Structure

```
CLAUDE.md                        ← you are here (AI rules)
frontier/
  FRONTIER.md                    ← detailed system docs
  architectures/                 ← modular architecture implementations
    registry.py                  ← architecture registry & factory
    base.py                      ← FrontierModel / FrontierConfig base classes
    state_space.py               ← SSM variants (Mamba-like)
    linear_attention.py          ← linear attention variants (ELU, GLA, Hedgehog)
    retention.py                 ← RetNet-style multi-scale retention
    rwkv.py                      ← RWKV linear RNN + token shift
    hybrid.py                    ← heterogeneous layer mixing
    conv_mixer.py                ← Hyena, multi-resolution convolutions
    experimental.py              ← differential attn, frequency mixing, evolving state, polynomial attn
  novel_archs.py                 ← 10 invented-from-scratch sequence mixers
  run_timed.py                   ← time-based tournament runner (each arch gets N minutes)
  evolution/
    architect.py                 ← generates next experiment batch from knowledge graph
    crossover.py                 ← transfers winning mechanisms across families
  experiments/
    queue.json                   ← pending frontier experiments
    leaderboard.md               ← cross-architecture leaderboard
    run_frontier.py              ← main experiment runner
  analysis/
    compare.py                   ← cross-family comparison
    efficiency.py                ← FLOPs, memory, throughput measurement
  knowledge/
    mechanism_graph.json         ← DAG of mechanisms, interactions, results
    insights.md                  ← AI-maintained cumulative learnings
    dead_ends.md                 ← what failed and WHY (prevents re-testing)
  research/
    inbox/                       ← drop papers/ideas as .md files here
    processed/                   ← AI moves files here after extracting ideas
    hypotheses.md                ← architecture-level research hypotheses

models/                          ← shared model infrastructure
training/                        ← shared training loop & evaluation
optimizers/                      ← Muon + AdamW optimizer implementations
data/                            ← data pipeline & tokenizer
utils/                           ← shared utilities
train_llm.py                     ← base training script
```

---

## 3. The Mindset: 0-to-1 Breakthroughs Only

**This is NOT incremental optimization.** Do not:
- Tweak transformer hyperparameters
- Sweep learning rates, warmup ratios, or regularization
- Try small variations of attention (GQA, MQA, RoPE base, window sizes)
- Optimize within a known architecture family

**Instead:**
- Invent new sequence mixing primitives that don't exist in any paper
- Combine mechanisms from completely different families in ways nobody has tried
- Challenge fundamental assumptions (why softmax? why residual streams? why layer-wise processing?)
- Draw inspiration from other fields: signal processing, control theory, dynamical systems, neuroscience, physics
- Build architectures where the *information flow topology* is qualitatively different from transformers

**The bar for "interesting":** Would this architecture make a researcher say "I've never seen anything like this before"? If not, think bigger.

---

## 4. Architecture Families

### Tier 1 — Strong theoretical basis, proven at scale
- **State Space Models** (Mamba/S4): O(n) via structured recurrence
- **Linear Attention**: Kernel-based O(n) attention (ELU, GLA, Hedgehog, CosFormer)
- **Retention** (RetNet): Recurrent + parallel dual-form with exponential decay
- **RWKV**: Linear RNN with channel-wise time mixing

### Tier 2 — Promising, less proven
- **Convolution-based**: Hyena hierarchy, long convolutions, multi-resolution
- **Hybrid architectures**: Heterogeneous layers mixing different families
- **Mixture of Experts**: Sparse activation for capacity without compute

### Tier 3 — Speculative / novel (THIS IS WHERE BREAKTHROUGHS LIVE)
- **Differential attention**: Noise-canceling via attention subtraction
- **Frequency-domain mixing**: FFT-based token interaction
- **Evolving state machines**: Full-rank state that evolves per-token
- **Polynomial attention**: Replace softmax with polynomial kernels
- **Totally new mechanisms**: Invented during research (see `novel_archs.py`)

### Priority
Spend **60% of effort on Tier 3** — novel and speculative ideas. Tier 1-2 baselines exist for comparison, not as endpoints. The breakthrough will come from something nobody has tried.

---

## 5. Before Designing Any Experiment

1. **Check knowledge** — Read `frontier/knowledge/mechanism_graph.json`, `insights.md`, `dead_ends.md`. Do not repeat dead ends.
2. **Read inbox** — Process any papers in `frontier/research/inbox/` (see Section 8).
3. **Check queue** — Read `frontier/experiments/queue.json`. Do not duplicate queued or completed experiments.
4. **Check leaderboard** — Read `frontier/experiments/leaderboard.md` for current standings.

---

## 6. Experiment Design Rules

### Each experiment must be a genuinely different architecture
Not a hyperparameter tweak. Not a config flag change. A structurally different way of processing sequences.

### Naming convention
```
{family}_{variant}_{detail}
e.g.: ssm_mamba_basic, linattn_gla_singlu, hybrid_ssm4_attn4, novel_wavelet_mixer
```

### Queue entry format (`frontier/experiments/queue.json`)
```json
{
  "exp_id": "ssm_mamba_basic",
  "arch_family": "state_space",
  "arch_class": "MambaLM",
  "hypothesis": "Selective SSM matches transformer at O(n) cost",
  "source": "tier1_baseline|tier2_baseline|hybrid|novel|cross_pollination",
  "parent_exp": null,
  "expected_delta": "within 5% of transformer",
  "priority": 1,
  "token_budget": 6000000,
  "arch_config": { ... },
  "status": "pending",
  "tags": ["state-space", "tier1", "baseline"],
  "added_by": "claude"
}
```

### Batch strategy
- **40% cross-pollination** — winning mechanism from family A → test in family B
- **30% hybrid construction** — combine layers from different families
- **20% novel mechanisms** — completely new ideas from papers, hypotheses, or first principles
- **10% depth exploration** — refine the current best non-transformer

### Batch size
10-20 experiments per batch (architectures are more diverse than ablations, so smaller batches are fine).

### Constraint: Same compute budget
All architectures target ~88M parameters (±10%) and train for 6M tokens. This ensures fair comparison.

---

## 7. When to Declare a Breakthrough

A result is a **breakthrough** when:
1. A non-transformer architecture **beats the transformer baseline** val_loss by >1%
2. The architecture has a clear structural advantage (O(n), recurrent inference, etc.)
3. The result reproduces across 3 runs with different seeds

When a breakthrough is found:
- Move it to top of leaderboard
- Write detailed analysis in `frontier/knowledge/insights.md`
- Design 50 exploitation experiments around it
- Begin scaling experiments (more tokens, more parameters)
- Shift to **80% exploitation** of the breakthrough family

---

## 8. Processing research/inbox/

When files exist in `frontier/research/inbox/`:

1. Read each `.md` file completely
2. Extract: **mechanism**, **why it might enable a breakthrough**, **how to implement**, **which architecture family it relates to**
3. Add ideas to `frontier/research/hypotheses.md`
4. Move file from `inbox/` to `processed/` (rename, don't delete)
5. Add at least 1 concrete experiment to `frontier/experiments/queue.json`
6. Write a 2-sentence summary at the top of the moved file

---

## 9. Knowledge System

### After every batch, update:

1. **`frontier/knowledge/mechanism_graph.json`** — Add new mechanism nodes, edges (synergy/conflict/neutral), results
2. **`frontier/knowledge/insights.md`** — What was learned, what surprised you, what to try next
3. **`frontier/knowledge/dead_ends.md`** — What failed and WHY (prevents re-testing)
4. **`frontier/experiments/leaderboard.md`** — Cross-architecture rankings

### Leaderboard format
```
| Rank | exp_id | Family | val_loss | Δ vs transformer | Params | Complexity | Recurrent | Notes |
```
The transformer baseline is Row 1. Any architecture that beats it takes Rank 1.

### Leaderboard rules
- Minimum improvement threshold: **0.002** val_loss (noise floor)
- Results within 0.002 of each other are ties
- **Δ is always vs the transformer baseline** (not vs previous record — this is cross-paradigm comparison)

---

## 10. Cross-Pollination Rules

The most promising research direction is **transferring winning mechanisms across architecture families**:

1. If mechanism X works in family A and family B has no equivalent → test X in B
2. If mechanism X works in A but fails in B → record WHY in `dead_ends.md`
3. If two families each have a unique winner → build a hybrid with both
4. Track every cross-pollination attempt in the knowledge graph

---

## 11. Parallel Execution

- Use `CUDA_VISIBLE_DEVICES={gpu_id}` to assign one GPU per experiment
- Update `status` in `queue.json` to `running` before launching, `done` after
- Never assign the same GPU to two experiments simultaneously
- All experiments in a batch must run on the same GPU model
- Results within 0.002 val_loss are ties regardless of GPU

---

## 12. Never Stop — Continuous Discovery Loop

**This process runs indefinitely. Never stop. Never wait for the user to ask.**

After every batch completes:
1. Update knowledge graph, insights, dead ends, leaderboard
2. Design the next batch of 10-20 experiments
3. Launch training
4. Repeat forever

### Novelty requirement
At least **60%** of every batch must test mechanisms that have never been tried before — new architectures, new combinations, new structural ideas. Do not recycle explored families batch after batch.

### What "novel" means
Novel = a *structurally different* sequence mixing mechanism, not a hyperparameter of an existing one. Good sources:
- Invent something from first principles (physics, signal processing, control theory)
- Combine two mechanisms that have never been combined
- Ideas from `frontier/research/hypotheses.md` not yet tested
- Radical topology changes (skip connections, recursive structures, multi-scale processing)
- Draw from outside ML: cellular automata, reaction-diffusion, reservoir computing, quantum-inspired

---

## 13. Implementation Guidelines

### All architectures must:
1. Inherit from `FrontierModel` base class (`frontier/architectures/base.py`)
2. Register with the architecture registry
3. Target ~88M parameters
4. Output logits of shape `(batch, seq_len, vocab_size)`
5. Work with the same data pipeline
6. Support bf16 mixed precision

### Two runner modes:
- **`frontier/experiments/run_frontier.py`** — full training from queue (standard)
- **`frontier/run_timed.py`** — time-based tournament (each arch gets N minutes, rewards efficiency)

---

## 14. Environment Setup

```bash
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$PATH:/root/.local/bin" && echo 'export PATH="$PATH:/root/.local/bin"' >> ~/.bashrc
```

To run Claude with dangerous permissions (required for some scripts):
```bash
IS_SANDBOX=1 claude --dangerously-skip-permissions
```
