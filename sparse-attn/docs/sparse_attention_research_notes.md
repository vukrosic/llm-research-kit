# Sparse Attention Research Notes

This is the working literature and ablation map for the MiniMax-style sparse attention experiment in `sparse-attn`.

## 1) Short literature map

The current mechanism sits in the content-aware sparse-attention family.

- [Sparse Transformer](https://arxiv.org/abs/1904.10509) showed that hand-designed sparse factorizations can extend autoregressive modeling to much longer sequences.
- [Longformer](https://arxiv.org/abs/2004.05150) combined local windows with global tokens for long documents.
- [BigBird](https://arxiv.org/abs/2007.14062) mixed local, random, and global links and added useful theory around expressivity.
- [Reformer](https://arxiv.org/abs/2001.04451) routed similar tokens together with locality-sensitive hashing.
- [Routing Transformer](https://arxiv.org/abs/2003.05997) used online clustering for content-based sparse routing.
- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) is not sparse attention itself, but it reduces KV cost and is the right backbone for this kind of router.

More recent work is closer to what we built:

- [MoBA](https://arxiv.org/abs/2502.13189) treats attention like a mixture of block experts and argues that router fidelity and block size matter.
- [Native Sparse Attention](https://arxiv.org/abs/2502.11089) combines coarse token compression with finer token selection and emphasizes native training efficiency.
- [IndexCache](https://arxiv.org/abs/2603.12201) shows that sparse index decisions can be reused across layers with little quality loss, which is a very relevant clue for our model.

Our current model is basically:

1. GQA backbone.
2. Index branch scores blocks.
3. Block max-pool compresses scores.
4. Top-k selects blocks.
5. Sparse branch attends only to selected blocks.

That means our main research question is no longer whether sparse attention can work at all. It can. The question is which routing changes improve recall, stability, and speed without giving up too much quality.

## 2) What the current runs are saying

Current MacBook-scale real-data runs on `speedrun_40M`:

- 500k token run: dense and sparse are almost tied on validation loss.
- 1M token run: dense is still slightly ahead on validation loss.
- The sparse path is real, not a no-op.
- The loss gap is small enough that a single line plot hides the signal.

So the useful interpretation is:

- routing matters, but not enough yet to beat dense on short next-token loss
- the next gain will probably come from better routing quality or a task that stresses long-range retrieval
- the current architecture is good enough for ablations, not good enough for a final claim

## 3) Best metrics for this project

For each experiment, track:

- validation loss
- validation perplexity
- validation accuracy
- throughput in tokens/sec
- selected-block density
- router entropy
- selected-block recall against a dense oracle probe
- long-context retrieval accuracy
- stability of selected blocks across nearby layers

The most important thing is to separate "the router chose better blocks" from "the backend happened to run faster."

## 4) Architecture explanations

The architecture itself has a few separable parts, and each one answers a different question.

### A. Base transformer shape

This is the ordinary LLM scaffold: embedding size, number of layers, FFN width, head count, and KV head count.

Why it matters:

- It sets capacity.
- It changes the compute budget.
- It can hide or expose sparse-attention effects.

What we learn by scaling it:

- whether the sparse mechanism survives at larger capacity
- whether any quality gap is just underparameterization
- whether the router gets easier or harder to train as the model grows

### B. GQA backbone

GQA is the shared attention backbone underneath the sparse router.

Why it matters:

- it lowers KV memory and compute
- it changes how much information each KV head must carry
- it creates the grouping structure the router uses

What we learn by changing it:

- whether sparse routing depends on KV sharing
- whether some group sizes are too coarse or too fine
- whether the router benefits from more or fewer KV heads

### C. Index branch

This is the router. It decides which blocks are worth reading.

Why it matters:

- it determines recall
- it adds extra parameters and compute
- it can fail either by missing important blocks or by selecting too many useless blocks

What we learn by changing it:

- whether the router is too weak, too sharp, or too noisy
- whether the block score representation is the right one
- whether the model prefers max pooling, mean pooling, or another aggregator

### D. Sparse content branch

This is the actual attention that reads the chosen blocks.

Why it matters:

- it defines the information flow after routing
- it decides whether the sparse model can still recover local detail
- it determines whether the architecture behaves like "selected dense attention" or like a genuinely different mechanism

What we learn by changing it:

- whether a local fallback window is needed
- whether hard selection is too brittle
- whether sparse attention should keep a small dense safety path

### E. Reuse and hierarchy

This is the layer-to-layer or stage-to-stage question.

Why it matters:

- sparse routers may be redundant across adjacent layers
- selected blocks may stay stable across depth
- a cached or reused index could save real time without much quality loss

What we learn by changing it:

- whether routing decisions are stable enough to reuse
- whether a layerwise indexer is actually necessary everywhere
- whether the architecture wants hierarchy rather than a flat router

## 5) Ablation grid

### A. Routing shape

| Change | Why it matters | What to watch |
|---|---|---|
| `top_k` | Controls how much context survives | quality vs throughput frontier |
| `block_size` | Smaller blocks can improve retrieval fidelity | recall, density, runtime |
| `index_dim` | Router capacity vs overfitting/noise | loss, stability, entropy |
| pooling op | Max/mean/log-sum-exp can change what the router sees | routing sharpness |

### B. GQA interaction

| Change | Why it matters | What to watch |
|---|---|---|
| `n_kv_heads` | KV sharing changes memory cost and may affect routing clarity | quality, speed, memory |
| query-to-KV grouping | Router behavior may depend on group size | per-group specialization |
| shared vs separate router projections | Can tell us whether the router should share content features with attention | routing overlap, loss |

### C. Sparsity strategy

| Change | Why it matters | What to watch |
|---|---|---|
| local fallback window + routed blocks | Protects near-context continuity | long-context quality |
| hard top-k vs soft routing | Tests whether discrete selection is too brittle | quality, entropy |
| cross-layer index reuse | Inspired by IndexCache | speed, overlap across layers |
| sparse router distillation | May stabilize selection | quality, recall |

### D. Task design

| Change | Why it matters | What to watch |
|---|---|---|
| long-context retrieval | Best stress test for routing | exact-match accuracy |
| needle/passkey style probes | Makes failures visible | recall under long contexts |
| repeated-prefix or replay tasks | Tests cache/reuse behavior | reuse stability |

## 6) How to organize the experiments

I would organize the work as three nested layers:

1. **Scale sweep**: current size, +50%, +50% again.
2. **Architecture sweep**: change one router or branch design at a time.
3. **Context sweep**: 256, 512, 1024, and longer eval-only if needed.

That gives us a clean story:

- scale tells us whether the mechanism survives capacity changes
- architecture tells us which part of sparse attention is actually helping
- context length tells us whether the mechanism is really doing long-context work

## 7) My current hypothesis

The current near-tie on short next-token loss probably means the model is operating in a regime where the router is not yet the bottleneck. The next useful signal is likely to come from one of three places:

1. better block selection quality
2. reuse of routing decisions across layers or positions
3. a task that really needs long-range retrieval

If we only keep extending the same short-context language-model run, we will probably keep seeing the same weak separation.

## 8) Recommended next three experiments

1. `top_k x block_size` grid at one seed, fixed token budget, with routing-density and recall logging.
2. local-window fallback vs pure routed blocks.
3. long-context retrieval probe with a dense oracle comparison, plus a router reuse test across adjacent layers.

That should tell us whether the best path is better routing, hybrid routing, or router caching.

## 9) What the sweeps showed

The first real sweep gave us a useful answer: the attention mechanism is not fragile, but the biggest signal is not pooling style.

- Dense and sparse stay close on short next-token loss at this scale.
- Scaling the model helps more than swapping router style.
- Longer evaluation contexts to 1024 tokens stay stable in this small-budget regime.
- Pooling style and shared-router projection were effectively degenerate in the mid-scale sweep; they routed to the same blocks on a held-out real batch.
- `top_k` and block size were the clearer architectural levers because they changed routing density, throughput, and loss more visibly.

That means the next useful set of experiments is not just “more of the same.” It should focus on:

1. better block selection quality
2. local fallback or hybrid sparse+dense behavior
3. whether routing decisions can be reused across layers
4. retrieval tasks that force the router to matter
