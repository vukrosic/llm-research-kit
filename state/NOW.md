# Current State

## Active Goal

Autonomous research loop — improving 88M LLM val_loss on TITAN X Pascal.
User instruction: "continue forever"

## Running Work

**H10: Fine-grained momentum sweep** (PID=164639, /tmp/h10.log)
- Testing momentum ∈ {0.80, 0.85, 0.92} vs reference 0.90 (val_loss=4.7952)
- Full-attn + squared_relu + cosine, 8M tokens
- ETA: ~2 hours (3 × ~42 min runs)

## Latest Findings (2026-04-05)

### Cumulative best config:
Full attention (n_kv=8) + squared_relu + cosine LR + muon_momentum=0.90
→ val_loss=**4.7952** (vs 4.9214 original = −0.126, −2.56%)

### Improvement chain:
| Step | Change | val_loss | Gain |
|---|---|---|---|
| Baseline | squared_relu + constant, n_kv=4, mom=0.95 | 4.9214 | — |
| H6 | + cosine LR | 4.8956 | −0.026 |
| H8 | + full attention (n_kv=8) | 4.8785 | −0.017 |
| H9 | + momentum=0.90 | **4.7952** | **−0.083** |

### H9 finding (momentum sweep):
- 0.90: 4.7952 (-0.083) ← **new best**
- 0.95: 4.8785 (ref)
- 0.98: 5.0581 (+0.179)
- 0.99: 5.2200 (+0.341)

Sharp asymmetry: too much momentum kills training with cosine LR.
Interpretation: Muon orthogonalization needs fresh gradient signal; high momentum
causes it to track stale gradient directions, worsening as LR decays.

### H10 goal:
Find exact optimum. Does 0.85 or 0.80 beat 0.90? Or is 0.90 optimal?

## After H10

- If lower momentum helps → try 0.70–0.80 range (H11)
- If 0.90 is optimal → move to new axis:
  - AdamW LR (embedding/bias optimizer, never tested)
  - FFN expansion ratio d_ff/d_model ∈ {2, 4, 6}
  - 20M token validation with full best config

## Blockers

None. H10 running.
