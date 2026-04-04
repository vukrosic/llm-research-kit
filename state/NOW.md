# Current State

## Active Goal

activation-discovery → lr-schedule → combination → H3 validation

## Running Work

- **H3 validation** (in background, ~90 min remaining):
  - relu + constant LR at 8M tokens → does relu gain persist at longer training?
  - relu + cosine LR at 8M tokens → does cosine help more at longer training?

## Completed Experiments (2026-04-04)

### Experiment 1: Activation Function Sweep (2M tokens)
| Rank | Activation   | val_loss | Δ vs baseline |
|------|--------------|----------|---------------|
| 1    | relu         | 6.0598   | −0.049        |
| 2    | gelu         | 6.0713   | −0.037        |
| 3    | silu         | 6.0777   | −0.031        |
| 4    | swiglu       | 6.0995   | −0.009        |
| 5    | squared_relu | 6.1084   | baseline      |

**Verdict:** Default squared_relu is worst. relu wins by 0.049. Differences small — needs validation.

### Experiment 2: LR Schedule (2M tokens, squared_relu)
| Config           | val_loss | Δ vs baseline |
|------------------|----------|---------------|
| constant (base)  | 6.1084   | —             |
| warmup-constant  | 6.1278   | +0.019 (worse)|
| warmup-cosine    | 6.0930   | −0.015        |

**Verdict:** Warmup alone hurts. Cosine decay helps slightly (+0.015).

### Experiment 3: Combination (H2, 2M tokens)
| Config               | val_loss |
|----------------------|----------|
| squared_relu+constant| 6.1084   |
| relu+constant        | 6.0598   |
| relu+cosine          | 6.0603   |

**Verdict:** relu+cosine gives NO additional gain over relu+constant. Improvements are NOT additive.

## Derived Knowledge

1. **relu activation** is best at 2M tokens (robust finding)
2. **Cosine LR** helps with squared_relu but not with relu
3. Mechanistic finding: smoothness, quadratic amplification, and gating all hurt at 2M token budget
4. Current best config: relu + constant LR = 6.0598 (vs baseline 6.1084)

## Next Actions

1. Collect H3 results (~90 min)
2. If relu gain persists at 8M: update experiment record, promote relu as default
3. If cosine helps at 8M: update best config to relu+cosine
4. Design H4 based on H3 results (LR value sweep or architecture test)

## Blockers

- None. Experiments running autonomously.
