# Optimization Plan

## Current Phase: Calibration & Baseline

### Hardware
L40S 46GB, 28 CPU cores, 1TB RAM

### Model
88M param MinimalLLM, Muon+AdamW, BF16

### Baseline
TBD — running calibration first

### Noise Floor
TBD — need 3 baseline runs with different seeds

### Scaling Decision
Target: 5-second experiments for screening, 10-20s for validation.
Need calibration to determine tokens_per_second.

### Experiment Budget
Unlimited batches of 5 experiments each.

### Strategy
1. Calibrate tokens/second throughput
2. Run 3 baseline experiments (seeds 42, 123, 7) to measure noise floor
3. LR sweep (most impactful HP) at 5s scale
4. Validate top LR configs at 15s scale
5. Continue through: schedule → weight_decay → momentum → batch_size → architecture
6. Track scaling behavior: does optimal LR change with duration?

### Banlist
(none yet)
