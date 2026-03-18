# Optimization Plan

## Setup
- GPU: L40S 46GB | Mode: eager (no torch.compile) | Throughput: ~51K tps
- Model: 88M MinimalLLM | Muon+AdamW | BF16
- Seed: 42 (fixed, deterministic)
- **5s: 250K tokens (~15 steps)** ← primary screening tier
- 10s: 500K tokens (~31 steps) — validate 5s winners
- 20s: 1M tokens (~62 steps) — confirm scaling

## Baseline (20s tier)
val_loss=6.4691 (default config)

## Current Best (20s tier)
val_loss=6.3934 (muon_lr=0.006, adamw_lr=0.0015)

## Key Findings So Far
- Lower LR is better at 20s: 0.006 > 0.012 > 0.024 > 0.048 > 0.096 > 0.192
- Reversal found: 0.004 (6.4365) worse than 0.006 (6.3934) — peak is near 0.006
- 0.002 much worse (6.6054) — too slow

## Strategy
1. **Re-run LR sweep at 5s** — does the optimal LR shift with shorter training?
2. Move to LR schedule (cosine/linear + warmup)
3. Weight decay, momentum, batch size
4. Compare optimal configs across 5s/10s/20s for scaling laws

## Banlist
- muon_lr >= 0.048 (worse than baseline at all durations)
