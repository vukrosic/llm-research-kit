# Experiment Checklist

## Phase 0: Feasibility (~5 min)
- [ ] **EXP-0: 1B on L40S feasibility**
  - [ ] Run OneBConfig for 500K tokens
  - [ ] Record: tokens/sec, peak memory, OOM or not
  - [ ] Decision: can we use 1B on this GPU?

## Phase 1: LR Stabilization (~30 min)
- [ ] **EXP-1: LR sweep at 8M tokens (88M model)**
  - [ ] Run muon_lr=0.008, adamw_lr=0.002, seed 42
  - [ ] Run muon_lr=0.012, adamw_lr=0.003, seed 42
  - [ ] Run muon_lr=0.018, adamw_lr=0.0045, seed 42
  - [ ] Run muon_lr=0.024, adamw_lr=0.006, seed 42
  - [ ] Compare rankings vs 80s (5M tok) data
  - [ ] Decision: did ranking stabilize? (same top-2 at 5M and 8M?)
- [ ] **EXP-2: LR at 20M tokens (only if EXP-1 ranking shifted)**
  - [ ] Run top 3 from EXP-1 at 20M tokens
  - [ ] Compare rankings vs EXP-1
  - [ ] Decision: lock LR

## Phase 2: Transfer Validation (~15 min)
- [ ] **EXP-3: 88M → 1B transfer check**
  - [ ] Run 88M winner at 1B, 500K-1M tokens
  - [ ] Run 88M runner-up at 1B, 500K-1M tokens
  - [ ] Decision: does 88M ranking predict 1B? Trust proxy or not?
- [ ] **Post: "Can 88M predict 1B?" results**

## Phase 3: Recipe Tuning (~30 min)
- [ ] **EXP-4: Warmup sweep (at locked LR)**
  - [ ] Run warmup_ratio=0.0, 8M tokens
  - [ ] Run warmup_ratio=0.005, 8M tokens
  - [ ] Run warmup_ratio=0.01, 8M tokens
  - [ ] Run warmup_ratio=0.02, 8M tokens
  - [ ] Decision: lock warmup
- [ ] **EXP-5: Weight decay sweep (at locked LR + warmup)**
  - [ ] Run weight_decay=0.05, 8M tokens
  - [ ] Run weight_decay=0.1, 8M tokens
  - [ ] Run weight_decay=0.15, 8M tokens
  - [ ] Run weight_decay=0.2, 8M tokens
  - [ ] Decision: lock weight decay
- [ ] **Lock full optimization recipe**
- [ ] **Post: "Best training recipe for 1B model"**

## Phase 4: Architecture (~1-2 hours)
- [ ] **ARCH-001: SwiGLU FFN**
  - [ ] Implement behind config flag
  - [ ] Sanity check: 88M, 1M tokens
  - [ ] Proxy: 88M, 8M tokens, compare vs baseline
  - [ ] Decision: ≥0.5% val loss improvement?
  - [ ] If yes: confirm at 1B
- [ ] **ARCH-002: Residual attention**
  - [ ] Implement behind config flag
  - [ ] Sanity check: 88M, 1M tokens
  - [ ] Proxy: 88M, 8M tokens, compare vs baseline
  - [ ] Decision: ≥0.5% val loss improvement?
  - [ ] If yes: confirm at 1B
- [ ] **ARCH-003: Remove embedding scale (× √d_model)**
  - [ ] Test with and without at 88M, 8M tokens
  - [ ] Decision: keep or remove?
- [ ] **ARCH-004: Residual scaling (1/√n_layers)**
  - [ ] Implement behind config flag
  - [ ] Proxy: 88M, 8M tokens
  - [ ] Decision: helps stability?
- [ ] **Post: architecture experiment results**

## Phase 5: Full Training
- [ ] **Full 1B training run**
  - [ ] Lock recipe + architecture
  - [ ] Run 1B at 20B tokens with best config
  - [ ] Evaluate on benchmarks (HellaSwag, ARC, PIQA)
  - [ ] Post: live training progress thread
- [ ] **Novita grant request**
  - [ ] Compile engagement report (all post views)
  - [ ] Send $1,000 ask with data

## Content Milestones
- [ ] Post: LR ranking shifts with duration (existing data) — **today**
- [ ] Post: "Does LR ranking stabilize at 8M tokens?" — after EXP-1
- [ ] Post: "Can 88M predict 1B?" — after EXP-3
- [ ] Post: "Best recipe for 1B on $X compute" — after Phase 3
- [ ] Post: architecture results — after Phase 4
- [ ] Post: full training progress — during Phase 5
