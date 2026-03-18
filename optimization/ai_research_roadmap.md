# AI Research Roadmap

## Goal

Build a 1B dense GPT that is competitive with public 1B baselines and use the optimization process itself as proof that the auto-research loop works.

Deadlines:
- March 25: produce one publishable, defensible result for professor outreach
- May 1: finish a concrete research portfolio before visa expiry

Current constraints:
- 1x L40S 48GB
- limited budget, so proxy quality matters more than search breadth
- every experiment must be trackable, comparable, and publishable

## Source Of Truth

- `optimization/experiments.jsonl`: append-only run ledger
- `optimization/decisions.md`: one decision note per completed sweep
- `optimization/analyze_sweep.py`: rank runs by mean validation loss after warmup, then final validation loss, then tokens/sec

Required fields in `experiments.jsonl`:
- `exp_id`
- `phase`
- `question`
- `model_config`
- `dataset_path`
- `seed`
- `changes`
- `train_tokens`
- `train_seconds`
- `status`
- `val_loss`
- `train_loss`
- `tokens_per_second`
- `actual_steps`
- `metrics_path`
- `checkpoint_path`
- `notes`

Status labels:
- `done`: evidence exists and is recorded
- `ready to run`: runnable now with current code
- `blocked by infra`: do not run until tooling is added

## Ground Truth As Of March 18, 2026

### 1B feasibility

Status: `done`

Evidence from `results/exp0_1b_feasibility/metrics.json`:
- completed without OOM
- `501,760` tokens seen
- `245` steps
- `93.3s` active training time
- `160.6s` wall time
- approximate throughput: `5.38k` tokens/sec during active training
- peak VRAM was not captured and must be recorded in the next 1B run

Implication:
- 1B training is feasible on this GPU
- throughput is no longer unknown
- future 1B confirmation runs should always log peak GPU memory

### Existing LR evidence

Status: `done`

Observed ranking movement:
- 5s: best around `0.007-0.008`
- 10s: best around `0.007`
- 20s: best around `0.006-0.007`
- 45s validation: `0.016` slightly beats `0.012`, `0.010` trails

Interpretation:
- the ranking has moved multiple times already
- 45s evidence is useful but not decisive
- LR is not locked

## Phase 0: Research Hygiene And Ground Truth

Status: `done`

Tasks:
1. Sync existing queue files into `optimization/experiments.jsonl`.
2. Record one decision note for the 5s/10s/20s/45s LR evidence.
3. Carry forward the completed 1B feasibility result.
4. Explicitly track that the next 1B run must capture peak VRAM.

Acceptance criteria:
- `experiments.jsonl` is non-empty
- `decisions.md` contains a completed LR decision note
- roadmap no longer claims 1B fit or throughput are unknown

## Phase 1: Lock The LR Proxy Before More Tuning

Status: `ready to run`

Question:
- what is the shortest proxy budget that preserves LR ranking well enough to choose candidates for 1B confirmation?

Fixed setup:
- config: `LLMConfig`
- dataset: `./processed_data/pretrain_1B`
- `batch_size=4`
- `gradient_accumulation_steps=1`
- default 88M tokenizer and sequence length
- seeds: `42`, `137`

Candidate set:
- `0.008 / 0.002`
- `0.012 / 0.003`
- `0.016 / 0.004`
- `0.024 / 0.006`

Run layout:
- 8 runs total
- save to `results/exp1_lr_8M/<exp_id>/`

Commands:

```bash
python train_llm.py --muon_lr 0.008 --adamw_lr 0.002 --train_tokens 8000000 --batch_size 4 --gradient_accumulation_steps 1 --seed 42 --dataset_path ./processed_data/pretrain_1B --output_dir results/exp1_lr_8M/lr0.008_s42
python train_llm.py --muon_lr 0.012 --adamw_lr 0.003 --train_tokens 8000000 --batch_size 4 --gradient_accumulation_steps 1 --seed 42 --dataset_path ./processed_data/pretrain_1B --output_dir results/exp1_lr_8M/lr0.012_s42
python train_llm.py --muon_lr 0.016 --adamw_lr 0.004 --train_tokens 8000000 --batch_size 4 --gradient_accumulation_steps 1 --seed 42 --dataset_path ./processed_data/pretrain_1B --output_dir results/exp1_lr_8M/lr0.016_s42
python train_llm.py --muon_lr 0.024 --adamw_lr 0.006 --train_tokens 8000000 --batch_size 4 --gradient_accumulation_steps 1 --seed 42 --dataset_path ./processed_data/pretrain_1B --output_dir results/exp1_lr_8M/lr0.024_s42
python train_llm.py --muon_lr 0.008 --adamw_lr 0.002 --train_tokens 8000000 --batch_size 4 --gradient_accumulation_steps 1 --seed 137 --dataset_path ./processed_data/pretrain_1B --output_dir results/exp1_lr_8M/lr0.008_s137
python train_llm.py --muon_lr 0.012 --adamw_lr 0.003 --train_tokens 8000000 --batch_size 4 --gradient_accumulation_steps 1 --seed 137 --dataset_path ./processed_data/pretrain_1B --output_dir results/exp1_lr_8M/lr0.012_s137
python train_llm.py --muon_lr 0.016 --adamw_lr 0.004 --train_tokens 8000000 --batch_size 4 --gradient_accumulation_steps 1 --seed 137 --dataset_path ./processed_data/pretrain_1B --output_dir results/exp1_lr_8M/lr0.016_s137
python train_llm.py --muon_lr 0.024 --adamw_lr 0.006 --train_tokens 8000000 --batch_size 4 --gradient_accumulation_steps 1 --seed 137 --dataset_path ./processed_data/pretrain_1B --output_dir results/exp1_lr_8M/lr0.024_s137
```

Scoring rule:
1. lowest mean validation loss after warmup across milestones
2. lowest final validation loss
3. fastest tokens/sec

Analysis command:

```bash
python optimization/analyze_sweep.py results/exp1_lr_8M
```

Promotion rule:
- promote the top two candidates only if top-two membership matches across both seeds

Acceptance criteria:
- eight runs complete
- one ranked summary exists
- if top-two membership is unstable, LR remains unlocked

## Phase 2: Confirm LR At A Longer Budget Only If Needed

Status: `ready to run`

Trigger:
- run only if Phase 1 top-two membership is unstable or the winner is within `<=0.01` val loss of the runner-up

Tasks:
1. take the top three Phase 1 candidates to `20M` tokens at seed `42`
2. if top-two separation is still within `<=0.01`, rerun the top two at seed `137`
3. lock LR only if the same top-two set persists from `8M` to `20M`
4. if no stable winner emerges, stop recipe sweeps and publish proxy instability as the result

Command template:

```bash
python train_llm.py --muon_lr <MUON_LR> --adamw_lr <ADAMW_LR> --train_tokens 20000000 --batch_size 4 --gradient_accumulation_steps 1 --seed <SEED> --dataset_path ./processed_data/pretrain_1B --output_dir results/exp2_lr_20M/<exp_id>
```

Acceptance criteria:
- either LR is locked with written evidence, or proxy instability is explicitly declared

## Phase 3: Validate 88M To 1B Transfer Immediately

Status: `ready to run` after LR lock

Question:
- does the 88M LR winner transfer to 1B at all?

Tasks:
1. run the locked 88M winner at 1B for `500k` tokens
2. run the 88M runner-up at 1B for `500k` tokens
3. save to `results/exp3_1b_transfer/<candidate>/`
4. record final val loss, training time, wall time, tokens seen, and peak GPU memory
5. if the two runs tie within `<=0.01`, run both again at `1M` tokens
6. if transfer fails, rewrite the remaining roadmap to use 1B-first screening

Command template:

```bash
python train_llm.py --config_class configs.llm_config.OneBConfig --muon_lr <MUON_LR> --adamw_lr <ADAMW_LR> --train_tokens 500000 --seed <SEED> --dataset_path ./processed_data/pretrain_1B --output_dir results/exp3_1b_transfer/<exp_id>
```

Acceptance criteria:
- yes/no answer on whether the 88M LR proxy is trustworthy

## Phase 4: Unblock Recipe Sweeps

Status: `done`

Completed infra:
- `train_llm.py` supports overrides for `warmup_ratio`, `weight_decay`, `schedule_type`, and `muon_momentum`
- training metrics now include experiment config, throughput, and peak GPU memory
- `optimization/analyze_sweep.py` can rank sweep outputs

Check command:

```bash
python train_llm.py --help
```

Dry-run validation target:
- confirm override values appear in the logged config before starting any Phase 5 sweep

## Phase 5: Tune The Recipe One Knob At A Time

Status: `ready to run` after LR lock

Rules:
- do not change multiple knobs at once
- use `8M` as the first budget
- use seeds `42` and `137`
- treat `<=0.01` val-loss gaps as ties unless longer confirmation breaks them

### Task Set A: Warmup Sweep

Values:
- `0.0`
- `0.005`
- `0.01`
- `0.02`

Command template:

```bash
python train_llm.py --muon_lr <LOCKED_MUON_LR> --adamw_lr <LOCKED_ADAMW_LR> --warmup_ratio <VALUE> --train_tokens 8000000 --batch_size 4 --gradient_accumulation_steps 1 --seed <SEED> --dataset_path ./processed_data/pretrain_1B --output_dir results/exp4_warmup/<exp_id>
```

### Task Set B: Weight Decay Sweep

Values:
- `0.05`
- `0.1`
- `0.15`
- `0.2`

Command template:

```bash
python train_llm.py --muon_lr <LOCKED_MUON_LR> --adamw_lr <LOCKED_ADAMW_LR> --warmup_ratio <LOCKED_WARMUP> --weight_decay <VALUE> --train_tokens 8000000 --batch_size 4 --gradient_accumulation_steps 1 --seed <SEED> --dataset_path ./processed_data/pretrain_1B --output_dir results/exp5_weight_decay/<exp_id>
```

### Task Set C: Muon Momentum

Run only after LR, warmup, and weight decay are locked.

Values:
- `0.90`
- `0.95`
- `0.98`

Command template:

```bash
python train_llm.py --muon_lr <LOCKED_MUON_LR> --adamw_lr <LOCKED_ADAMW_LR> --warmup_ratio <LOCKED_WARMUP> --weight_decay <LOCKED_WEIGHT_DECAY> --muon_momentum <VALUE> --train_tokens 8000000 --batch_size 4 --gradient_accumulation_steps 1 --seed <SEED> --dataset_path ./processed_data/pretrain_1B --output_dir results/exp6_momentum/<exp_id>
```

Acceptance criteria:
- final recipe is a tuple, not prose:
  - `muon_lr`
  - `adamw_lr`
  - `warmup_ratio`
  - `weight_decay`
  - `muon_momentum`
  - `schedule_type`

## Phase 6: Architecture Work Only After The Recipe Is Stable

Status: `blocked by recipe lock`

Execution order:
1. `ARCH-001` SwiGLU
2. `ARCH-003` embedding scale removal
3. `ARCH-004` residual scaling
4. `ARCH-002` residual attention

Rule:
- every architecture change must be behind a config flag
- run a short sanity check first
- run an `8M` proxy second
- confirm at 1B only if it wins by at least `0.5%` relative val-loss improvement or clearly improves stability

## Required Checks Before Declaring A Sweep Complete

1. every completed run has a matching `experiments.jsonl` line
2. every completed sweep has a `decisions.md` entry
3. the ranked output from `optimization/analyze_sweep.py` is saved or copied into the decision note
4. if a sweep fails to transfer, the roadmap is updated before the next sweep starts

## Immediate Next Actions

1. Run Phase 1 8M LR sweep with seeds `42` and `137`
2. Rank the results with `optimization/analyze_sweep.py`
3. Either lock LR or trigger Phase 2
4. Run Phase 3 immediately after LR lock
