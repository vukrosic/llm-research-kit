#!/bin/bash
# Run LR schedule hypothesis (H1) experiments.
# Baseline already done: baseline-squaredrelu-2M (val_loss=6.1084)
# Runs the 2 new configs and reports results.

set -e
cd /workspace/llm-research-kit

TOKENS=2000000

echo "=============================="
echo "H1: LR Schedule Experiment"
echo "Baseline val_loss: 6.1084"
echo "=============================="

for CFG in "configs.lr_schedule_configs.TitanXWarmupConfig:warmup-constant" \
           "configs.lr_schedule_configs.TitanXCosineConfig:warmup-cosine"; do
    CLASS="${CFG%%:*}"
    NAME="${CFG##*:}"
    DIR="experiments/lr-schedule-h1/runs/${NAME}"
    mkdir -p "$DIR"
    echo ""
    echo "Running: $NAME ($CLASS)"
    python3 train_llm.py \
        --config_class "$CLASS" \
        --activation squared_relu \
        --train_tokens "$TOKENS" \
        --output_dir "$DIR" \
        --compile false \
        2>&1 | tee "${DIR}/run.log"
    echo "Done: $NAME"
done

echo ""
echo "All H1 runs complete. Run: python3 analyze_lr_results.py"
