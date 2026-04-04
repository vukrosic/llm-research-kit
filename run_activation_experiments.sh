#!/bin/bash
# Run all activation experiments for the activation-discovery sweep.
# Runs sequentially to avoid GPU contention.
# Usage: ./run_activation_experiments.sh [--skip-baseline]

set -e
cd /workspace/llm-research-kit

ACTIVATIONS=("squared_relu" "gelu" "silu" "swiglu" "relu")
CONFIG="configs.titan_x_config.TitanXConfig"
TOKENS=2000000

for ACT in "${ACTIVATIONS[@]}"; do
    NAME=$([ "$ACT" = "squared_relu" ] && echo "baseline-squaredrelu-2M" || echo "test-${ACT}-2M")
    DIR="experiments/activation-discovery/runs/${NAME}"
    mkdir -p "$DIR"
    echo "=============================="
    echo "Running activation: $ACT"
    echo "Output: $DIR"
    echo "=============================="
    python3 train_llm.py \
        --config_class "$CONFIG" \
        --activation "$ACT" \
        --train_tokens "$TOKENS" \
        --output_dir "$DIR" \
        --compile false \
        2>&1 | tee "${DIR}/run.log"
    echo "Done: $ACT"
done

echo "All activation experiments complete."
