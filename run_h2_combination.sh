#!/bin/bash
# H2: Validate that relu + cosine improvements combine additively.
# Baseline: squared_relu + constant LR = 6.1084
# Expected: relu + cosine ≈ 6.044 (sum of gains)
set -e
cd /workspace/llm-research-kit

TOKENS=2000000
mkdir -p experiments/h2-combination/runs/{relu-constant,relu-cosine}

echo "H2: Combination Test"
echo "Baseline: 6.1084 (squared_relu, constant LR)"
echo ""

echo "--- relu + constant LR ---"
python3 train_llm.py \
    --config_class configs.best_combos.ReluConstantConfig \
    --train_tokens $TOKENS \
    --output_dir experiments/h2-combination/runs/relu-constant \
    --compile false \
    2>&1 | tee experiments/h2-combination/runs/relu-constant/run.log

echo "--- relu + cosine LR ---"
python3 train_llm.py \
    --config_class configs.best_combos.ReluCosineConfig \
    --train_tokens $TOKENS \
    --output_dir experiments/h2-combination/runs/relu-cosine \
    --compile false \
    2>&1 | tee experiments/h2-combination/runs/relu-cosine/run.log

echo ""
echo "H2 complete."
python3 -c "
import json
runs = {
    'squared_relu+constant (base)': 'experiments/activation-discovery/runs/baseline-squaredrelu-2M',
    'relu+constant':                'experiments/h2-combination/runs/relu-constant',
    'relu+cosine (predicted best)': 'experiments/h2-combination/runs/relu-cosine',
}
print(f'{\"Config\":<32} {\"val_loss\":>9}')
print('-'*43)
for name, path in runs.items():
    try:
        d = json.load(open(f'{path}/metrics.json'))
        print(f'{name:<32} {d[\"final_metrics\"][\"val_loss\"]:>9.4f}')
    except:
        print(f'{name:<32} (missing)')
"
