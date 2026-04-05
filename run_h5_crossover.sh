#!/bin/bash
# H5: Find crossover token count between relu and squared_relu (constant LR).
# At 2M tokens: relu (6.060) < squared_relu (6.108) — relu better
# At 8M tokens: relu (4.966) > squared_relu (4.921) — squared_relu better
# Goal: find the crossover between 2M and 8M (test at 4M tokens).
set -e
cd /workspace/llm-research-kit

TOKENS=4000000
mkdir -p experiments/h5-crossover/runs/{squaredrelu-4M,relu-4M}

echo "H5: Crossover Search at 4M tokens"
echo "Known: relu wins at 2M, squared_relu wins at 8M"
echo ""

echo "--- squared_relu + constant LR (4M) ---"
python3 train_llm.py \
    --config_class configs.titan_x_config.TitanXConfig \
    --activation squared_relu \
    --train_tokens $TOKENS \
    --output_dir experiments/h5-crossover/runs/squaredrelu-4M \
    --compile false \
    2>&1 | tee experiments/h5-crossover/runs/squaredrelu-4M/run.log

echo "--- relu + constant LR (4M) ---"
python3 train_llm.py \
    --config_class configs.best_combos.ReluConstantConfig \
    --train_tokens $TOKENS \
    --output_dir experiments/h5-crossover/runs/relu-4M \
    --compile false \
    2>&1 | tee experiments/h5-crossover/runs/relu-4M/run.log

echo ""
echo "H5 complete."
python3 -c "
import json, os

runs = {
    'squared_relu, 2M (ref)': 'experiments/activation-discovery/runs/baseline-squaredrelu-2M',
    'relu, 2M (ref)':         'experiments/activation-discovery/runs/test-relu-2M',
    'squared_relu, 4M':       'experiments/h5-crossover/runs/squaredrelu-4M',
    'relu, 4M':               'experiments/h5-crossover/runs/relu-4M',
    'squared_relu, 8M (ref)': 'experiments/activation-discovery/runs/baseline-squaredrelu',
    'relu, 8M (ref)':         'experiments/h3-validation/runs/relu-constant-8M',
}
print(f'{\"Config\":<30} {\"val_loss\":>9}')
print('-'*42)
for name, path in runs.items():
    mf = os.path.join(path, 'metrics.json')
    try:
        d = json.load(open(mf))
        print(f'{name:<30} {d[\"final_metrics\"][\"val_loss\"]:>9.4f}')
    except:
        print(f'{name:<30} (missing)')
"
