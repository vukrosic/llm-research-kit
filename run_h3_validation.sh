#!/bin/bash
# H3: Validate relu gain at 8M tokens.
# Also test relu+cosine at 8M to see if cosine benefit appears at longer runs.
#
# Baselines:
#   squared_relu+constant, 8M tokens → val_loss=4.9214  (from earlier run)
#
# Predictions:
#   relu+constant, 8M   → ~4.87  (relu gain ~0.049 should persist)
#   relu+cosine, 8M     → ~4.84  (cosine may help more at longer training)
set -e
cd /workspace/llm-research-kit

TOKENS=8000000
mkdir -p experiments/h3-validation/runs/{relu-constant-8M,relu-cosine-8M}

echo "H3: Validation at 8M tokens"
echo "Reference: squared_relu+constant = 4.9214"
echo ""

echo "--- relu + constant LR (8M) ---"
python3 train_llm.py \
    --config_class configs.best_combos.ReluConstantConfig \
    --train_tokens $TOKENS \
    --output_dir experiments/h3-validation/runs/relu-constant-8M \
    --compile false \
    2>&1 | tee experiments/h3-validation/runs/relu-constant-8M/run.log

echo "--- relu + cosine LR (8M) ---"
python3 train_llm.py \
    --config_class configs.best_combos.ReluCosineConfig \
    --train_tokens $TOKENS \
    --output_dir experiments/h3-validation/runs/relu-cosine-8M \
    --compile false \
    2>&1 | tee experiments/h3-validation/runs/relu-cosine-8M/run.log

echo ""
echo "H3 complete."
python3 -c "
import json
runs = {
    'squared_relu+constant 8M (ref)': 'experiments/activation-discovery/runs/baseline-squaredrelu',
    'relu+constant 8M':               'experiments/h3-validation/runs/relu-constant-8M',
    'relu+cosine 8M':                 'experiments/h3-validation/runs/relu-cosine-8M',
}
print(f'{\"Config\":<34} {\"val_loss\":>9}')
print('-'*45)
for name, path in runs.items():
    try:
        d = json.load(open(f'{path}/metrics.json'))
        print(f'{name:<34} {d[\"final_metrics\"][\"val_loss\"]:>9.4f}')
    except:
        print(f'{name:<34} (missing)')
"
