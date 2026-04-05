#!/bin/bash
# H7: Validate squared_relu+cosine improvement holds at 20M tokens.
# At 8M: squared_relu+cosine = 4.8956 vs squared_relu+constant = 4.9214 (−0.026)
# Question: does the cosine schedule advantage persist or grow at scale?
set -e
cd /workspace/llm-research-kit

TOKENS=20000000
mkdir -p experiments/h7-scale/runs/{squaredrelu-constant-20M,squaredrelu-cosine-20M}

echo "H7: Scale Validation at 20M tokens"
echo "Expected 8M trend: squared_relu+cosine wins by ~0.026"
echo ""

echo "--- squared_relu + constant LR (20M) ---"
python3 train_llm.py \
    --config_class configs.titan_x_config.TitanXConfig \
    --activation squared_relu \
    --train_tokens $TOKENS \
    --output_dir experiments/h7-scale/runs/squaredrelu-constant-20M \
    --compile false \
    2>&1 | tee experiments/h7-scale/runs/squaredrelu-constant-20M/run.log

echo ""
echo "--- squared_relu + cosine LR (20M) ---"
python3 train_llm.py \
    --config_class configs.final_config.BestConfig \
    --train_tokens $TOKENS \
    --output_dir experiments/h7-scale/runs/squaredrelu-cosine-20M \
    --compile false \
    2>&1 | tee experiments/h7-scale/runs/squaredrelu-cosine-20M/run.log

echo ""
echo "H7 complete."
python3 -c "
import json, os
runs = {
    'squaredrelu+constant 8M (ref)':   'experiments/activation-discovery/runs/baseline-squaredrelu',
    'squaredrelu+cosine 8M (ref)':     'experiments/h6-squaredrelu-cosine/runs/squaredrelu-cosine-8M',
    'squaredrelu+constant 20M':        'experiments/h7-scale/runs/squaredrelu-constant-20M',
    'squaredrelu+cosine 20M':          'experiments/h7-scale/runs/squaredrelu-cosine-20M',
}
print(f'{\"Config\":<40} {\"val_loss\":>9}')
print('-'*52)
for name, path in runs.items():
    mf = os.path.join(path, 'metrics.json')
    try:
        d = json.load(open(mf))
        print(f'{name:<40} {d[\"final_metrics\"][\"val_loss\"]:>9.4f}')
    except:
        print(f'{name:<40} (missing)')
"
