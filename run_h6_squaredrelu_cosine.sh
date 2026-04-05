#!/bin/bash
# H6: Does cosine LR benefit squared_relu as much as it benefits relu at 8M tokens?
# Known: squared_relu+constant = 4.9214 at 8M
#        relu+cosine = 4.9159 at 8M (new best)
# Question: can squared_relu+cosine beat relu+cosine?
set -e
cd /workspace/llm-research-kit

TOKENS=8000000
mkdir -p experiments/h6-squaredrelu-cosine/runs/squaredrelu-cosine-8M

echo "H6: squared_relu + cosine LR at 8M tokens"
echo "Reference: relu+cosine = 4.9159, squared_relu+constant = 4.9214"
echo ""

python3 train_llm.py \
    --config_class configs.lr_schedule_configs.TitanXCosineConfig \
    --activation squared_relu \
    --train_tokens $TOKENS \
    --output_dir experiments/h6-squaredrelu-cosine/runs/squaredrelu-cosine-8M \
    --compile false \
    2>&1 | tee experiments/h6-squaredrelu-cosine/runs/squaredrelu-cosine-8M/run.log

echo ""
echo "H6 complete."
python3 -c "
import json, os
runs = {
    'squared_relu+constant 8M':       'experiments/activation-discovery/runs/baseline-squaredrelu',
    'relu+cosine 8M (best so far)':   'experiments/h3-validation/runs/relu-cosine-8M',
    'squared_relu+cosine 8M':         'experiments/h6-squaredrelu-cosine/runs/squaredrelu-cosine-8M',
}
print(f'{\"Config\":<38} {\"val_loss\":>9}')
print('-'*50)
for name, path in runs.items():
    mf = os.path.join(path, 'metrics.json')
    try:
        d = json.load(open(mf))
        print(f'{name:<38} {d[\"final_metrics\"][\"val_loss\"]:>9.4f}')
    except:
        print(f'{name:<38} (missing)')
"
