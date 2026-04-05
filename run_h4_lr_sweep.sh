#!/bin/bash
# H4: Learning rate magnitude sweep.
# Baseline: relu+cosine, muon_lr=0.024 → val_loss=6.0603 at 2M tokens
# Goal: find if a different LR is better with relu+cosine.
set -e
cd /workspace/llm-research-kit

TOKENS=2000000
mkdir -p experiments/h4-lr-sweep/runs/{lr-half,lr-default,lr-150pct}

echo "H4: LR Magnitude Sweep (relu+cosine, 2M tokens)"
echo "Reference: relu+cosine, muon_lr=0.024 → ~6.0603"
echo ""

for CFG in \
    "configs.lr_magnitude_configs.ReluCosine_LR_Half:lr-half" \
    "configs.lr_magnitude_configs.ReluCosine_LR_Default:lr-default" \
    "configs.lr_magnitude_configs.ReluCosine_LR_150pct:lr-150pct"; do
    CLASS="${CFG%%:*}"
    NAME="${CFG##*:}"
    DIR="experiments/h4-lr-sweep/runs/${NAME}"
    mkdir -p "$DIR"
    echo "--- $NAME ($CLASS) ---"
    python3 train_llm.py \
        --config_class "$CLASS" \
        --train_tokens "$TOKENS" \
        --output_dir "$DIR" \
        --compile false \
        2>&1 | tee "${DIR}/run.log"
    echo "Done: $NAME"
done

echo ""
echo "H4 complete."
python3 -c "
import json, os

runs = {
    'relu+cosine LR=0.012': 'experiments/h4-lr-sweep/runs/lr-half',
    'relu+cosine LR=0.024': 'experiments/h4-lr-sweep/runs/lr-default',
    'relu+cosine LR=0.036': 'experiments/h4-lr-sweep/runs/lr-150pct',
}
print(f'{\"Config\":<28} {\"val_loss\":>9} {\"val_acc\":>9}')
print('-'*50)
for name, path in runs.items():
    mf = os.path.join(path, 'metrics.json')
    try:
        d = json.load(open(mf))
        fm = d['final_metrics']
        print(f'{name:<28} {fm[\"val_loss\"]:>9.4f} {fm[\"val_accuracy\"]:>9.4f}')
    except:
        print(f'{name:<28} (missing)')
"
