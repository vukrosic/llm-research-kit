#!/bin/bash
# H10: Fine-grained momentum sweep — is 0.90 optimal or can lower momentum help more?
# H9 showed sharp asymmetry: 0.90 >> 0.95 >> 0.98 >> 0.99
# Testing: 0.80, 0.85, 0.92 to pin down the optimum.
set -e
cd /workspace/llm-research-kit

TOKENS=8000000
mkdir -p experiments/h10-momentum-fine/runs/{momentum-080-8M,momentum-085-8M,momentum-092-8M}

echo "H10: Fine-Grained Muon Momentum Sweep (full-attn+squared_relu+cosine, 8M tokens)"
echo "Reference: momentum=0.90 → val_loss=4.7952"
echo ""

for CFG in \
    "configs.momentum_configs.Momentum080Config:momentum-080-8M" \
    "configs.momentum_configs.Momentum085Config:momentum-085-8M" \
    "configs.momentum_configs.Momentum092Config:momentum-092-8M"; do
    CLASS="${CFG%%:*}"
    NAME="${CFG##*:}"
    DIR="experiments/h10-momentum-fine/runs/${NAME}"
    mkdir -p "$DIR"
    echo "--- $NAME ---"
    python3 train_llm.py \
        --config_class "$CLASS" \
        --train_tokens "$TOKENS" \
        --output_dir "$DIR" \
        --compile false \
        2>&1 | tee "${DIR}/run.log"
    echo "Done: $NAME"
done

echo ""
echo "H10 complete."
python3 -c "
import json, os
runs = {
    'momentum=0.80':        'experiments/h10-momentum-fine/runs/momentum-080-8M',
    'momentum=0.85':        'experiments/h10-momentum-fine/runs/momentum-085-8M',
    'momentum=0.90 (ref)':  'experiments/h9-momentum/runs/momentum-low-8M',
    'momentum=0.92':        'experiments/h10-momentum-fine/runs/momentum-092-8M',
    'momentum=0.95 (orig)': 'experiments/h8-gqa/runs/full-attn-8M',
}
ref = 4.7952
print(f'{\"Config\":<28} {\"val_loss\":>9} {\"vs 0.90\":>9}')
print('-'*50)
for name, path in runs.items():
    mf = os.path.join(path, 'metrics.json')
    try:
        d = json.load(open(mf))
        v = d['final_metrics']['val_loss']
        print(f'{name:<28} {v:>9.4f} {v-ref:>+9.4f}')
    except:
        print(f'{name:<28} (missing)')
"
