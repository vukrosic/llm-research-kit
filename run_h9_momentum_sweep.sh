#!/bin/bash
# H9: Muon momentum sweep — is momentum=0.95 optimal with full-attn+squared_relu+cosine?
# Using 8M tokens. Tests momentum ∈ {0.90, 0.98, 0.99} vs reference (0.95 = 4.8785).
set -e
cd /workspace/llm-research-kit

TOKENS=8000000
mkdir -p experiments/h9-momentum/runs/{momentum-low-8M,momentum-high-8M,momentum-veryhigh-8M}

echo "H9: Muon Momentum Sweep (full-attn+squared_relu+cosine, 8M tokens)"
echo "Reference: momentum=0.95, full-attn, val_loss=4.8785"
echo ""

for CFG in \
    "configs.momentum_configs.MomentumLowConfig:momentum-low-8M" \
    "configs.momentum_configs.MomentumHighConfig:momentum-high-8M" \
    "configs.momentum_configs.MomentumVeryHighConfig:momentum-veryhigh-8M"; do
    CLASS="${CFG%%:*}"
    NAME="${CFG##*:}"
    DIR="experiments/h9-momentum/runs/${NAME}"
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
echo "H9 complete."
python3 -c "
import json, os
runs = {
    'default (momentum=0.95) 8M ref': 'experiments/h8-gqa/runs/full-attn-8M',
    'low (momentum=0.90) 8M':        'experiments/h9-momentum/runs/momentum-low-8M',
    'high (momentum=0.98) 8M':       'experiments/h9-momentum/runs/momentum-high-8M',
    'veryhigh (momentum=0.99) 8M':   'experiments/h9-momentum/runs/momentum-veryhigh-8M',
}
print(f'{\"Config\":<35} {\"val_loss\":>9}')
print('-'*47)
for name, path in runs.items():
    mf = os.path.join(path, 'metrics.json')
    try:
        d = json.load(open(mf))
        print(f'{name:<35} {d[\"final_metrics\"][\"val_loss\"]:>9.4f}')
    except:
        print(f'{name:<35} (missing)')
"
