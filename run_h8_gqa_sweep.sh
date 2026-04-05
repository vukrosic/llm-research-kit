#!/bin/bash
# H8: GQA ratio sweep — is n_kv_heads=4 (2:1 ratio) optimal with squared_relu+cosine?
# Using 8M tokens for fair comparison with established H6 baseline.
# Tests: full attention (8 KV), default (4 KV), aggressive GQA (2 KV), MQA (1 KV).
set -e
cd /workspace/llm-research-kit

TOKENS=8000000
mkdir -p experiments/h8-gqa/runs/{full-attn-8M,default-gqa-8M,agg-gqa-8M,mqa-8M}

echo "H8: GQA Ratio Sweep (squared_relu+cosine, 8M tokens)"
echo "Reference: squared_relu+cosine, n_kv=4 = 4.8956"
echo ""

for CFG in \
    "configs.gqa_configs.FullAttentionConfig:full-attn-8M" \
    "configs.gqa_configs.AggressiveGQAConfig:agg-gqa-8M" \
    "configs.gqa_configs.MQAConfig:mqa-8M"; do
    CLASS="${CFG%%:*}"
    NAME="${CFG##*:}"
    DIR="experiments/h8-gqa/runs/${NAME}"
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
echo "H8 complete."
python3 -c "
import json, os
runs = {
    'default (n_kv=4) 8M ref': 'experiments/h6-squaredrelu-cosine/runs/squaredrelu-cosine-8M',
    'full-attn (n_kv=8) 8M':   'experiments/h8-gqa/runs/full-attn-8M',
    'agg-gqa (n_kv=2) 8M':     'experiments/h8-gqa/runs/agg-gqa-8M',
    'mqa (n_kv=1) 8M':         'experiments/h8-gqa/runs/mqa-8M',
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
