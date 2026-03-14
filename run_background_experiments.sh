#!/bin/bash
# run_background_experiments.sh
# Runs the Generation 3 6M token experiments in the background.

export PYTHONUNBUFFERED=1

nohup python run_ablations.py \
    --tokens 6000000 \
    --experiments \
        new_full_mha \
        new_attn_bias \
        new_ffn_wide \
        new_small_embed_init \
        deepnorm_sweep_05 \
        deepnorm_sweep_03 \
        combo_deepnorm_bilinear \
        combo_deepnorm_full_mha \
        value_norm \
        layer_scale_001 \
    > experiments_background.log 2>&1 &

echo "🚀 Experiments started in background with PID $!"
echo "📄 Logs are being written to experiments_background.log"
