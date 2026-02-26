#!/bin/bash

# Experiment 1: With QK Norm (100M)
echo "Starting Experiment: With QK Norm (100M)"
python train_llm.py \
    --config_class configs.ablation_config.QKNormAblationConfig \
    --output_dir ./checkpoints/qk_norm \
    --train_tokens 100000000

# Experiment 2: Per Head Scaling (100M)
echo "Starting Experiment: Per Head Scaling (100M)"
python train_llm.py \
    --config_class configs.ablation_config.PerHeadScalingAblationConfig \
    --output_dir ./checkpoints/per_head_scaling \
    --train_tokens 100000000

# Experiment 3: K Only Norm (100M)
echo "Starting Experiment: K Only Norm (100M)"
python train_llm.py \
    --config_class configs.ablation_config.KOnlyNormAblationConfig \
    --output_dir ./checkpoints/k_only_norm \
    --train_tokens 100000000

# Experiment 4: Shared Norm (100M)
echo "Starting Experiment: Shared Norm (100M)"
python train_llm.py \
    --config_class configs.ablation_config.SharedNormAblationConfig \
    --output_dir ./checkpoints/shared_norm \
    --train_tokens 100000000

# Experiment 5: QK Bias (100M)
echo "Starting Experiment: QK Bias (100M)"
python train_llm.py \
    --config_class configs.ablation_config.QKBiasAblationConfig \
    --output_dir ./checkpoints/qk_bias \
    --train_tokens 100000000

# Final plotting
python plot_ablation_results.py
