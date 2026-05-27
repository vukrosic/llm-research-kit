# LLM Research Kit

A high-performance codebase for LLM research, pretraining, and optimization: testing new architectures, optimizers, or training.

- Modular transformer with GQA, RoPE, and RMSNorm
- Muon optimizer alongside AdamW
- Training script, flexible configuration

- `models/`: Transformer layers and components (RoPE, RMSNorm, Multi-Head Attention).
- `optimizers/`: Muon optimizer (outperforms AdamW and all others).
- `training/`: Core trainer logic and utilities.
- `configs/`: Hyperparameter and dataset configurations.
- `utils/`: Logging, plotting, and helper functions.

## 🚀 Getting Started

#### Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download the Dataset

### Option A: 1B tokens
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 1B Pretraining Data...')
ds = load_dataset('vukrosic/blueberry-1B-pretrain')
os.makedirs('processed_data/pretrain_1B', exist_ok=True)
ds.save_to_disk('processed_data/pretrain_1B')
print('✅ Full Data Ready!')
"
```

### Option B: 2B tokens
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 2B Pretraining Data...')
ds = load_dataset('vukrosic/blueberry-2B-pretrain')
os.makedirs('processed_data/pretrain_2B', exist_ok=True)
ds.save_to_disk('processed_data/pretrain_2B')
print('✅ Full Data Ready!')
"
```

### Option C: Quick Start (40M Tokens)
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 40M Token Subset...')
ds = load_dataset('vukrosic/blueberry-1B-pretrain', split='train[:20000]')
os.makedirs('processed_data/speedrun_40M', exist_ok=True)
ds.save_to_disk('processed_data/speedrun_40M')
print('✅ Speedrun Data Ready!')
"
```

## 🧠 LLM Architecture

Default is an **88M parameter** transformer LLM, you can modify configs.

- **Layers**: 22 Transformer blocks.
- **Hidden Dimension (`d_model`)**: 512.
- **Feed-Forward Dimension (`d_ff`)**: 2048.
- **Attention System**:
  - 8 Query heads, 4 Key-Value heads (**Grouped Query Attention**).
  - Rotary Positional Embeddings (**RoPE**).
  - Fused QKVO projection for optimized compute.
  - QK-Normalization for training stability.
- **Normalization**: Pre-norm **RMSNorm**.
- **Activation**: **Squared ReLU** (Primer-style).
- **Vocab Size**: 49,152.
- **Sequence Length**: 2048 tokens.

### Optimization Highlights
- **Weight Tying**: Shared weights between token embeddings and the LM head.
- **Muon Support**: Architecture optimized for the Muon optimizer's orthogonal updates.
- **Efficiency**: Designed for `torch.compile` compatibility and mixed-precision (BF16) training.

## Scaling-Law Presets

The repo now includes a clean size ladder for scaling-law work:

| Preset | Approx params | Architecture |
|---|---:|---|
| `5m` | `5.1M` | `d_model=96`, `n_heads=3`, `n_layers=4`, `d_ff=384` |
| `25m` | `25.4M` | `d_model=384`, `n_heads=12`, `n_layers=4`, `d_ff=1536` |
| `50m` | `51.1M` | `d_model=512`, `n_heads=16`, `n_layers=9`, `d_ff=2048` |
| `100m` | `100.2M` | `d_model=512`, `n_heads=16`, `n_layers=26`, `d_ff=2048` |

Run any preset with:

```bash
python train_llm.py --config 100m
```

To print the full ladder with exact parameter counts:

```bash
python scripts/print_model_scales.py
```

