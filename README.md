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

The repo includes named presets for scaling-law work. The names are rough size
labels, not exact parameter contracts; the priority is clean architecture
numbers and comparable shapes.

| Preset | Params | Architecture | Purpose |
|---|---:|---|---|
| `fast_research` | `14.0M` | `d_model=256`, `n_heads=4`, `n_kv_heads=2`, `n_layers=2`, `d_ff=1024`, `seq=512` | Quick smoke-test preset |
| `5m` | `6.7M` | `d_model=128`, `n_heads=2`, `n_kv_heads=1`, `n_layers=2`, `d_ff=512`, `seq=2048` | Tiny pipeline check |
| `25m` | `25.4M` | `d_model=384`, `n_heads=8`, `n_kv_heads=4`, `n_layers=4`, `d_ff=1536`, `seq=2048` | Small scaling point |
| `50m` | `48.2M` | `d_model=512`, `n_heads=8`, `n_kv_heads=4`, `n_layers=8`, `d_ff=2048`, `seq=2048` | Mid scaling point |
| `default` | `88.6M` | `d_model=512`, `n_heads=8`, `n_kv_heads=4`, `n_layers=22`, `d_ff=2048`, `seq=2048` | Legacy tuned large preset |
| `100m` | `100.2M` | `d_model=512`, `n_heads=8`, `n_kv_heads=4`, `n_layers=26`, `d_ff=2048`, `seq=2048` | Large scaling point |
| `research` | `25.4M` | `d_model=384`, `n_heads=8`, `n_kv_heads=4`, `n_layers=4`, `d_ff=1536`, `seq=1024` | Legacy research preset |

The main ladder is `5m -> 25m -> 50m -> default -> 100m`. The `default` and
`100m` presets use the same width, head layout, KV-head layout, and FFN width;
`100m` is just the deeper version. This keeps the largest models close to the
legacy tuned architecture while still giving a clean progression for scaling
curves.

Run any preset with:

```bash
python train_llm.py --config 100m
```
