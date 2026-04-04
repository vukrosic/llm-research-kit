#!/usr/bin/env python3
"""
Quick activation function debug experiment.
Uses a slightly larger model with proper train/val split to detect real differences.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import time

torch.set_num_threads(4)

class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, activation='relu'):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.ff = self._make_ffn(d_model, d_ff, activation)

    def _make_ffn(self, d_model, d_ff, activation):
        if activation == 'swiglu':
            class SwiGLUFFN(nn.Module):
                def __init__(self, d_model, d_ff):
                    super().__init__()
                    self.w1 = nn.Linear(d_model, d_ff, bias=False)
                    self.w3 = nn.Linear(d_model, d_ff, bias=False)
                    self.down = nn.Linear(d_ff, d_model, bias=False)
                def forward(self, x):
                    return self.down(F.silu(self.w1(x)) * self.w3(x))
            return SwiGLUFFN(d_model, d_ff)
        else:
            class FFN(nn.Module):
                def __init__(self, d_model, d_ff, act):
                    super().__init__()
                    self.up = nn.Linear(d_model, d_ff, bias=False)
                    self.down = nn.Linear(d_ff, d_model, bias=False)
                    self.act = act
                def forward(self, x):
                    h = self.up(x)
                    if self.act == 'squared_relu':
                        h = torch.square(F.relu(h))
                    elif self.act == 'relu':
                        h = F.relu(h)
                    elif self.act == 'gelu':
                        h = F.gelu(h)
                    elif self.act == 'silu':
                        h = F.silu(h)
                    elif self.act == 'mish':
                        h = F.mish(h)
                    elif self.act == 'softplus':
                        h = F.softplus(h)
                    else:
                        h = F.relu(h)
                    return self.down(h)
            return FFN(d_model, d_ff, activation)

    def forward(self, x):
        seq_len = x.shape[1]
        attn_out, _ = self.attn(x, x, x, attn_mask=torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool())
        x = x + attn_out
        x = self.norm1(x)
        x = x + self.ff(self.norm2(x))
        return x

class TinyLLM(nn.Module):
    def __init__(self, vocab_size=8000, d_model=256, n_layers=4, d_ff=1024, activation='relu'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TinyTransformerBlock(d_model, d_ff, activation) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)

def make_dataset(n_samples, seq_len, vocab_size, seed):
    """Make random integer sequences"""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (n_samples, seq_len), generator=g)

def run_experiment(activation, steps=80, seed=42):
    torch.manual_seed(seed)

    vocab_size = 8000
    d_model = 256
    n_layers = 4
    d_ff = 1024
    seq_len = 128
    batch_size = 32

    # Create model
    model = TinyLLM(vocab_size, d_model, n_layers, d_ff, activation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    # Train on seeds 100, val on seed 200 - completely different distributions
    train_data = make_dataset(10000, seq_len, vocab_size, seed=100)
    val_data = make_dataset(2000, seq_len, vocab_size, seed=200)

    train_losses = []
    val_losses = []
    grad_norms = []

    n_train = len(train_data)

    for step in range(steps):
        model.train()
        idx = torch.randint(0, n_train, (batch_size,))
        x = train_data[idx]

        optimizer.zero_grad()
        logits = model(x[:, :-1])          # [B, T-1, V]  predict next token
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), x[:, 1:].reshape(-1))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()

        train_losses.append(loss.item())
        grad_norms.append(grad_norm)

        # Validate
        if step % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_idx = torch.randint(0, len(val_data), (batch_size,))
                vx = val_data[val_idx]
                val_logits = model(vx[:, :-1])
                val_loss = F.cross_entropy(val_logits.reshape(-1, vocab_size), vx[:, 1:].reshape(-1)).item()
                val_losses.append((step, val_loss))

    model.eval()
    with torch.no_grad():
        fv = val_data[:512]
        final_val_logits = model(fv[:, :-1])
        final_val = F.cross_entropy(final_val_logits.reshape(-1, vocab_size), fv[:, 1:].reshape(-1)).item()

    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'grad_norm': grad_norms,
        'final_val_loss': final_val,
        'final_train_loss': train_losses[-1]
    }

ACTIVATIONS = ['squared_relu', 'gelu', 'silu', 'swiglu', 'relu']
COLORS = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

print("=" * 60)
print("ACTIVATION FUNCTION DEBUG EXPERIMENT")
print("=" * 60)
print(f"Model: 4 layers, 256 dim, 1024 FFN, 8K vocab, 128 seq")
print(f"Steps per activation: 80")
print(f"Train: 10000 seqs (seed 100), Val: 2000 seqs (seed 200)")
print("=" * 60)

results = {}
for i, act in enumerate(ACTIVATIONS):
    print(f"\n[{i+1}/{len(ACTIVATIONS)}] {act}...", end=" ", flush=True)
    start = time.time()
    r = run_experiment(act, steps=80, seed=42)
    elapsed = time.time() - start
    results[act] = r
    print(f"val={r['final_val_loss']:.4f} train={r['final_train_loss']:.4f} ({elapsed:.1f}s)")

sorted_acts = sorted(results.keys(), key=lambda a: results[a]['final_val_loss'])

print("\n" + "=" * 60)
print("RESULTS (sorted by validation loss)")
print("=" * 60)
for rank, act in enumerate(sorted_acts, 1):
    r = results[act]
    print(f"  {rank}. {act:15s} | val={r['final_val_loss']:.4f} | train={r['final_train_loss']:.4f}")

# Create graphs
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Validation loss curves
ax = axes[0, 0]
for i, act in enumerate(ACTIVATIONS):
    r = results[act]
    steps = [s for s, v in r['val_loss']]
    vals = [v for s, v in r['val_loss']]
    ax.plot(steps, vals, label=act, color=COLORS[i], linewidth=2, marker='o', markersize=3)
ax.set_xlabel('Step')
ax.set_ylabel('Validation Loss')
ax.set_title('Validation Loss Curves')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Final val loss bar chart
ax = axes[0, 1]
bars = ax.bar(range(len(sorted_acts)), [results[a]['final_val_loss'] for a in sorted_acts],
              color=[COLORS[ACTIVATIONS.index(a)] for a in sorted_acts])
ax.set_xticks(range(len(sorted_acts)))
ax.set_xticklabels(sorted_acts, rotation=45, ha='right')
ax.set_ylabel('Validation Loss')
ax.set_title('Validation Loss Ranking (lower is better)')
ax.grid(True, alpha=0.3, axis='y')
for bar, act in zip(bars, sorted_acts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{results[act]["final_val_loss"]:.2f}', ha='center', va='bottom', fontsize=9)

# Plot 3: Train loss curves
ax = axes[1, 0]
for i, act in enumerate(ACTIVATIONS):
    r = results[act]
    ax.plot(range(len(r['train_loss'])), r['train_loss'], label=act, color=COLORS[i], linewidth=2)
ax.set_xlabel('Step')
ax.set_ylabel('Training Loss')
ax.set_title('Training Loss Curves')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Mechanism tests
ax = axes[1, 1]
gelu_v = results['gelu']['final_val_loss']
relu_v = results['relu']['final_val_loss']
smooth_gain = relu_v - gelu_v

sqr_v = results['squared_relu']['final_val_loss']
quad_gain = relu_v - sqr_v

swiglu_v = results['swiglu']['final_val_loss']
silu_v = results['silu']['final_val_loss']
gate_gain = silu_v - swiglu_v

mechanisms = ['Smoothness\n(GELU - ReLU)', 'Quadratic\n(SquaredReLU - ReLU)', 'Gating\n(SwiGLU - SiLU)']
gains = [smooth_gain, quad_gain, gate_gain]
bar_colors = ['#27ae60' if g > 0 else '#e74c3c' for g in gains]
bars = ax.bar(mechanisms, gains, color=bar_colors, edgecolor='black', linewidth=1)
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('Val Loss Diff (positive = left wins)')
ax.set_title('Mechanism Tests')
for bar, g in zip(bars, gains):
    ypos = bar.get_height() + max(gains)*0.02 if g >= 0 else bar.get_height() - max(gains)*0.08
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f'{g:.3f}', ha='center', va='bottom' if g >= 0 else 'top', fontweight='bold')

plt.tight_layout()
plt.savefig('experiments/activation-discovery/debug_results.png', dpi=150, bbox_inches='tight')
plt.savefig('experiments/activation-discovery/debug_results.pdf', bbox_inches='tight')
print("\nGraphs saved!")

# Derived rules
print("\n" + "=" * 60)
print("DERIVED RULES")
print("=" * 60)
print(f"\n[H-Smoothness] GELU vs ReLU: {smooth_gain:+.4f}")
if abs(smooth_gain) < 0.05:
    print(f"  -> No significant difference at small scale")
elif smooth_gain > 0:
    print(f"  -> GELU better (smoothness helps)")
else:
    print(f"  -> ReLU surprisingly competitive")

print(f"\n[H-Quadratic] SquaredReLU vs ReLU: {quad_gain:+.4f}")
if abs(quad_gain) < 0.05:
    print(f"  -> No significant difference")
elif quad_gain > 0:
    print(f"  -> SquaredReLU better (quadratic helps)")
else:
    print(f"  -> ReLU competitive with SquaredReLU")

print(f"\n[H-Gating] SwiGLU vs SiLU: {gate_gain:+.4f}")
if abs(gate_gain) < 0.05:
    print(f"  -> No significant difference")
elif gate_gain > 0:
    print(f"  -> SwiGLU better (gating helps)")
else:
    print(f"  -> SiLU competitive with SwiGLU")

print("\n" + "=" * 60)
print("NEXT QUESTIONS FOR FULL EXPERIMENT")
print("=" * 60)
print("""
1. Do these rankings HOLD at 88M params, 22 layers?
2. Do curves diverge or converge at 200+ steps?
3. Is there an activation x learning rate interaction?
4. Which activation to use as default baseline?
""")
