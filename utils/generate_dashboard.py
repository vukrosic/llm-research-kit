import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('dark_background')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 20,
    'grid.alpha': 0.2
})

# Load the data
JSON_PATH = '/root/llm-research-kit/ablation_results/50000000tok/ablation_comparison.json'
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

# Create figure
fig = plt.figure(figsize=(20, 15))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 0.8])

# Color palette
colors = {
    'baseline': '#FFD700',       # Gold
    'no_embed_scale': '#FF7043', # Deep Orange
    'no_qk_norm': '#26A69A',     # Teal
    'polar_express_2': '#9575CD'  # Purple/Violet
}

labels = {
    'baseline': 'Baseline (Muon 5-steps, √d-Scaling, QK-Norm)',
    'no_embed_scale': 'Ablation: No √d Embedding Scale',
    'no_qk_norm': 'Ablation: No QK-Normalization',
    'polar_express_2': 'Ablation: Polar Express (Muon) 2-steps'
}

# 1. Validation Loss Full Convergence
ax1 = fig.add_subplot(gs[0, 0])
for res in data['all_results']:
    name = res['experiment_name']
    history = res['history']
    # Convert steps to Millions of Tokens
    # Tokens per step = batch_size (4) * seq_len (2048) = 8192
    tokens_m = np.array(history['steps']) * 8192 / 1e6
    ax1.plot(tokens_m, history['val_losses'], label=labels.get(name, name), 
             color=colors.get(name), linewidth=2.5, marker='o' if len(tokens_m) < 10 else None)

ax1.set_title('Validation Loss (Convergence Overview)')
ax1.set_xlabel('Tokens Seen (Millions)')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# 2. Validation Loss Tail (Zoomed)
ax2 = fig.add_subplot(gs[0, 1])
for res in data['all_results']:
    name = res['experiment_name']
    history = res['history']
    tokens_m = np.array(history['steps']) * 8192 / 1e6
    ax2.plot(tokens_m, history['val_losses'], color=colors.get(name), linewidth=3, marker='o')

ax2.set_title('Validation Loss (Zoomed Convergence Tail)')
ax2.set_xlabel('Tokens Seen (Millions)')
ax2.set_ylabel('Loss')
ax2.set_ylim(4.0, 4.6)
ax2.set_xlim(2, 52)
ax2.grid(True)

# 3. Validation Accuracy
ax3 = fig.add_subplot(gs[1, 0])
for res in data['all_results']:
    name = res['experiment_name']
    history = res['history']
    tokens_m = np.array(history['steps']) * 8192 / 1e6
    ax3.plot(tokens_m, history['val_accuracies'], color=colors.get(name), linewidth=2.5, marker='o')

ax3.set_title('Validation Accuracy')
ax3.set_xlabel('Tokens Seen (Millions)')
ax3.set_ylabel('Accuracy')
ax3.grid(True)

# 4. Final Loss Degradation (Bar Chart)
ax4 = fig.add_subplot(gs[1, 1])
summary = data['summary']
names = [entry['experiment'] for entry in summary]
deltas = [entry['loss_delta_pct'] for entry in summary]
bar_colors = [colors.get(n) for n in names]

bars = ax4.bar(names, deltas, color=bar_colors, alpha=0.8)
ax4.set_title('Final Loss Degradation vs Baseline (%)')
ax4.set_ylabel('Δ Loss (%)')
ax4.set_ylim(0, max(deltas) * 1.3)

# Add text labels on bars
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'+{height:.2f}%', ha='center', va='bottom', fontsize=12, color='white', fontweight='bold')

# 5. Training Throughput
ax5 = fig.add_subplot(gs[2, 0])
throughputs = []
names_tp = []
for res in data['all_results']:
    name = res['experiment_name']
    # Throughput = Total Tokens / Training Time
    tp = (res['tokens_seen'] / res['active_training_time_seconds']) / 1000 # kTokens/sec
    throughputs.append(tp)
    names_tp.append(name)

ax5.barh(names_tp, throughputs, color=[colors.get(n) for n in names_tp], alpha=0.7)
ax5.set_title('Training Throughput (kTokens/sec)')
ax5.set_xlabel('kTokens per second')
for i, v in enumerate(throughputs):
    ax5.text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold')

# 6. Global Summary Table
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')
table_data = []
for entry in summary:
    table_data.append([
        entry['experiment'],
        f"{entry['val_loss']:.4f}",
        f"{entry['val_accuracy']*100:.1f}%",
        f"{entry['val_perplexity']:.1f}",
        f"{entry['loss_delta_pct']:+.2f}%"
    ])

table = ax6.table(cellText=table_data, 
                  colLabels=['Experiment', 'Loss', 'Acc', 'PPL', 'Δ Loss'],
                  loc='center', cellLoc='center')
table.scale(1, 2.5)
table.set_fontsize(13)
table.auto_set_font_size(False)

# Color the first row labels
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_facecolor('#333333')
        cell.set_text_props(color='white', fontweight='bold')
    elif i > 0:
        cell.set_facecolor('#1a1a1a')
        cell.set_text_props(color='white')

plt.suptitle('LLM Research Kit - Ablation Study Results (50M Tokens)', y=0.98, fontsize=24, fontweight='bold')
plt.figtext(0.5, 0.01, f"* Experiments run in 88M parameter model. 1M token smoke tests confirmed behavior before full 50M run. Baseline uses Muon (5-step NS), √d-Scaling, and QK-Norm.", 
            ha='center', fontsize=12, color='#aaaaaa', style='italic')

os.makedirs('/root/llm-research-kit/plots', exist_ok=True)
SAVE_PATH = '/root/llm-research-kit/plots/ablation_50M_dashboard.png'
plt.savefig(SAVE_PATH, dpi=120, bbox_inches='tight')
print(f"✅ Dashboard generated: {SAVE_PATH}")
