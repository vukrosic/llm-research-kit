#!/usr/bin/env python3
"""
Summary plot of all experiments from the research session (2026-04-04).
Shows the progression of val_loss improvements across hypotheses.
"""
import json, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Collected results at 8M tokens
results_8M = {
    "squared_relu\n+constant\n(baseline)": 4.9214,
    "relu\n+constant": 4.9662,
    "relu\n+cosine": 4.9159,
    "squared_relu\n+cosine\n(new best)": 4.8956,
}

# Crossover chart: relu vs squared_relu across token counts
crossover = {
    "squared_relu+constant": [(2e6, 6.1084), (4e6, 5.4572), (8e6, 4.9214)],
    "relu+constant":         [(2e6, 6.0598), (4e6, 5.4553), (8e6, 4.9662)],
    "squared_relu+cosine":   [(2e6, 6.0930),                (8e6, 4.8956)],
    "relu+cosine":           [(2e6, 6.0603),                (8e6, 4.9159)],
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Research Session Summary — 88M Model, TITAN X Pascal (2026-04-04)", fontsize=12)

# --- Plot 1: 8M benchmark bar chart ---
ax = axes[0]
names = list(results_8M.keys())
vals = list(results_8M.values())
colors = ["#e74c3c", "#3498db", "#9b59b6", "#2ecc71"]
bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="black", linewidth=0.8)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("Validation Loss @ 8M tokens")
ax.set_title("Config Comparison at 8M Tokens")
ax.grid(True, alpha=0.3, axis="y")
ymin = min(vals) - 0.01
ymax = max(vals) + 0.01
ax.set_ylim(ymin, ymax)
for bar, v in zip(bars, vals):
    delta = "" if v == 4.9214 else f" ({4.9214-v:+.4f})"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f"{v:.4f}{delta}", ha="center", va="bottom", fontsize=8, fontweight="bold")

# --- Plot 2: Crossover curves ---
ax = axes[1]
styles = {
    "squared_relu+constant": dict(color="#e74c3c", linestyle="-",  marker="o", linewidth=2),
    "relu+constant":         dict(color="#3498db", linestyle="-",  marker="s", linewidth=2),
    "squared_relu+cosine":   dict(color="#27ae60", linestyle="--", marker="^", linewidth=2),
    "relu+cosine":           dict(color="#8e44ad", linestyle="--", marker="D", linewidth=2),
}
for name, pts in crossover.items():
    xs = [p[0]/1e6 for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, label=name, **styles[name], markersize=7)
ax.set_xlabel("Training Tokens (M)")
ax.set_ylabel("Final Validation Loss")
ax.set_title("Activation × LR Schedule × Token Count")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
ax.axvline(x=4, color="gray", linestyle=":", alpha=0.7, label="crossover ~4M")
ax.text(4.1, ax.get_ylim()[0] + 0.02, "crossover\n~4M tok", fontsize=8, color="gray")

plt.tight_layout()
out = "experiments/research_summary_2026-04-04.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Summary plot saved to {out}")
