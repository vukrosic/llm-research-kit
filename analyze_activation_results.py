#!/usr/bin/env python3
"""
Collect and plot results from the activation-discovery sweep.
Run after experiments/activation-discovery/runs/*-2M/metrics.json exist.
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE = "experiments/activation-discovery/runs"
RUNS = {
    "squared_relu": f"{BASE}/baseline-squaredrelu-2M",
    "gelu":         f"{BASE}/test-gelu-2M",
    "silu":         f"{BASE}/test-silu-2M",
    "swiglu":       f"{BASE}/test-swiglu-2M",
    "relu":         f"{BASE}/test-relu-2M",
}
COLORS = {
    "squared_relu": "#2ecc71",
    "gelu":         "#3498db",
    "silu":         "#9b59b6",
    "swiglu":       "#e74c3c",
    "relu":         "#f39c12",
}


def load(path):
    mf = os.path.join(path, "metrics.json")
    if not os.path.exists(mf):
        return None
    with open(mf) as f:
        return json.load(f)


results = {}
for name, path in RUNS.items():
    d = load(path)
    if d:
        results[name] = d
        fm = d["final_metrics"]
        print(f"{name:15s}  val_loss={fm['val_loss']:.4f}  val_acc={fm['val_accuracy']:.4f}  time={d['total_time_minutes']:.1f}m")
    else:
        print(f"{name:15s}  (not done)")

if len(results) < 2:
    print("\nNot enough results to plot yet.")
    raise SystemExit(0)

sorted_names = sorted(results, key=lambda n: results[n]["final_metrics"]["val_loss"])
print("\nRanking (lower val_loss = better):")
for rank, name in enumerate(sorted_names, 1):
    print(f"  {rank}. {name:15s}  {results[name]['final_metrics']['val_loss']:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Activation Function Comparison — 88M model, 2M tokens, TITAN X Pascal", fontsize=12)

# 1. Val loss curves
ax = axes[0]
for name in RUNS:
    if name not in results:
        continue
    h = results[name]["history"]
    ax.plot(h["steps"], h["val_losses"], label=name, color=COLORS[name], linewidth=2, marker="o", markersize=4)
ax.set_xlabel("Mini-batch step")
ax.set_ylabel("Validation Loss")
ax.set_title("Val Loss Curves")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 2. Final val loss bar chart
ax = axes[1]
names_done = sorted_names
vals = [results[n]["final_metrics"]["val_loss"] for n in names_done]
bars = ax.bar(range(len(names_done)), vals, color=[COLORS[n] for n in names_done])
ax.set_xticks(range(len(names_done)))
ax.set_xticklabels(names_done, rotation=30, ha="right")
ax.set_ylabel("Final Validation Loss")
ax.set_title("Ranking (lower = better)")
ax.grid(True, alpha=0.3, axis="y")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9)
ymin = min(vals) - 0.05
ymax = max(vals) + 0.05
ax.set_ylim(ymin, ymax)

# 3. Mechanism breakdown
ax = axes[2]
ready = set(results.keys())
mechanisms = []
gains = []
if {"gelu", "relu"} <= ready:
    mechanisms.append("Smoothness\n(ReLU→GELU)")
    gains.append(results["relu"]["final_metrics"]["val_loss"] - results["gelu"]["final_metrics"]["val_loss"])
if {"squared_relu", "relu"} <= ready:
    mechanisms.append("Quadratic\n(ReLU→Sq.ReLU)")
    gains.append(results["relu"]["final_metrics"]["val_loss"] - results["squared_relu"]["final_metrics"]["val_loss"])
if {"swiglu", "silu"} <= ready:
    mechanisms.append("Gating\n(SiLU→SwiGLU)")
    gains.append(results["silu"]["final_metrics"]["val_loss"] - results["swiglu"]["final_metrics"]["val_loss"])

if mechanisms:
    colors = ["#27ae60" if g > 0 else "#e74c3c" for g in gains]
    bars = ax.bar(mechanisms, gains, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Val Loss Diff (positive = right wins)")
    ax.set_title("Mechanism Gains")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, g in zip(bars, gains):
        ypos = bar.get_height() + 0.002 if g >= 0 else bar.get_height() - 0.005
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f"{g:+.4f}",
                ha="center", va="bottom" if g >= 0 else "top", fontweight="bold", fontsize=10)

plt.tight_layout()
out = "experiments/activation-discovery/results_2M.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {out}")
