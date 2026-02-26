import csv
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


@dataclass
class RunMetrics:
    name: str
    label: str
    steps: List[int]
    val_losses: List[float]
    val_accuracies: List[float]
    val_perplexities: List[float]
    final_val_loss: float
    final_val_accuracy: float
    final_val_perplexity: float
    tokens_seen: int
    train_tokens: int
    actual_steps: int
    total_time_minutes: float
    active_training_seconds: float
    group: str


EXPERIMENT_META: Dict[str, Dict[str, str]] = {
    "qk_norm": {"label": "QK Norm (Base)", "color": "#1f77b4"},
    "per_head_scaling": {"label": "Per-Head Scaling", "color": "#d62728"},
    "k_only_norm": {"label": "K-only Norm", "color": "#2ca02c"},
    "shared_norm": {"label": "Shared Norm", "color": "#8c564b"},
    "qk_bias": {"label": "QK Bias", "color": "#ff7f0e"},
    "no_qk_norm": {"label": "No QK Norm", "color": "#7f7f7f"},
    "rope_then_norm": {"label": "RoPE then Norm", "color": "#9467bd"},
}


def load_metrics() -> Dict[str, RunMetrics]:
    runs: Dict[str, RunMetrics] = {}
    for metrics_path in sorted(glob.glob("checkpoints/*/metrics.json")):
        name = os.path.basename(os.path.dirname(metrics_path))
        meta = EXPERIMENT_META.get(name, {"label": name, "color": "#333333"})
        with open(metrics_path, "r") as f:
            data = json.load(f)

        history = data.get("history", {})
        steps = history.get("steps", [])
        val_losses = history.get("val_losses", [])
        val_accuracies = history.get("val_accuracies", [])
        val_perplexities = history.get("val_perplexities", [])

        if not steps or not val_losses:
            continue

        n = min(len(steps), len(val_losses), len(val_accuracies), len(val_perplexities))
        steps = steps[:n]
        val_losses = val_losses[:n]
        val_accuracies = val_accuracies[:n]
        val_perplexities = val_perplexities[:n]

        train_tokens = int(data.get("train_tokens", 0))
        tokens_seen = int(data.get("tokens_seen", 0))
        group = "100M" if train_tokens >= 100_000_000 else "15M"

        final_metrics = data.get("final_metrics", {})
        runs[name] = RunMetrics(
            name=name,
            label=meta["label"],
            steps=steps,
            val_losses=val_losses,
            val_accuracies=val_accuracies,
            val_perplexities=val_perplexities,
            final_val_loss=float(final_metrics.get("val_loss", val_losses[-1])),
            final_val_accuracy=float(final_metrics.get("val_accuracy", val_accuracies[-1])),
            final_val_perplexity=float(final_metrics.get("val_perplexity", val_perplexities[-1])),
            tokens_seen=tokens_seen,
            train_tokens=train_tokens,
            actual_steps=int(data.get("actual_steps", steps[-1])),
            total_time_minutes=float(data.get("total_time_minutes", 0.0)),
            active_training_seconds=float(data.get("active_training_time_seconds", 0.0)),
            group=group,
        )
    return runs


def avg_auc_loss(run: RunMetrics) -> float:
    x = np.array(run.steps, dtype=float)
    y = np.array(run.val_losses, dtype=float)
    if len(x) < 2:
        return float(y.mean())
    area = np.trapezoid(y, x)
    span = x[-1] - x[0]
    return float(area / span) if span > 0 else float(y.mean())


def ensure_plots_dir() -> None:
    os.makedirs("plots", exist_ok=True)


def fig1_all_runs_val_loss(runs: Dict[str, RunMetrics]) -> str:
    plt.figure(figsize=(14, 8))
    for name, run in sorted(runs.items(), key=lambda kv: (kv[1].train_tokens, kv[0])):
        color = EXPERIMENT_META.get(name, {}).get("color", "#333333")
        linestyle = "-" if run.group == "100M" else "--"
        plt.plot(
            run.steps,
            run.val_losses,
            marker="o",
            markersize=5,
            linewidth=2,
            alpha=0.9,
            linestyle=linestyle,
            color=color,
            label=f"{run.label} ({run.group})",
        )

    plt.title("Validation Loss Across All Ablations", fontsize=20, pad=16, fontweight="bold")
    plt.xlabel("Training Step", fontsize=13)
    plt.ylabel("Validation Loss", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10)
    plt.tight_layout()
    out = "plots/paper_fig1_all_runs_val_loss.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def fig2_long_zoom(runs: Dict[str, RunMetrics], zoom_start: int = 1000) -> str:
    long_runs = {k: v for k, v in runs.items() if v.group == "100M"}
    plt.figure(figsize=(14, 8))
    ymin = None
    ymax = None
    xmax = 0

    for name, run in sorted(long_runs.items()):
        color = EXPERIMENT_META.get(name, {}).get("color", "#333333")
        pts = [(s, l) for s, l in zip(run.steps, run.val_losses) if s >= zoom_start]
        if not pts:
            continue
        xs, ys = zip(*pts)
        xmax = max(xmax, max(xs))
        ymin = min(ys) if ymin is None else min(ymin, min(ys))
        ymax = max(ys) if ymax is None else max(ymax, max(ys))
        plt.plot(xs, ys, marker="o", markersize=6, linewidth=2.2, color=color, label=run.label, alpha=0.95)

    if ymin is not None and ymax is not None:
        pad = max((ymax - ymin) * 0.2, 0.01)
        plt.ylim(ymin - pad, ymax + pad)
        plt.xlim(zoom_start - 100, xmax + 120)

    plt.title("100M Runs: Late-Stage Validation Loss (Zoomed)", fontsize=20, pad=16, fontweight="bold")
    plt.xlabel("Training Step", fontsize=13)
    plt.ylabel("Validation Loss", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=11)
    plt.tight_layout()
    out = "plots/paper_fig2_100M_zoom.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def fig3_final_val_loss_bar_100m(runs: Dict[str, RunMetrics]) -> str:
    rows = []
    for name, run in runs.items():
        if run.group == "100M":
            rows.append((run.label, run.final_val_loss, EXPERIMENT_META.get(name, {}).get("color", "#333333")))
    rows.sort(key=lambda x: x[1])

    labels = [x[0] for x in rows]
    values = [x[1] for x in rows]
    colors = [x[2] for x in rows]

    plt.figure(figsize=(12, 7))
    bars = plt.barh(labels, values, color=colors, alpha=0.92)
    plt.xlabel("Final Validation Loss (lower is better)", fontsize=12)
    plt.title("100M Runs: Final Validation Loss Ranking", fontsize=18, pad=12, fontweight="bold")
    plt.grid(True, axis="x", linestyle="--", alpha=0.35)
    plt.xlim(min(values) - 0.015, max(values) + 0.015)
    for bar, v in zip(bars, values):
        plt.text(v + 0.0012, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center", fontsize=10)
    plt.tight_layout()
    out = "plots/paper_fig3_100M_final_val_loss_bar.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def fig4_delta_vs_base(runs: Dict[str, RunMetrics], base_name: str = "qk_norm") -> str:
    if base_name not in runs:
        raise ValueError(f"Base run '{base_name}' not found")
    base = runs[base_name]
    base_map = {s: l for s, l in zip(base.steps, base.val_losses)}

    plt.figure(figsize=(14, 8))
    plt.axhline(0.0, color="black", linewidth=1.2, linestyle="--", alpha=0.8)
    for name, run in sorted(runs.items()):
        if run.group != "100M":
            continue
        if name == base_name:
            continue
        color = EXPERIMENT_META.get(name, {}).get("color", "#333333")
        xs = []
        deltas = []
        for s, l in zip(run.steps, run.val_losses):
            if s in base_map:
                xs.append(s)
                deltas.append(l - base_map[s])
        if xs:
            plt.plot(xs, deltas, marker="o", markersize=6, linewidth=2.2, color=color, alpha=0.95, label=run.label)

    plt.title("100M Runs: Delta Validation Loss vs QK Norm Base", fontsize=20, pad=16, fontweight="bold")
    plt.xlabel("Training Step", fontsize=13)
    plt.ylabel(r"$\Delta$ Val Loss (Variant - Base)", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=11)
    plt.tight_layout()
    out = "plots/paper_fig4_100M_delta_vs_base.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def fig5_efficiency_scatter(runs: Dict[str, RunMetrics]) -> str:
    plt.figure(figsize=(12, 8))
    for name, run in sorted(runs.items()):
        if run.group != "100M":
            continue
        color = EXPERIMENT_META.get(name, {}).get("color", "#333333")
        x = run.active_training_seconds / 60.0
        y = run.final_val_loss
        plt.scatter([x], [y], s=120, color=color, alpha=0.95)
        plt.text(x + 0.05, y + 0.0005, run.label, fontsize=10)

    plt.title("100M Runs: Compute Efficiency vs Quality", fontsize=18, pad=12, fontweight="bold")
    plt.xlabel("Active Training Time (minutes)", fontsize=12)
    plt.ylabel("Final Validation Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    out = "plots/paper_fig5_100M_efficiency.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def fig6_auc_bar_100m(runs: Dict[str, RunMetrics]) -> str:
    rows: List[Tuple[str, float, str]] = []
    for name, run in runs.items():
        if run.group == "100M":
            rows.append((run.label, avg_auc_loss(run), EXPERIMENT_META.get(name, {}).get("color", "#333333")))
    rows.sort(key=lambda x: x[1])

    labels = [x[0] for x in rows]
    vals = [x[1] for x in rows]
    colors = [x[2] for x in rows]

    plt.figure(figsize=(12, 7))
    bars = plt.barh(labels, vals, color=colors, alpha=0.92)
    plt.xlabel("Mean Validation Loss Over Trajectory (AUC-normalized)", fontsize=12)
    plt.title("100M Runs: Trajectory Quality Ranking", fontsize=18, pad=12, fontweight="bold")
    plt.grid(True, axis="x", linestyle="--", alpha=0.35)
    for bar, v in zip(bars, vals):
        plt.text(v + 0.0012, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center", fontsize=10)
    plt.tight_layout()
    out = "plots/paper_fig6_100M_auc_bar.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def fig7_shortrun_comparison(runs: Dict[str, RunMetrics]) -> str:
    rows = []
    for name, run in runs.items():
        if run.group == "15M":
            rows.append((run.label, run.final_val_loss, EXPERIMENT_META.get(name, {}).get("color", "#333333")))
    rows.sort(key=lambda x: x[1])
    labels = [x[0] for x in rows]
    vals = [x[1] for x in rows]
    colors = [x[2] for x in rows]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, vals, color=colors, alpha=0.92)
    plt.ylabel("Final Validation Loss", fontsize=12)
    plt.title("15M Runs: Dropped Ablations Snapshot", fontsize=17, pad=10, fontweight="bold")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)
    plt.tight_layout()
    out = "plots/paper_fig7_15M_shortrun_bar.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def write_summary_csv(runs: Dict[str, RunMetrics]) -> str:
    out = "plots/paper_results_summary.csv"
    fields = [
        "experiment",
        "label",
        "group",
        "train_tokens",
        "tokens_seen",
        "actual_steps",
        "final_val_loss",
        "final_val_perplexity",
        "final_val_accuracy",
        "mean_val_loss_auc_norm",
        "active_training_seconds",
        "total_time_minutes",
    ]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, run in sorted(runs.items()):
            writer.writerow(
                {
                    "experiment": name,
                    "label": run.label,
                    "group": run.group,
                    "train_tokens": run.train_tokens,
                    "tokens_seen": run.tokens_seen,
                    "actual_steps": run.actual_steps,
                    "final_val_loss": f"{run.final_val_loss:.9f}",
                    "final_val_perplexity": f"{run.final_val_perplexity:.9f}",
                    "final_val_accuracy": f"{run.final_val_accuracy:.9f}",
                    "mean_val_loss_auc_norm": f"{avg_auc_loss(run):.9f}",
                    "active_training_seconds": f"{run.active_training_seconds:.6f}",
                    "total_time_minutes": f"{run.total_time_minutes:.6f}",
                }
            )
    return out


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk", palette="muted")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.edgecolor"] = "0.2"
    plt.rcParams["axes.linewidth"] = 1.1

    ensure_plots_dir()
    runs = load_metrics()
    if not runs:
        raise SystemExit("No metrics found under checkpoints/*/metrics.json")

    outputs = [
        fig1_all_runs_val_loss(runs),
        fig2_long_zoom(runs),
        fig3_final_val_loss_bar_100m(runs),
        fig4_delta_vs_base(runs),
        fig5_efficiency_scatter(runs),
        fig6_auc_bar_100m(runs),
        fig7_shortrun_comparison(runs),
        write_summary_csv(runs),
    ]
    print("Generated artifacts:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
