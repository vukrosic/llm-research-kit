import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _x_axes(metrics: dict) -> dict[str, tuple[list[float], str]]:
    history = metrics["history"]
    steps = history["steps"]
    elapsed_minutes = [t / 60.0 for t in history["elapsed_times"]]
    tokens_per_step = metrics.get("tokens_seen", metrics.get("train_tokens", 1)) / max(
        metrics.get("actual_steps", max(steps[-1], 1)),
        1,
    )
    tokens = [step * tokens_per_step for step in steps]
    return {
        "step": (steps, "Training step"),
        "tokens": (tokens, "Tokens seen"),
        "time": (elapsed_minutes, "Elapsed time (minutes)"),
    }


def plot_run_comparison(a_path: Path, b_path: Path, out_path: Path, a_label: str, b_label: str, title: str) -> None:
    a = load_metrics(a_path)
    b = load_metrics(b_path)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex="col")
    runs = [
        (a_label, a),
        (b_label, b),
    ]
    views = ["step", "tokens", "time"]

    for label, data in runs:
        history = data["history"]
        axes_map = _x_axes(data)
        for col, view in enumerate(views):
            x_values, xlabel = axes_map[view]
            axes[0, col].plot(x_values, history["val_losses"], marker="o", linewidth=2, label=label)
            axes[1, col].plot(x_values, history["val_accuracies"], marker="o", linewidth=2, label=label)
            axes[0, col].set_xlabel(xlabel)
            axes[1, col].set_xlabel(xlabel)

    axes[0, 0].set_title("Validation loss vs step")
    axes[0, 1].set_title("Validation loss vs tokens")
    axes[0, 2].set_title("Validation loss vs time")
    axes[1, 0].set_title("Validation accuracy vs step")
    axes[1, 1].set_title("Validation accuracy vs tokens")
    axes[1, 2].set_title("Validation accuracy vs time")

    axes[0, 0].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Accuracy")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 1].set_ylabel("Accuracy")
    axes[0, 2].set_ylabel("Loss")
    axes[1, 2].set_ylabel("Accuracy")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a dense vs sparse run comparison.")
    parser.add_argument("--run_a", required=True, type=Path, help="First metrics JSON")
    parser.add_argument("--run_b", required=True, type=Path, help="Second metrics JSON")
    parser.add_argument("--out", required=True, type=Path, help="Output PNG path")
    parser.add_argument("--label_a", default="run A", help="Label for first run")
    parser.add_argument("--label_b", default="run B", help="Label for second run")
    parser.add_argument("--title", default="Run comparison", help="Figure title")
    args = parser.parse_args()
    plot_run_comparison(args.run_a, args.run_b, args.out, args.label_a, args.label_b, args.title)


if __name__ == "__main__":
    main()
