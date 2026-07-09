import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def plot_attention_comparison(dense_path: Path, sparse_path: Path, out_path: Path) -> None:
    dense = load_metrics(dense_path)
    sparse = load_metrics(sparse_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex="col")
    runs = [
        ("5M dense baseline", dense),
        ("5M MiniMax sparse", sparse),
    ]

    for label, data in runs:
        history = data["history"]
        steps = history["steps"]
        elapsed_minutes = [t / 60.0 for t in history["elapsed_times"]]
        axes[0, 0].plot(steps, history["val_losses"], marker="o", linewidth=2, label=label)
        axes[0, 1].plot(elapsed_minutes, history["val_losses"], marker="o", linewidth=2, label=label)
        axes[1, 0].plot(steps, history["val_accuracies"], marker="o", linewidth=2, label=label)
        axes[1, 1].plot(elapsed_minutes, history["val_accuracies"], marker="o", linewidth=2, label=label)

    axes[0, 0].set_title("Validation loss vs step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_title("Validation loss vs time")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_title("Validation accuracy vs step")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 1].set_title("Validation accuracy vs time")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 0].set_xlabel("Training step")
    axes[1, 1].set_xlabel("Elapsed time (minutes)")
    axes[0, 0].set_xlabel("Training step")
    axes[0, 1].set_xlabel("Elapsed time (minutes)")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Matched 5M run: dense attention vs MiniMax sparse attention")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot matched dense vs sparse attention runs.")
    parser.add_argument("--dense", required=True, type=Path, help="Dense baseline metrics JSON")
    parser.add_argument("--sparse", required=True, type=Path, help="Sparse attention metrics JSON")
    parser.add_argument("--out", required=True, type=Path, help="Output PNG path")
    args = parser.parse_args()
    plot_attention_comparison(args.dense, args.sparse, args.out)


if __name__ == "__main__":
    main()
