import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.minimax_sparse_config import LLMConfig, MiniMaxSparseConfig
from data.real_dataset import prepare_real_corpus
from training.tiny_trainer import train_tiny_lm


def build_config(attention_impl: str, vocab_size: int, seq_len: int) -> LLMConfig:
    return LLMConfig(
        d_model=96,
        n_heads=6,
        n_kv_heads=2,
        n_layers=6,
        d_ff=384,
        max_seq_len=seq_len,
        vocab_size=vocab_size,
        dropout=0.0,
        attention_impl=attention_impl,
        minimax_sparse=MiniMaxSparseConfig(block_size=16, top_k=2, index_dim=16, dropout=0.0),
    )


def read_metrics(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def plot_compare(run_dir: Path, names: list[str], run_tag: str = "") -> Path:
    out = run_dir / ("comparison.png" if not run_tag else f"comparison_{run_tag}.png")
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex="col")
    series = {}
    for name in names:
        rows = read_metrics(run_dir / name / "metrics.jsonl")
        xs = [r["tokens_seen"] for r in rows]
        val_loss = [r["val_loss"] for r in rows]
        train_loss = [r["train_loss"] for r in rows]
        val_ppl = [r["val_ppl"] for r in rows]
        val_acc = [r["val_acc"] for r in rows]
        tps = [r["tokens_per_second"] for r in rows]
        series[name] = (xs, val_loss)
        axes[0, 0].plot(xs, val_loss, marker="o", label=name)
        axes[0, 1].plot(xs, val_ppl, marker="o", label=name)
        axes[1, 0].plot(xs, train_loss, marker="o", label=name)
        axes[1, 1].plot(xs, val_acc, marker="o", label=name)
        axes[2, 0].plot(xs, tps, marker="o", label=name)
    if "dense" in series and "minimax_sparse" in series:
        xs = series["dense"][0]
        deltas = [
            sparse_loss - dense_loss
            for dense_loss, sparse_loss in zip(series["dense"][1], series["minimax_sparse"][1])
        ]
        axes[2, 1].axhline(0.0, color="black", linewidth=1, alpha=0.5)
        axes[2, 1].plot(xs, deltas, marker="o", color="tab:red", label="sparse - dense")
        axes[2, 1].legend()
    labels = [
        (axes[0, 0], "A. Validation loss", "Loss"),
        (axes[0, 1], "B. Validation perplexity", "PPL"),
        (axes[1, 0], "C. Training loss", "Loss"),
        (axes[1, 1], "D. Validation accuracy", "Accuracy"),
        (axes[2, 0], "E. Tokens per second", "Tok/s"),
        (axes[2, 1], "F. Sparse - dense validation loss", "Delta"),
    ]
    for ax, title, ylabel in labels:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[2, 0].set_xlabel("Tokens seen")
    axes[2, 1].set_xlabel("Tokens seen")
    plot_title = "5M parameter real-data smoke: dense vs MiniMax sparse"
    if run_tag:
        plot_title = f"{plot_title} [{run_tag}]"
    fig.suptitle(plot_title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tiny dense and MiniMax sparse LMs on real 40M data.")
    parser.add_argument("--dataset_path", default="processed_data/speedrun_40M")
    parser.add_argument("--token_budget", type=int, default=500_000)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_tag", default="")
    parser.add_argument("--run_root", type=Path, default=Path("runs/tiny_real_compare"))
    args = parser.parse_args()

    corpus = prepare_real_corpus(
        dataset_path=args.dataset_path,
        seq_len=args.seq_len,
        token_budget=args.token_budget,
    )
    run_dir = args.run_root / time.strftime("%Y%m%d_%H%M%S")
    if args.run_tag:
        run_dir = run_dir.with_name(f"{run_dir.name}_{args.run_tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "data_manifest.json").write_text(
        json.dumps(
            {
                "dataset_path": args.dataset_path,
                "tokenizer_name": corpus.tokenizer_name,
                "vocab_size": corpus.vocab_size,
                "tokens_available_loaded": corpus.tokens_available,
                "train_sequences": len(corpus.train),
                "val_sequences": len(corpus.val),
                "seq_len": args.seq_len,
                "token_budget": args.token_budget,
                "run_tag": args.run_tag,
            },
            indent=2,
        )
        + "\n"
    )

    summaries = {}
    for attention_impl in ["dense", "minimax_sparse"]:
        config = build_config(attention_impl, corpus.vocab_size, args.seq_len)
        summaries[attention_impl] = train_tiny_lm(
            config=config,
            train_dataset=corpus.train,
            val_dataset=corpus.val,
            output_dir=run_dir / attention_impl,
            token_budget=args.token_budget,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device_name=args.device,
            eval_every=args.eval_every,
            seed=args.seed,
        )

    plot_path = plot_compare(run_dir, ["dense", "minimax_sparse"], run_tag=args.run_tag)
    plot_tags = {
        "A": "Validation loss",
        "B": "Validation perplexity",
        "C": "Training loss",
        "D": "Validation accuracy",
        "E": "Tokens per second",
        "F": "Sparse minus dense validation loss",
    }
    (run_dir / "plot_tags.json").write_text(json.dumps(plot_tags, indent=2) + "\n")
    summary = {
        "run_dir": str(run_dir),
        "plot": str(plot_path),
        "run_tag": args.run_tag,
        "runs": summaries,
        "plot_tags": plot_tags,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
