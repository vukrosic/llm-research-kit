from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.minimax_sparse_config import LLMConfig, MiniMaxSparseConfig
from data.real_dataset import prepare_real_corpus
from models.llm import MinimalLLM
from training.tiny_trainer import evaluate, train_tiny_lm


@dataclass(frozen=True)
class ArchSpec:
    name: str
    block_size: int
    top_k: int


def build_config(spec: ArchSpec, vocab_size: int, max_seq_len: int) -> LLMConfig:
    sparse = MiniMaxSparseConfig(
        block_size=spec.block_size,
        top_k=spec.top_k,
        index_dim=24,
        pooling="max",
        router_source="separate",
        dropout=0.0,
    )
    return LLMConfig(
        d_model=144,
        n_heads=6,
        n_kv_heads=2,
        n_layers=6,
        d_ff=576,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        dropout=0.0,
        attention_impl="minimax_sparse",
        minimax_sparse=sparse,
    )


def context_eval(model: MinimalLLM, corpus_cache: dict[int, object], eval_seq_lens: list[int], device: torch.device, max_batches: int) -> dict[str, dict]:
    results = {}
    for seq_len in eval_seq_lens:
        loader = DataLoader(corpus_cache[seq_len].val, batch_size=4, shuffle=False, num_workers=0)
        results[str(seq_len)] = evaluate(model, loader, device, max_batches=max_batches)
    return results


def plot_summary(study_dir: Path, rows: list[dict], eval_seq_lens: list[int]) -> Path:
    plot_path = study_dir / "arch_summary.png"
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for row in rows:
        xs = [int(k) for k in row["context_eval"].keys()]
        ys = [row["context_eval"][str(x)]["val_loss"] for x in xs]
        ax.plot(xs, ys, marker="o", label=row["name"])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Validation loss")
    ax.set_title("Mid-scale sparse attention architecture ablation")
    ax.set_xticks(eval_seq_lens)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mid-scale architecture ablations for MiniMax sparse attention.")
    parser.add_argument("--dataset_path", default="processed_data/speedrun_40M")
    parser.add_argument("--train_seq_len", type=int, default=256)
    parser.add_argument("--eval_seq_lens", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--dataset_token_budget", type=int, default=300_000)
    parser.add_argument("--train_token_budget", type=int, default=80_000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_batches", type=int, default=4)
    parser.add_argument("--run_root", type=Path, default=Path("runs/mid_arch_ablation"))
    args = parser.parse_args()

    if args.train_seq_len not in args.eval_seq_lens:
        args.eval_seq_lens = sorted(set(args.eval_seq_lens + [args.train_seq_len]))

    study_dir = args.run_root / time.strftime("%Y%m%d_%H%M%S")
    study_dir.mkdir(parents=True, exist_ok=True)

    corpus_cache = {
        seq_len: prepare_real_corpus(
            dataset_path=args.dataset_path,
            seq_len=seq_len,
            token_budget=args.dataset_token_budget,
        )
        for seq_len in args.eval_seq_lens
    }
    vocab_size = next(iter(corpus_cache.values())).vocab_size
    max_seq_len = max(args.eval_seq_lens)

    specs = [
        ArchSpec("mid_sparse_k1", block_size=16, top_k=1),
        ArchSpec("mid_sparse_k2", block_size=16, top_k=2),
        ArchSpec("mid_sparse_k4", block_size=16, top_k=4),
        ArchSpec("mid_sparse_bs8", block_size=8, top_k=2),
        ArchSpec("mid_sparse_bs32", block_size=32, top_k=2),
    ]

    records: list[dict] = []
    for spec in specs:
        run_dir = study_dir / spec.name
        run_dir.mkdir(parents=True, exist_ok=True)
        config = build_config(spec, vocab_size, max_seq_len=max_seq_len)
        train_summary = train_tiny_lm(
            config=config,
            train_dataset=corpus_cache[args.train_seq_len].train,
            val_dataset=corpus_cache[args.train_seq_len].val,
            output_dir=run_dir,
            token_budget=args.train_token_budget,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device_name=args.device,
            eval_every=max(1, args.train_token_budget // max(args.batch_size * args.train_seq_len * 8, 1)),
            seed=args.seed,
        )

        device = torch.device(train_summary["manifest"]["device"])
        model = MinimalLLM(config).to(device)
        state = torch.load(run_dir / "model.pt", map_location=device)
        model.load_state_dict(state)
        model.eval()

        context_results = context_eval(
            model=model,
            corpus_cache=corpus_cache,
            eval_seq_lens=args.eval_seq_lens,
            device=device,
            max_batches=args.eval_batches,
        )

        record = {
            "name": spec.name,
            "block_size": spec.block_size,
            "top_k": spec.top_k,
            "train_summary": train_summary,
            "context_eval": context_results,
        }
        records.append(record)
        (run_dir / "study_summary.json").write_text(json.dumps(record, indent=2) + "\n")
        (run_dir / "context_eval.json").write_text(json.dumps(context_results, indent=2) + "\n")

    (study_dir / "study_results.json").write_text(json.dumps(records, indent=2) + "\n")
    with (study_dir / "study_results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "block_size", "top_k", "train_val_loss", "train_val_ppl", "train_tokens_per_second"])
        for row in records:
            final = row["train_summary"]["final"]
            writer.writerow([
                row["name"],
                row["block_size"],
                row["top_k"],
                final.get("val_loss", ""),
                final.get("val_ppl", ""),
                final.get("tokens_per_second", ""),
            ])
    plot_path = plot_summary(study_dir, records, args.eval_seq_lens)
    summary = {
        "study_dir": str(study_dir),
        "plot": str(plot_path),
        "records": records,
    }
    (study_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
