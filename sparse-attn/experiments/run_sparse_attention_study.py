from __future__ import annotations

import argparse
import csv
import json
import math
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
from experiments.train_tiny_real_compare import read_metrics
from models.llm import MinimalLLM
from training.tiny_trainer import evaluate, train_tiny_lm


@dataclass(frozen=True)
class StudySpec:
    name: str
    scale: str
    d_model: int
    d_ff: int
    n_layers: int
    attention_impl: str
    pooling: str = "max"
    router_source: str = "separate"


def build_config(spec: StudySpec, vocab_size: int, max_seq_len: int) -> LLMConfig:
    index_dim = spec.d_model // 6
    sparse = MiniMaxSparseConfig(
        block_size=16,
        top_k=2,
        index_dim=index_dim,
        pooling=spec.pooling,
        router_source=spec.router_source,
        dropout=0.0,
    )
    return LLMConfig(
        d_model=spec.d_model,
        n_heads=6,
        n_kv_heads=2,
        n_layers=spec.n_layers,
        d_ff=spec.d_ff,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        dropout=0.0,
        attention_impl=spec.attention_impl,
        minimax_sparse=sparse,
    )


def build_specs() -> list[StudySpec]:
    scales = [
        ("current", 96, 384, 6),
        ("mid", 144, 576, 6),
        ("large", 192, 768, 6),
    ]
    specs: list[StudySpec] = []
    for scale_name, d_model, d_ff, n_layers in scales:
        specs.append(
            StudySpec(
                name=f"{scale_name}_dense",
                scale=scale_name,
                d_model=d_model,
                d_ff=d_ff,
                n_layers=n_layers,
                attention_impl="dense",
            )
        )
        if scale_name == "mid":
            specs.extend(
                [
                    StudySpec(
                        name="mid_sparse_max",
                        scale=scale_name,
                        d_model=d_model,
                        d_ff=d_ff,
                        n_layers=n_layers,
                        attention_impl="minimax_sparse",
                        pooling="max",
                        router_source="separate",
                    ),
                    StudySpec(
                        name="mid_sparse_mean",
                        scale=scale_name,
                        d_model=d_model,
                        d_ff=d_ff,
                        n_layers=n_layers,
                        attention_impl="minimax_sparse",
                        pooling="mean",
                        router_source="separate",
                    ),
                    StudySpec(
                        name="mid_sparse_lse",
                        scale=scale_name,
                        d_model=d_model,
                        d_ff=d_ff,
                        n_layers=n_layers,
                        attention_impl="minimax_sparse",
                        pooling="logsumexp",
                        router_source="separate",
                    ),
                    StudySpec(
                        name="mid_sparse_shared",
                        scale=scale_name,
                        d_model=d_model,
                        d_ff=d_ff,
                        n_layers=n_layers,
                        attention_impl="minimax_sparse",
                        pooling="max",
                        router_source="group_mean_q",
                    ),
                ]
            )
        else:
            specs.append(
                StudySpec(
                    name=f"{scale_name}_sparse",
                    scale=scale_name,
                    d_model=d_model,
                    d_ff=d_ff,
                    n_layers=n_layers,
                    attention_impl="minimax_sparse",
                    pooling="max",
                    router_source="separate",
                )
            )
    return specs


def context_eval(model: MinimalLLM, spec: StudySpec, corpus_cache: dict[int, object], eval_seq_lens: list[int], device: torch.device, max_batches: int) -> dict[str, dict]:
    results = {}
    for seq_len in eval_seq_lens:
        corpus = corpus_cache[seq_len]
        loader = DataLoader(corpus.val, batch_size=4, shuffle=False, num_workers=0)
        results[str(seq_len)] = evaluate(model, loader, device, max_batches=max_batches)
    return results


def plot_summary(study_dir: Path, rows: list[dict], eval_seq_lens: list[int]) -> Path:
    plot_path = study_dir / "study_summary.png"
    scales = ["current", "mid", "large"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, scale in zip(axes, scales):
        subset = [r for r in rows if r["scale"] == scale]
        for row in subset:
            xs = [int(k) for k in row["context_eval"].keys()]
            ys = [row["context_eval"][str(x)]["val_loss"] for x in xs]
            style = "-" if row["attention_impl"] == "dense" else "--"
            ax.plot(xs, ys, style, marker="o", label=row["name"])
        ax.set_title(scale)
        ax.set_xlabel("Context length")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(eval_seq_lens)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Validation loss")
    fig.suptitle("Sparse attention study: scale, architecture, and context length", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sparse-attention study across scale, architecture, and context.")
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
    parser.add_argument("--run_root", type=Path, default=Path("runs/sparse_attention_study"))
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

    specs = build_specs()
    records: list[dict] = []
    manifest = {
        "dataset_path": args.dataset_path,
        "dataset_token_budget": args.dataset_token_budget,
        "train_seq_len": args.train_seq_len,
        "eval_seq_lens": args.eval_seq_lens,
        "train_token_budget": args.train_token_budget,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "eval_batches": args.eval_batches,
        "scales": [{"name": s.scale, "d_model": s.d_model, "d_ff": s.d_ff, "n_layers": s.n_layers} for s in specs if s.attention_impl == "dense"],
    }
    (study_dir / "study_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

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
            spec=spec,
            corpus_cache=corpus_cache,
            eval_seq_lens=args.eval_seq_lens,
            device=device,
            max_batches=args.eval_batches,
        )

        eval_path = run_dir / "context_eval.json"
        eval_path.write_text(json.dumps(context_results, indent=2) + "\n")
        record = {
            "name": spec.name,
            "scale": spec.scale,
            "attention_impl": spec.attention_impl,
            "pooling": spec.pooling,
            "router_source": spec.router_source,
            "train_summary": train_summary,
            "context_eval": context_results,
        }
        records.append(record)
        (run_dir / "study_summary.json").write_text(json.dumps(record, indent=2) + "\n")

    rows_path = study_dir / "study_results.json"
    rows_path.write_text(json.dumps(records, indent=2) + "\n")
    csv_path = study_dir / "study_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name",
            "scale",
            "attention_impl",
            "pooling",
            "router_source",
            "train_tokens_seen",
            "train_val_loss",
            "train_val_ppl",
            "train_val_acc",
        ])
        for row in records:
            final = row["train_summary"]["final"]
            writer.writerow([
                row["name"],
                row["scale"],
                row["attention_impl"],
                row["pooling"],
                row["router_source"],
                final.get("tokens_seen", ""),
                final.get("val_loss", ""),
                final.get("val_ppl", ""),
                final.get("val_acc", ""),
            ])

    plot_path = plot_summary(study_dir, records, args.eval_seq_lens)
    summary = {
        "study_dir": str(study_dir),
        "plot": str(plot_path),
        "results_json": str(rows_path),
        "results_csv": str(csv_path),
        "records": records,
    }
    (study_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
