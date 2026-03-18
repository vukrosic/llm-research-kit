import argparse
import json
from pathlib import Path


def discover_json_files(inputs):
    files = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            files.append(path)
    return files


def load_record(path):
    data = json.loads(path.read_text())

    if "final_metrics" in data:
        history = data.get("history", {})
        config = data.get("experiment_config", {})
        summary = data.get("summary", {})
        val_losses = history.get("val_losses", [])
        steps = history.get("steps", [])
        warmup_ratio = config.get("warmup_ratio", 0.0)
        train_tokens = data.get("train_tokens")
        actual_steps = data.get("actual_steps")

        warmup_steps = 0
        if actual_steps is not None and warmup_ratio:
            warmup_steps = int(actual_steps * warmup_ratio)

        filtered_losses = [
            loss for step, loss in zip(steps, val_losses)
            if step >= warmup_steps and not (step == 0 and len(val_losses) > 1)
        ]
        mean_val = summary.get("mean_val_loss_after_warmup")
        if mean_val is None:
            mean_val = sum(filtered_losses) / len(filtered_losses) if filtered_losses else None

        return {
            "exp_id": path.parent.name,
            "path": str(path),
            "mean_val_loss_after_warmup": mean_val,
            "final_val_loss": data["final_metrics"]["val_loss"],
            "tokens_per_second": data.get("tokens_per_second"),
            "train_tokens": train_tokens,
            "seed": config.get("seed"),
            "model_config": config.get("model_config"),
        }

    if "val_loss" in data:
        return {
            "exp_id": data.get("exp_id", path.stem),
            "path": str(path),
            "mean_val_loss_after_warmup": data.get("val_loss"),
            "final_val_loss": data.get("val_loss"),
            "tokens_per_second": data.get("tokens_per_second"),
            "train_tokens": data.get("tokens_seen"),
            "seed": data.get("seed"),
            "model_config": data.get("model_config", "LLMConfig"),
        }

    return None


def main():
    parser = argparse.ArgumentParser(description="Rank experiment JSON files by sweep quality.")
    parser.add_argument("inputs", nargs="+", help="Metrics files or directories containing metrics/result JSON files")
    args = parser.parse_args()

    records = []
    for path in discover_json_files(args.inputs):
        record = load_record(path)
        if record is not None:
            records.append(record)

    if not records:
        raise SystemExit("No readable experiment JSON files found.")

    records.sort(
        key=lambda row: (
            float("inf") if row["mean_val_loss_after_warmup"] is None else row["mean_val_loss_after_warmup"],
            row["final_val_loss"],
            -(row["tokens_per_second"] or 0.0),
        )
    )

    print(
        f"{'rank':<4} {'exp_id':<28} {'mean_val':<12} {'final_val':<12} "
        f"{'tok/s':<10} {'seed':<6} {'model':<12} path"
    )
    for idx, row in enumerate(records, start=1):
        mean_val = "n/a" if row["mean_val_loss_after_warmup"] is None else f"{row['mean_val_loss_after_warmup']:.6f}"
        final_val = f"{row['final_val_loss']:.6f}"
        tok_s = "n/a" if row["tokens_per_second"] is None else f"{row['tokens_per_second']:.1f}"
        seed = "n/a" if row["seed"] is None else str(row["seed"])
        model = row["model_config"] or "n/a"
        print(
            f"{idx:<4} {row['exp_id']:<28} {mean_val:<12} {final_val:<12} "
            f"{tok_s:<10} {seed:<6} {model:<12} {row['path']}"
        )


if __name__ == "__main__":
    main()
