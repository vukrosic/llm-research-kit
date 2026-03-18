#!/usr/bin/env python3
"""Build and optionally launch the duration->LR scaling sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimization.run_experiments import run_batch


DEFAULT_QUEUE = Path("optimization/queue_duration_lr.json")
DEFAULT_RESULTS = Path("results/duration_lr_sweep")

LR_GRID = [0.005, 0.008, 0.012, 0.018, 0.028, 0.042]
NOISE_SEEDS = [42, 137, 256, 7, 1337, 2024, 999]
SWEEP_SEEDS_FULL = [42, 137]
SWEEP_SEEDS_BUDGETED = {
    5: [42, 137],
    15: [42],
    45: [42],
    80: [42, 137],
}


def lr_changes(muon_lr: float) -> dict:
    return {
        "batch_size": 4,
        "weight_decay": 0.2,
        "compile_model": False,
        "schedule_type": "constant",
        "warmup_steps": 50,
        "muon_lr": muon_lr,
        "adamw_lr": round(muon_lr / 4, 6),
    }


def build_queue(full: bool = False) -> list[dict]:
    queue: list[dict] = []

    # Step 1: seed noise floor
    for seed in NOISE_SEEDS:
        queue.append({
            "exp_id": f"noise20s_seed{seed}_lr0.012",
            "batch": "duration_lr_sweep",
            "status": "pending",
            "hypothesis": "Noise floor at 20s for muon_lr=0.012",
            "train_seconds": 20,
            "seed": seed,
            "changes": lr_changes(0.012),
        })

    # Step 2: sweep
    durations = [5, 15, 45, 80]
    for duration in durations:
        seeds = SWEEP_SEEDS_FULL if full else SWEEP_SEEDS_BUDGETED[duration]
        for muon_lr in LR_GRID:
            for seed in seeds:
                queue.append({
                    "exp_id": f"{duration}s_seed{seed}_lr{muon_lr:.3f}",
                    "batch": "duration_lr_sweep",
                    "status": "pending",
                    "hypothesis": "Duration->LR scaling sweep",
                    "train_seconds": duration,
                    "seed": seed,
                    "changes": lr_changes(muon_lr),
                })

    return queue


def estimate_wall_clock_seconds(queue: list[dict]) -> float:
    total = 0.0
    for exp in queue:
        duration = float(exp.get("train_seconds", 0))
        total += duration
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--full", action="store_true",
                        help="Run the exact 55-run plan (7 noise + 48 sweep).")
    parser.add_argument("--run", action="store_true",
                        help="Launch the queue immediately after writing it.")
    args = parser.parse_args()

    queue = build_queue(full=args.full)
    args.queue.parent.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    with args.queue.open("w") as f:
        json.dump(queue, f, indent=2)

    est_seconds = estimate_wall_clock_seconds(queue)
    print(f"Wrote {len(queue)} experiments to {args.queue}")
    print(f"Estimated compute time from requested durations: {est_seconds / 60:.1f} min")
    if not args.full:
        print("Using budgeted sweep: 2 seeds at 5s/80s, 1 seed at 15s/45s.")

    if args.run:
        run_batch(str(args.queue), str(args.results_dir))


if __name__ == "__main__":
    main()
