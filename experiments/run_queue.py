"""Run experiments from experiments/queue.json with flags_override.

This script mirrors run_ablations.py but builds configs on the fly
from BaselineConfig + per-entry overrides.
"""

import json
import os
import time
import gc
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.ablation_configs import BaselineConfig
from configs.dataset_config import DataConfig
from data.loader import setup_tokenizer
from train_llm import prepare_datasets, worker_init_fn, print_system_info
from run_ablations import run_single_experiment, setup_logging
from utils.helpers import format_time

QUEUE_PATH = os.environ.get("QUEUE_PATH", "experiments/queue.json")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./ablation_results")

# Keep torch/inductor caches inside the workspace to avoid permission issues.
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".torchinductor_cache"))
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)


def build_config(entry):
    cfg = BaselineConfig()
    cfg.experiment_name = entry["exp_id"]
    cfg.train_tokens = entry.get("token_budget", cfg.train_tokens)

    overrides = entry.get("flags_override", {})
    for k, v in overrides.items():
        setattr(cfg, k, v)

    return cfg


def set_eval_schedule(cfg, tokens):
    tokens_per_step = cfg.batch_size * cfg.max_seq_len * cfg.gradient_accumulation_steps
    est_steps = max(1, tokens // tokens_per_step)

    if tokens <= 2_000_000:
        cfg.eval_milestones = (0, est_steps // 4, est_steps // 2, 3 * est_steps // 4)
        cfg.log_every = max(10, est_steps // 20)
        cfg.eval_every = None
    elif tokens <= 8_000_000:
        cfg.eval_milestones = (0, 50, 100, 150, 200, 300, 400)
        cfg.log_every = 50
        cfg.eval_every = None
    elif tokens <= 20_000_000:
        cfg.eval_milestones = (0, 100, 250, 500, 750, 1000)
        cfg.log_every = 100
        cfg.eval_every = None
    elif tokens <= 100_000_000:
        cfg.eval_milestones = (0, 500, 1000, 2000, 3000, 4000, 5000)
        cfg.log_every = 250
        cfg.eval_every = None
    else:
        cfg.eval_milestones = (0, 1000, 5000, 10000, 20000, 30000, 40000, 50000)
        cfg.log_every = 1000
        cfg.eval_every = None


def load_queue():
    with open(QUEUE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_queue(entries):
    with open(QUEUE_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
        f.write("\n")


def main():
    logger = setup_logging(log_dir="./logs")
    print_system_info()

    queue = load_queue()
    pending = [e for e in queue if e.get("status") == "pending"]
    if not pending:
        print("No pending experiments in queue.")
        return

    if not torch.cuda.is_available() and os.environ.get("ALLOW_CPU") != "1":
        print("⚠️ CUDA is not available. Running 6M-token experiments on CPU is impractical.")
        print("Set ALLOW_CPU=1 to proceed anyway, or run in a CUDA-enabled environment.")
        return

    # Shared dataset setup from first pending config
    first_cfg = build_config(pending[0])
    tokens = first_cfg.train_tokens

    avg_tokens_per_doc = 1000
    safety_factor = 2.0
    calc_num_docs = max(100, int((tokens / avg_tokens_per_doc) * safety_factor))

    data_cfg = DataConfig(
        dataset_path="auto",
        seq_length=first_cfg.max_seq_len,
        num_samples=calc_num_docs,
        cache_dir="./hf_cache",
    )

    tokenizer = setup_tokenizer(data_cfg)

    print("📂 Loading shared dataset...")
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)

    total_start = time.time()

    for entry in pending:
        exp_id = entry["exp_id"]
        print(f"\n{'#'*70}")
        print(f"  RUNNING: {exp_id}")
        print(f"{'#'*70}")

        # Mark running
        entry["status"] = "running"
        save_queue(queue)

        cfg = build_config(entry)
        cfg.vocab_size = tokenizer.vocab_size
        set_eval_schedule(cfg, cfg.train_tokens)

        g_exp = torch.Generator()
        g_exp.manual_seed(42)

        loader_args = dict(
            batch_size=cfg.batch_size,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
            generator=g_exp,
        )
        train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

        exp_output = os.path.join(OUTPUT_DIR, f"{cfg.train_tokens}tok", exp_id)
        metrics_path = os.path.join(exp_output, "metrics.json")

        if os.path.exists(metrics_path):
            print(f"⏩ [RESUME] Skipping {exp_id} - results found at {metrics_path}")
            entry["status"] = "done"
            save_queue(queue)
            continue

        try:
            run_single_experiment(
                config=cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=exp_output,
                use_compile=cfg.compile_model,
            )
            entry["status"] = "done"
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"❌ [CRASH] Experiment {exp_id} failed: {e}")
            crash_dir = Path(exp_output)
            crash_dir.mkdir(parents=True, exist_ok=True)
            crash_file = crash_dir / "CRASH.log"
            with open(crash_file, "w") as f:
                f.write(f"experiment: {exp_id}\n")
                f.write(f"tokens: {cfg.train_tokens}\n")
                f.write(f"error: {e}\n\n")
                f.write(tb)
            entry["status"] = "failed"

        save_queue(queue)

        # Cleanup
        del train_loader
        del val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    total_time = time.time() - total_start
    print(f"\n⏱️ Total queue time: {format_time(total_time)}")


if __name__ == "__main__":
    main()
