"""
Frontier Experiment Runner
===========================
Runs frontier architecture experiments from frontier/experiments/queue.json.

Uses the same training infrastructure as the ablation system but with
the architecture registry to instantiate any registered model.
"""

import json
import os
import sys
import time
import gc
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from frontier.architectures.base import FrontierConfig
from frontier.architectures.registry import build_model, list_architectures

# Import all architecture modules to register them
import frontier.architectures.state_space
import frontier.architectures.linear_attention
import frontier.architectures.retention
import frontier.architectures.rwkv
import frontier.architectures.hybrid
import frontier.architectures.conv_mixer
import frontier.architectures.experimental
import frontier.architectures.wave_interference
import frontier.architectures.oscillatory_recurrence

from configs.dataset_config import DataConfig
from data.loader import setup_tokenizer
from train_llm import prepare_datasets, worker_init_fn, print_system_info
from training.trainer import train_model, EarlyStopping
from training.evaluation import evaluate_model
from optimizers.muon import Muon
from utils.helpers import set_seed, format_time

# Keep torch/inductor caches inside the workspace
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".torchinductor_cache"))
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)

QUEUE_PATH = os.environ.get("FRONTIER_QUEUE_PATH", str(ROOT / "frontier" / "experiments" / "queue.json"))
OUTPUT_DIR = os.environ.get("FRONTIER_OUTPUT_DIR", str(ROOT / "frontier_results"))


def build_config_from_entry(entry: dict) -> FrontierConfig:
    """Build a FrontierConfig from a queue entry."""
    cfg = FrontierConfig()

    # Standard overrides
    cfg.d_model = entry.get("arch_config", {}).get("d_model", 512)
    cfg.n_layers = entry.get("arch_config", {}).get("n_layers", 22)
    cfg.d_ff = entry.get("arch_config", {}).get("d_ff", 2048)
    cfg.train_tokens = entry.get("token_budget", 6000000)
    cfg.arch_family = entry.get("arch_family", "unknown")
    cfg.arch_config = entry.get("arch_config", {})

    # Training overrides
    train_cfg = entry.get("train_config", {})
    for k, v in train_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return cfg


def setup_optimizer_for_model(model, config: FrontierConfig):
    """
    Setup Muon + AdamW optimizer, using the model's optimizer groups.
    """
    muon_params, adamw_params = model.get_optimizer_groups()

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    optimizers = []

    if muon_params:
        muon_optimizer = Muon(
            muon_params,
            lr=config.muon_lr,
            momentum=config.muon_momentum,
        )
        optimizers.append(muon_optimizer)

    if adamw_params:
        adamw_optimizer = torch.optim.AdamW(
            adamw_params,
            lr=config.adamw_lr,
            weight_decay=config.weight_decay,
            fused=torch.cuda.is_available(),
        )
        optimizers.append(adamw_optimizer)

    return optimizers


def setup_schedulers(optimizers, config: FrontierConfig):
    """Setup LR schedulers matching the ablation system."""
    tokens_per_opt = config.batch_size * config.max_seq_len * config.gradient_accumulation_steps
    total_steps = config.train_tokens // tokens_per_opt
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    schedulers = []
    for optimizer in optimizers:
        if config.schedule_type == 'cosine':
            def lr_lambda(step, warmup=warmup_steps, total=total_steps):
                if step < warmup:
                    return step / warmup
                progress = (step - warmup) / max(1, total - warmup)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        elif config.schedule_type == 'linear':
            def lr_lambda(step, warmup=warmup_steps, total=total_steps):
                if step < warmup:
                    return step / warmup
                progress = (step - warmup) / max(1, total - warmup)
                return max(0.1, 1.0 - progress)
        else:
            def lr_lambda(step, warmup=warmup_steps):
                return step / warmup if step < warmup else 1.0

        schedulers.append(torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda))

    return schedulers


def set_eval_schedule(cfg, tokens):
    """Set evaluation milestones based on token budget."""
    tokens_per_step = cfg.batch_size * cfg.max_seq_len * cfg.gradient_accumulation_steps
    est_steps = max(1, tokens // tokens_per_step)

    if tokens <= 8_000_000:
        cfg.eval_milestones = (0, 50, 100, 150, 200, 300, 400)
        cfg.log_every = 50
    elif tokens <= 20_000_000:
        cfg.eval_milestones = (0, 100, 250, 500, 750, 1000)
        cfg.log_every = 100
    else:
        cfg.eval_milestones = (0, 500, 1000, 2000, 3000, 4000, 5000)
        cfg.log_every = 250

    cfg.eval_every = None


def run_single_frontier_experiment(
    entry: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str,
):
    """Run a single frontier experiment."""
    exp_id = entry["exp_id"]
    arch_class = entry["arch_class"]

    print(f"\n{'='*70}")
    print(f"  FRONTIER EXPERIMENT: {exp_id}")
    print(f"  Architecture: {arch_class} ({entry.get('arch_family', '?')})")
    print(f"{'='*70}")

    cfg = build_config_from_entry(entry)
    set_eval_schedule(cfg, cfg.train_tokens)

    # vocab_size is set from the tokenizer in main(); pass it via entry
    # Default 49152 matches SmolLM2-135M tokenizer
    cfg.vocab_size = entry.get("vocab_size", 49152)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    set_seed(42)
    model = build_model(arch_class, cfg)
    total_params = model.count_parameters()
    print(f"  Total parameters: {total_params:,}")
    print(f"  Architecture: {model.describe()}")
    print(f"  Sequence complexity: {model.sequence_mixing_complexity()}")
    print(f"  Recurrent inference: {model.supports_recurrent_inference()}")

    # Parameter breakdown
    breakdown = model.parameter_breakdown()
    for group, count in sorted(breakdown.items(), key=lambda x: -x[1])[:5]:
        print(f"    {group}: {count:,} ({100*count/total_params:.1f}%)")

    model = model.to(device, dtype=torch.bfloat16)

    # Try to compile
    compile_success = False
    if cfg.compile_model:
        try:
            model = torch.compile(model)
            compile_success = True
            print("  torch.compile: success")
        except Exception as e:
            print(f"  torch.compile: failed ({e}), using eager mode")

    # Setup optimizer
    set_seed(42)
    optimizers = setup_optimizer_for_model(
        model._orig_mod if compile_success else model, cfg
    )
    schedulers = setup_schedulers(optimizers, cfg)

    # Reset seed for reproducible training
    set_seed(42)

    # Train
    start_time = time.time()
    results = train_model(
        model=model,
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizers=optimizers,
        schedulers=schedulers,
        early_stopper=None,
        output_dir=output_dir,
        extra_config={
            "exp_id": exp_id,
            "arch_class": arch_class,
            "arch_family": entry.get("arch_family", "unknown"),
            "arch_config": entry.get("arch_config", {}),
            "hypothesis": entry.get("hypothesis", ""),
            "total_params": total_params,
            "sequence_complexity": model.sequence_mixing_complexity() if not compile_success else "unknown",
            "recurrent_inference": model.supports_recurrent_inference() if not compile_success else False,
            "compiled": compile_success,
        },
        log_every=getattr(cfg, 'log_every', 50),
    )

    wall_time = time.time() - start_time
    final = results['final_metrics']

    print(f"\n  Results for {exp_id}:")
    print(f"    val_loss:   {final['val_loss']:.4f}")
    print(f"    val_acc:    {final['val_accuracy']:.4f}")
    print(f"    wall_time:  {format_time(wall_time)}")
    print(f"    params:     {total_params:,}")

    return results


def load_queue():
    with open(QUEUE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_queue(entries):
    with open(QUEUE_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
        f.write("\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Frontier architecture experiment runner")
    parser.add_argument("--exp", type=str, help="Run a specific experiment by exp_id")
    parser.add_argument("--list", action="store_true", help="List registered architectures")
    args = parser.parse_args()

    if args.list:
        print("\nRegistered architectures:")
        for name, info in sorted(list_architectures().items()):
            print(f"  {name:20s} [{info['family']:15s}] {info['description']}")
        return

    print_system_info()

    # Load queue
    queue = load_queue()
    if args.exp:
        pending = [e for e in queue if e["exp_id"] == args.exp and e.get("status") != "done"]
    else:
        pending = [e for e in queue if e.get("status") == "pending"]

    pending.sort(key=lambda e: e.get("priority", 3))

    if not pending:
        print("No pending experiments in frontier queue.")
        return

    if not torch.cuda.is_available() and os.environ.get("ALLOW_CPU") != "1":
        print("CUDA not available. Set ALLOW_CPU=1 to run on CPU.")
        return

    # Shared dataset
    first_cfg = build_config_from_entry(pending[0])
    data_cfg = DataConfig(
        dataset_path="auto",
        seq_length=first_cfg.max_seq_len,
        num_samples=max(100, int((first_cfg.train_tokens / 1000) * 2.0)),
        cache_dir="./hf_cache",
    )

    tokenizer = setup_tokenizer(data_cfg)
    print("Loading shared dataset...")
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)

    total_start = time.time()

    for entry in pending:
        exp_id = entry["exp_id"]
        arch_class = entry["arch_class"]

        # Validate architecture exists
        if arch_class not in list_architectures():
            print(f"Unknown architecture '{arch_class}' for {exp_id}, skipping.")
            entry["status"] = "failed"
            save_queue(queue)
            continue

        # Mark running
        entry["status"] = "running"
        save_queue(queue)

        cfg = build_config_from_entry(entry)
        cfg.vocab_size = tokenizer.vocab_size

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
            print(f"Skipping {exp_id} - results found at {metrics_path}")
            entry["status"] = "done"
            save_queue(queue)
            continue

        try:
            run_single_frontier_experiment(entry, train_loader, val_loader, exp_output)
            entry["status"] = "done"
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"CRASH: {exp_id} failed: {e}")
            crash_dir = Path(exp_output)
            crash_dir.mkdir(parents=True, exist_ok=True)
            with open(crash_dir / "CRASH.log", "w") as f:
                f.write(f"experiment: {exp_id}\n")
                f.write(f"arch_class: {arch_class}\n")
                f.write(f"error: {e}\n\n")
                f.write(tb)
            entry["status"] = "failed"

        save_queue(queue)

        # Cleanup
        del train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    total_time = time.time() - total_start
    print(f"\nTotal frontier queue time: {format_time(total_time)}")


if __name__ == "__main__":
    main()
