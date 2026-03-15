"""
Time-Based Training Runner
============================
Trains each architecture for a fixed wall-clock time (default: 4 minutes).
Measures final val_loss after the time is up.

This is the fairest comparison: architectures that are faster per step
get more training iterations, which rewards both quality and efficiency.

Usage:
    python frontier/run_timed.py                    # Run all 10, 4 min each
    python frontier/run_timed.py --time 120         # 2 min each
    python frontier/run_timed.py --arch butterfly   # Run one
"""

import os
import sys
import time
import json
import math
import gc
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".torchinductor_cache"))

from frontier.novel_archs import build_model, ALL_ARCHS
from configs.dataset_config import DataConfig
from data.loader import setup_tokenizer
from train_llm import prepare_datasets, worker_init_fn
from training.evaluation import evaluate_model
from optimizers.muon import Muon
from utils.helpers import set_seed


def setup_optimizer(model, muon_lr=0.024, adamw_lr=0.006, momentum=0.95, wd=0.2):
    """Split params into Muon (2D matrices) and AdamW (rest)."""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and 'emb' not in name and 'norm' not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    optimizers = []
    if muon_params:
        optimizers.append(Muon(muon_params, lr=muon_lr, momentum=momentum))
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=wd, fused=True))
    return optimizers


def setup_schedulers(optimizers, total_steps, warmup_ratio=0.02):
    """Linear schedule with warmup."""
    warmup = max(1, int(total_steps * warmup_ratio))
    schedulers = []
    for opt in optimizers:
        def lr_fn(step, w=warmup, t=total_steps):
            if step < w:
                return step / w
            return max(0.1, 1.0 - (step - w) / max(1, t - w))
        schedulers.append(torch.optim.lr_scheduler.LambdaLR(opt, lr_fn))
    return schedulers


def train_timed(
    model,
    train_loader,
    val_loader,
    time_budget_sec: float,
    vocab_size: int = 49152,
    grad_clip: float = 1.0,
    batch_size: int = 8,
    seq_len: int = 2048,
):
    """
    Train a model for a fixed wall-clock time.
    Returns dict with val_loss, tokens_seen, steps, tokens_per_sec.
    """
    device = torch.device('cuda')
    model = model.to(device, dtype=torch.bfloat16)

    # Estimate total steps for scheduler (rough: assume ~X ms per step)
    est_steps = int(time_budget_sec / 0.15)  # ~150ms per step estimate

    optimizers = setup_optimizer(model)
    schedulers = setup_schedulers(optimizers, est_steps)

    set_seed(42)

    model.train()
    step = 0
    tokens_seen = 0
    train_iter = iter(train_loader)

    # Warm up (2 steps, doesn't count toward time)
    print("    Warming up...", end=" ", flush=True)
    for _ in range(2):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), y[:, 1:].reshape(-1))
        loss.backward()
        for opt in optimizers:
            opt.step()
            opt.zero_grad()
    torch.cuda.synchronize()
    print("done")

    # Reset model to fresh state
    # (Skip full reset for speed — warmup is just 2 steps, negligible)
    for opt in optimizers:
        opt.zero_grad()

    # TIMED TRAINING
    torch.cuda.synchronize()
    t_start = time.time()
    deadline = t_start + time_budget_sec
    last_loss = 0.0
    last_print = t_start

    while True:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        batch_tokens = x.numel()

        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            shift_labels = torch.full_like(y, -100)
            shift_labels[:, :-1] = y[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), shift_labels.reshape(-1), ignore_index=-100)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        for opt in optimizers:
            opt.step()
            opt.zero_grad()
        for sched in schedulers:
            sched.step()

        step += 1
        tokens_seen += batch_tokens

        if step % 50 == 0:
            last_loss = loss.item()

        now = time.time()
        if now - last_print > 30:
            elapsed = now - t_start
            remaining = deadline - now
            tps = tokens_seen / elapsed
            print(f"    Step {step:5d} | loss={last_loss:.4f} | tokens={tokens_seen:,} | "
                  f"{tps:,.0f} tok/s | {remaining:.0f}s left")
            last_print = now

        if now >= deadline:
            break

    torch.cuda.synchronize()
    train_time = time.time() - t_start
    last_loss = loss.item()

    # Final evaluation
    model.eval()
    val_losses = []
    val_correct = 0
    val_total = 0
    eval_steps = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            with autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                shift_labels = torch.full_like(y, -100)
                shift_labels[:, :-1] = y[:, 1:]
                vloss = F.cross_entropy(logits.reshape(-1, vocab_size), shift_labels.reshape(-1), ignore_index=-100)
            val_losses.append(vloss.item())
            preds = logits.argmax(-1)
            mask = shift_labels != -100
            val_correct += (preds[mask] == shift_labels[mask]).sum().item()
            val_total += mask.sum().item()
            eval_steps += 1
            if eval_steps >= 100:
                break

    val_loss = sum(val_losses) / len(val_losses)
    val_acc = val_correct / max(val_total, 1)
    tokens_per_sec = tokens_seen / train_time

    return {
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "train_loss": last_loss,
        "tokens_seen": tokens_seen,
        "steps": step,
        "train_time_sec": train_time,
        "tokens_per_sec": tokens_per_sec,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=int, default=240, help="Training time per arch in seconds")
    parser.add_argument("--arch", type=str, default=None, help="Run a single architecture")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    archs = [args.arch] if args.arch else ALL_ARCHS

    print(f"{'='*70}")
    print(f" FRONTIER NOVEL ARCHITECTURE TOURNAMENT")
    print(f" Time budget: {args.time}s per architecture")
    print(f" Architectures: {len(archs)}")
    print(f"{'='*70}\n")

    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}\n")

    # Load data
    data_cfg = DataConfig(
        dataset_path="auto",
        seq_length=2048,
        num_samples=20000,  # plenty for 4 min training
        cache_dir="./hf_cache",
    )
    tokenizer = setup_tokenizer(data_cfg)
    vocab_size = tokenizer.vocab_size

    print("Loading dataset...")
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)

    results = {}
    total_start = time.time()

    for arch_name in archs:
        print(f"\n{'#'*70}")
        print(f"  ARCHITECTURE: {arch_name}")
        print(f"{'#'*70}")

        set_seed(42)
        model = build_model(arch_name)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        g = torch.Generator()
        g.manual_seed(42)
        loader_args = dict(
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
            generator=g,
        )
        train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

        try:
            result = train_timed(
                model, train_loader, val_loader,
                time_budget_sec=args.time,
                vocab_size=vocab_size,
                batch_size=args.batch_size,
            )
            result["params"] = n_params
            result["arch"] = arch_name
            results[arch_name] = result

            print(f"\n  RESULT: {arch_name}")
            print(f"    val_loss:      {result['val_loss']:.4f}")
            print(f"    val_accuracy:  {result['val_accuracy']:.4f}")
            print(f"    tokens_seen:   {result['tokens_seen']:,}")
            print(f"    steps:         {result['steps']}")
            print(f"    tokens/sec:    {result['tokens_per_sec']:,.0f}")

        except Exception as e:
            import traceback
            print(f"\n  CRASHED: {arch_name} — {e}")
            traceback.print_exc()
            results[arch_name] = {"error": str(e), "arch": arch_name, "params": n_params}

        # Cleanup
        del model, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    total_time = time.time() - total_start

    # ── Final Leaderboard ──
    print(f"\n{'='*70}")
    print(f" TOURNAMENT RESULTS ({args.time}s per architecture)")
    print(f"{'='*70}")
    print(f" Total time: {total_time/60:.1f} min\n")

    valid = {k: v for k, v in results.items() if "error" not in v}
    ranked = sorted(valid.items(), key=lambda x: x[1]["val_loss"])

    print(f"{'Rank':>4s}  {'Architecture':25s}  {'val_loss':>9s}  {'val_acc':>8s}  {'tokens':>12s}  {'tok/s':>10s}  {'steps':>6s}  {'params':>10s}")
    print("-" * 100)
    for i, (name, r) in enumerate(ranked, 1):
        print(f"{i:4d}  {name:25s}  {r['val_loss']:9.4f}  {r['val_accuracy']:8.4f}  {r['tokens_seen']:12,}  {r['tokens_per_sec']:10,.0f}  {r['steps']:6d}  {r['params']:10,}")

    if any("error" in v for v in results.values()):
        print("\nCrashed:")
        for name, r in results.items():
            if "error" in r:
                print(f"  {name}: {r['error']}")

    # Save results
    out_dir = ROOT / "frontier_results" / "timed_tournament"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"results_{args.time}s_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
