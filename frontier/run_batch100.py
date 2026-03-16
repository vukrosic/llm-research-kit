"""
Batch 100 Runner — Run all 100 architectures with 5-minute time budget each.
Usage:
    python frontier/run_batch100.py [--start N] [--end N] [--time 300]
"""

import os, sys, time, json, gc, argparse, math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".torchinductor_cache"))

from frontier.architectures.base import FrontierConfig
from frontier.architectures.registry import build_model, list_architectures
import frontier.architectures.batch100

from configs.dataset_config import DataConfig
from data.loader import setup_tokenizer
from train_llm import prepare_datasets, worker_init_fn
from optimizers.muon import Muon
from utils.helpers import set_seed


RESULTS_DIR = ROOT / "frontier_results" / "batch100"


def setup_optimizer(model, muon_lr=0.024, adamw_lr=0.006, momentum=0.95, wd=0.2):
    muon_params, adamw_params = model.get_optimizer_groups()
    optimizers = []
    if muon_params:
        optimizers.append(Muon(muon_params, lr=muon_lr, momentum=momentum))
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=wd,
                                            fused=torch.cuda.is_available()))
    return optimizers


def train_timed(model, train_loader, val_loader, time_budget_sec, vocab_size, grad_accum=2, grad_clip=1.0):
    device = torch.device('cuda')
    model = model.to(device, dtype=torch.bfloat16)

    est_steps = int(time_budget_sec / 0.15)
    optimizers = setup_optimizer(model)

    warmup = max(1, int(est_steps * 0.02))
    schedulers = []
    for opt in optimizers:
        def lr_fn(step, w=warmup, t=est_steps):
            if step < w: return step / w
            return max(0.1, 1.0 - (step - w) / max(1, t - w))
        schedulers.append(torch.optim.lr_scheduler.LambdaLR(opt, lr_fn))

    set_seed(42)
    model.train()
    step = 0
    tokens_seen = 0
    micro_step = 0
    train_iter = iter(train_loader)

    def get_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    # Warmup (2 steps, not timed)
    for _ in range(2):
        for _ in range(grad_accum):
            batch = get_batch()
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            with autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), y[:, 1:].reshape(-1))
                loss = loss / grad_accum
            loss.backward()
        for opt in optimizers:
            opt.step()
            opt.zero_grad()
    torch.cuda.synchronize()

    for opt in optimizers:
        opt.zero_grad()

    # TIMED TRAINING
    torch.cuda.synchronize()
    t_start = time.time()
    deadline = t_start + time_budget_sec
    last_print = t_start

    while True:
        batch = get_batch()
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            shift_labels = torch.full_like(y, -100)
            shift_labels[:, :-1] = y[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), shift_labels.reshape(-1), ignore_index=-100)
            scaled_loss = loss / grad_accum

        scaled_loss.backward()
        tokens_seen += x.numel()
        micro_step += 1

        if micro_step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            for opt in optimizers:
                opt.step()
                opt.zero_grad()
            for sched in schedulers:
                sched.step()
            step += 1

        now = time.time()
        if now - last_print > 30:
            elapsed = now - t_start
            remaining = deadline - now
            tps = tokens_seen / elapsed
            print(f"    Step {step:5d} | loss={loss.item():.4f} | tokens={tokens_seen:,} | "
                  f"{tps:,.0f} tok/s | {remaining:.0f}s left")
            last_print = now

        if now >= deadline:
            break

    # Final optimizer step if we have pending gradients
    if micro_step % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        for opt in optimizers:
            opt.step()
            opt.zero_grad()
        step += 1

    torch.cuda.synchronize()
    train_time = time.time() - t_start
    last_loss = loss.item()

    # Evaluation
    model.eval()
    val_losses = []
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 100: break
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

    return {
        "val_loss": sum(val_losses) / len(val_losses),
        "val_accuracy": val_correct / max(val_total, 1),
        "train_loss": last_loss,
        "tokens_seen": tokens_seen,
        "steps": step,
        "train_time_sec": train_time,
        "tokens_per_sec": tokens_seen / train_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, help="Start from arch N (1-100)")
    parser.add_argument("--end", type=int, default=100, help="End at arch N (1-100)")
    parser.add_argument("--time", type=int, default=300, help="Time budget per arch (seconds)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Get all batch100 architectures sorted by number
    all_archs = sorted([k for k in list_architectures() if k.startswith('B100_')])
    selected = [a for a in all_archs if args.start <= int(a.split('_')[1]) <= args.end]

    # Check which are already done
    todo = []
    for arch_name in selected:
        result_file = RESULTS_DIR / f"{arch_name}.json"
        if result_file.exists():
            print(f"  SKIP {arch_name} (already done)")
        else:
            todo.append(arch_name)

    print(f"\n{'='*70}")
    print(f" Batch 100 Runner")
    print(f" Architectures: {len(todo)} to run ({len(selected)-len(todo)} already done)")
    print(f" Time budget: {args.time}s per arch")
    print(f" Estimated total: {len(todo) * (args.time + 60) / 3600:.1f} hours")
    print(f"{'='*70}\n")

    if not todo:
        print("Nothing to do!")
        return

    # Setup shared data
    data_cfg = DataConfig(dataset_path="./processed_data", seq_length=2048, num_samples=20000, cache_dir="./hf_cache")
    tokenizer = setup_tokenizer(data_cfg)
    print("Loading dataset...")
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)

    g = torch.Generator()
    g.manual_seed(42)
    loader_args = dict(
        batch_size=args.batch_size, num_workers=2, pin_memory=True,
        persistent_workers=True, worker_init_fn=worker_init_fn, generator=g,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    # Summary tracker
    summary = []
    total_start = time.time()

    for idx, arch_name in enumerate(todo):
        print(f"\n{'='*70}")
        print(f" [{idx+1}/{len(todo)}] {arch_name}")
        print(f"{'='*70}")

        cfg = FrontierConfig(d_model=512, n_layers=22, d_ff=2048, vocab_size=49152)

        try:
            set_seed(42)
            model = build_model(arch_name, cfg)
            n_params = model.count_parameters()
            print(f"  Parameters: {n_params:,}")
            print(f"  Description: {model.describe()}")

            result = train_timed(model, train_loader, val_loader, args.time, tokenizer.vocab_size, grad_accum=args.grad_accum)
            result["params"] = n_params
            result["arch"] = arch_name

            print(f"\n  RESULT: val_loss={result['val_loss']:.4f} | "
                  f"val_acc={result['val_accuracy']:.4f} | "
                  f"tokens={result['tokens_seen']:,} | "
                  f"{result['tokens_per_sec']:,.0f} tok/s")

            # Save individual result
            result_file = RESULTS_DIR / f"{arch_name}.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)

            summary.append(result)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"  CRASHED: {e}")
            crash_result = {"arch": arch_name, "status": "crashed", "error": str(e), "traceback": tb}
            result_file = RESULTS_DIR / f"{arch_name}.json"
            with open(result_file, "w") as f:
                json.dump(crash_result, f, indent=2)
            summary.append(crash_result)

        # Cleanup
        try:
            del model
        except:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    total_time = time.time() - total_start

    # Print final leaderboard
    valid = [s for s in summary if "val_loss" in s]
    valid.sort(key=lambda x: x["val_loss"])

    print(f"\n{'='*70}")
    print(f" BATCH 100 LEADERBOARD (transformer baseline: 3.4486)")
    print(f" Total time: {total_time/3600:.1f} hours")
    print(f"{'='*70}")
    print(f" {'Rank':>4} | {'Architecture':<35} | {'val_loss':>8} | {'Δ':>7} | {'tok/s':>8} | {'params':>10}")
    print(f" {'-'*4} | {'-'*35} | {'-'*8} | {'-'*7} | {'-'*8} | {'-'*10}")
    for i, r in enumerate(valid[:30]):
        delta = r['val_loss'] - 3.4486
        print(f" {i+1:4d} | {r['arch']:<35} | {r['val_loss']:8.4f} | {delta:+7.4f} | {r['tokens_per_sec']:8,.0f} | {r['params']:10,}")

    crashed = [s for s in summary if "status" in s and s["status"] == "crashed"]
    if crashed:
        print(f"\n  Crashed: {len(crashed)}")
        for c in crashed:
            print(f"    {c['arch']}: {c['error'][:80]}")

    # Save full summary
    summary_file = RESULTS_DIR / "summary.json"
    with open(summary_file, "w") as f:
        json.dump({"results": summary, "total_time": total_time}, f, indent=2)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
