"""
Timed training for registry-based frontier architectures.
Usage:
    python frontier/run_registry_timed.py --arch OscillatoryRecurrenceLM --time 300
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
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".torchinductor_cache"))

from frontier.architectures.base import FrontierConfig
from frontier.architectures.registry import build_model, list_architectures

# Import all architecture modules
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
from train_llm import prepare_datasets, worker_init_fn
from optimizers.muon import Muon
from utils.helpers import set_seed


def setup_optimizer(model, muon_lr=0.024, adamw_lr=0.006, momentum=0.95, wd=0.2):
    muon_params, adamw_params = model.get_optimizer_groups()
    optimizers = []
    if muon_params:
        optimizers.append(Muon(muon_params, lr=muon_lr, momentum=momentum))
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=wd,
                                            fused=torch.cuda.is_available()))
    return optimizers


def train_timed(model, train_loader, val_loader, time_budget_sec, vocab_size, grad_clip=1.0):
    device = torch.device('cuda')
    model = model.to(device, dtype=torch.bfloat16)

    est_steps = int(time_budget_sec / 0.15)
    optimizers = setup_optimizer(model)

    # Scheduler
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
    train_iter = iter(train_loader)

    # Warmup (2 steps, not timed)
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
        tokens_seen += x.numel()

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
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--time", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--config", type=str, default="{}", help="JSON arch_config")
    args = parser.parse_args()

    import json as json_mod
    arch_config = json_mod.loads(args.config)
    if args.n_layers: arch_config["n_layers"] = args.n_layers
    if args.d_ff: arch_config["d_ff"] = args.d_ff

    print(f"{'='*70}")
    print(f" Architecture: {args.arch}")
    print(f" Time budget:  {args.time}s")
    print(f" Config:       {arch_config}")
    print(f"{'='*70}")

    cfg = FrontierConfig(
        d_model=args.d_model,
        n_layers=arch_config.get("n_layers", 22),
        d_ff=arch_config.get("d_ff", 2048),
        vocab_size=49152,
        arch_config=arch_config,
    )

    set_seed(42)
    model = build_model(args.arch, cfg)
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,}")
    print(f"  Description: {model.describe()}")

    breakdown = model.parameter_breakdown()
    for g, c in sorted(breakdown.items(), key=lambda x: -x[1])[:5]:
        print(f"    {g}: {c:,} ({100*c/n_params:.1f}%)")

    data_cfg = DataConfig(dataset_path="auto", seq_length=2048, num_samples=20000, cache_dir="./hf_cache")
    tokenizer = setup_tokenizer(data_cfg)
    print("\nLoading dataset...")
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)

    g = torch.Generator()
    g.manual_seed(42)
    loader_args = dict(
        batch_size=args.batch_size, num_workers=2, pin_memory=True,
        persistent_workers=True, worker_init_fn=worker_init_fn, generator=g,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    result = train_timed(model, train_loader, val_loader, args.time, tokenizer.vocab_size)
    result["params"] = n_params
    result["arch"] = args.arch

    print(f"\n{'='*70}")
    print(f" RESULT: {args.arch}")
    print(f"   val_loss:     {result['val_loss']:.4f}")
    print(f"   val_accuracy: {result['val_accuracy']:.4f}")
    print(f"   tokens_seen:  {result['tokens_seen']:,}")
    print(f"   steps:        {result['steps']}")
    print(f"   tokens/sec:   {result['tokens_per_sec']:,.0f}")
    print(f"   params:       {n_params:,}")
    print(f"{'='*70}")

    out_dir = ROOT / "frontier_results" / "timed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.arch}_{args.time}s.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
