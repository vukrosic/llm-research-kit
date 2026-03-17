"""
Run batch 13: Radical topology changes, 5 min each.
"""
import os, sys, time, json, gc
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
import frontier.architectures.novel_noattn_v13

from configs.dataset_config import DataConfig
from data.loader import setup_tokenizer
from train_llm import prepare_datasets, worker_init_fn
from optimizers.muon import Muon
from utils.helpers import set_seed

RESULTS_DIR = ROOT / "frontier_results" / "novel_v13"


def setup_optimizer(model, muon_lr=0.024, adamw_lr=0.006, momentum=0.95, wd=0.2):
    groups = model.get_optimizer_groups()
    if len(groups) == 3:
        # V13_06: 3rd group is attention params with 2x LR
        muon_params, adamw_params, attn_params = groups
    else:
        muon_params, adamw_params = groups
        attn_params = []
    optimizers = []
    if muon_params:
        optimizers.append(Muon(muon_params, lr=muon_lr, momentum=momentum))
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=wd,
                                            fused=torch.cuda.is_available()))
    if attn_params:
        # Attention params get 2x Muon LR
        optimizers.append(Muon(attn_params, lr=muon_lr * 2, momentum=momentum))
    return optimizers


def train_timed(model, train_loader, val_loader, time_budget_sec, vocab_size, seed=42, grad_accum=2, grad_clip=1.0):
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

    set_seed(seed)
    model.train()
    step = 0; tokens_seen = 0; micro_step = 0
    train_iter = iter(train_loader)

    def get_batch():
        nonlocal train_iter
        try: return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    for _ in range(2):
        for _ in range(grad_accum):
            batch = get_batch()
            x = batch["input_ids"].to(device); y = batch["labels"].to(device)
            with autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), y[:, 1:].reshape(-1)) / grad_accum
            loss.backward()
        for opt in optimizers: opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    for opt in optimizers: opt.zero_grad()

    torch.cuda.synchronize()
    t_start = time.time(); deadline = t_start + time_budget_sec; last_print = t_start

    while True:
        batch = get_batch()
        x = batch["input_ids"].to(device); y = batch["labels"].to(device)
        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            shift_logits = logits[:, :-1].contiguous(); shift_labels = y[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            scaled_loss = loss / grad_accum
        scaled_loss.backward()
        tokens_seen += x.numel(); micro_step += 1

        if micro_step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            for opt in optimizers: opt.step(); opt.zero_grad()
            for sched in schedulers: sched.step()
            step += 1

        now = time.time()
        if now - last_print > 30:
            tps = tokens_seen / (now - t_start)
            print(f"    Step {step:5d} | loss={loss.item():.4f} | tokens={tokens_seen:,} | "
                  f"{tps:,.0f} tok/s | {deadline-now:.0f}s left")
            last_print = now
        if now >= deadline: break

    if micro_step % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        for opt in optimizers: opt.step(); opt.zero_grad()
        step += 1

    torch.cuda.synchronize()
    train_time = time.time() - t_start

    model.eval()
    total_loss = 0; total_tokens_eval = 0; total_correct = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 100: break
            x = batch["input_ids"].to(device); y = batch["labels"].to(device)
            with autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                sl = logits[:, :-1].contiguous(); sy = y[:, 1:].contiguous()
                vloss = F.cross_entropy(sl.view(-1, vocab_size), sy.view(-1))
            nt = sy.numel()
            total_loss += vloss.item() * nt; total_tokens_eval += nt
            total_correct += (sl.argmax(-1) == sy).sum().item()

    return {
        "val_loss": total_loss / total_tokens_eval,
        "val_accuracy": total_correct / total_tokens_eval,
        "train_loss": loss.item(), "tokens_seen": tokens_seen,
        "steps": step, "train_time_sec": train_time,
        "tokens_per_sec": tokens_seen / train_time,
        "seed": seed,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TIME_BUDGET = 300

    experiments = sorted([k for k in list_architectures() if k.startswith('V13_')])
    print(f"V11 batch: {len(experiments)} architectures, {TIME_BUDGET}s each")

    data_cfg = DataConfig(dataset_path="./processed_data", seq_length=2048, num_samples=20000, cache_dir="./hf_cache")
    tokenizer = setup_tokenizer(data_cfg)
    print("Loading dataset...")
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)

    g = torch.Generator(); g.manual_seed(42)
    loader_args = dict(batch_size=4, num_workers=2, pin_memory=True,
                       persistent_workers=True, worker_init_fn=worker_init_fn, generator=g)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    all_results = []
    for idx, arch_name in enumerate(experiments):
        result_file = RESULTS_DIR / f"{arch_name}.json"
        if result_file.exists():
            print(f"\n  SKIP {arch_name} (already done)")
            all_results.append(json.load(open(result_file)))
            continue

        print(f"\n{'='*70}")
        print(f" [{idx+1}/{len(experiments)}] {arch_name}")
        print(f"{'='*70}")

        cfg = FrontierConfig(d_model=512, n_layers=22, d_ff=2048, vocab_size=49152)
        try:
            set_seed(42)
            model = build_model(arch_name, cfg)
            n_params = model.count_parameters()
            print(f"  Parameters: {n_params:,}")
            print(f"  Description: {model.describe()}")

            result = train_timed(model, train_loader, val_loader, TIME_BUDGET, tokenizer.vocab_size)
            result["params"] = n_params; result["arch"] = arch_name
            print(f"\n  RESULT: val_loss={result['val_loss']:.4f} | "
                  f"val_acc={result['val_accuracy']:.4f} | "
                  f"tokens={result['tokens_seen']:,} | {result['tokens_per_sec']:,.0f} tok/s")

            with open(result_file, "w") as f: json.dump(result, f, indent=2)
            all_results.append(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            crash = {"arch": arch_name, "status": "crashed", "error": str(e)}
            with open(result_file, "w") as f: json.dump(crash, f, indent=2)
            all_results.append(crash)

        try: del model
        except: pass
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    valid = [r for r in all_results if "val_loss" in r]
    valid.sort(key=lambda x: x["val_loss"])

    print(f"\n{'='*70}")
    print(f" V13 RESULTS (5min)")
    print(f" Baselines: transformer=3.784, V10_best=3.665")
    print(f"{'='*70}")
    for i, r in enumerate(valid):
        d1 = r['val_loss'] - 3.784; d2 = r['val_loss'] - 3.665
        print(f" {i+1:3d}. {r['arch']:<30s} val={r['val_loss']:.4f} "
              f"(Δtrans={d1:+.3f} Δv10={d2:+.3f}) "
              f"{r['tokens_per_sec']:,.0f} tok/s  {r['params']:,} params")


if __name__ == "__main__":
    main()
