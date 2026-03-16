"""
Fair comparison: match baseline exactly (107M params, 12M tokens).
d=512, n=20, d_ff=2048 (~104M params), train until 12M tokens seen.
"""
import os, sys, time, json, gc, math
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
from frontier.architectures.registry import build_model
import frontier.architectures.batch100

from configs.dataset_config import DataConfig
from data.loader import setup_tokenizer
from train_llm import prepare_datasets, worker_init_fn
from optimizers.muon import Muon
from utils.helpers import set_seed

RESULTS_DIR = ROOT / "frontier_results" / "fair_compare"


def setup_optimizer(model, muon_lr=0.024, adamw_lr=0.006, momentum=0.95, wd=0.2):
    muon_params, adamw_params = model.get_optimizer_groups()
    optimizers = []
    if muon_params:
        optimizers.append(Muon(muon_params, lr=muon_lr, momentum=momentum))
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=wd,
                                            fused=torch.cuda.is_available()))
    return optimizers


def train_tokens(model, train_loader, val_loader, target_tokens, vocab_size, grad_accum=2, grad_clip=1.0):
    """Train until target_tokens seen (not time-based)."""
    device = torch.device('cuda')
    model = model.to(device, dtype=torch.bfloat16)

    # Estimate steps
    batch_tokens = 4 * 2048  # batch_size * seq_len
    est_steps = target_tokens // (batch_tokens * grad_accum)
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

    # Warmup (2 steps)
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

    # Train until target tokens
    torch.cuda.synchronize()
    t_start = time.time()
    last_print = t_start

    while tokens_seen < target_tokens:
        batch = get_batch()
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = y[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
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
        if now - last_print > 20:
            elapsed = now - t_start
            tps = tokens_seen / elapsed
            pct = 100 * tokens_seen / target_tokens
            print(f"    Step {step:5d} | loss={loss.item():.4f} | tokens={tokens_seen:,} ({pct:.0f}%) | "
                  f"{tps:,.0f} tok/s")
            last_print = now

    # Final step if pending
    if micro_step % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        for opt in optimizers:
            opt.step()
            opt.zero_grad()
        step += 1

    torch.cuda.synchronize()
    train_time = time.time() - t_start

    # Eval — match baseline eval method exactly: shift_logits[:, :-1] vs y[:, 1:]
    model.eval()
    total_loss = 0
    total_tokens_eval = 0
    total_correct = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 100: break
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            with autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                shift_logits = logits[:, :-1].contiguous()
                shift_labels = y[:, 1:].contiguous()
                vloss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))

            num_tokens = shift_labels.numel()
            total_loss += vloss.item() * num_tokens
            total_tokens_eval += num_tokens
            preds = shift_logits.argmax(-1)
            total_correct += (preds == shift_labels).sum().item()

    val_loss = total_loss / total_tokens_eval
    val_acc = total_correct / total_tokens_eval

    return {
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "train_loss": loss.item(),
        "tokens_seen": tokens_seen,
        "steps": step,
        "train_time_sec": train_time,
        "tokens_per_sec": tokens_seen / train_time,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_TOKENS = 12_000_000  # Match baseline exactly

    experiments = [
        ("fair_CosineAttnVR",   "B100_10_CosineAttnVRLM",  "Best from batch100"),
        ("fair_SigmoidAttnVR",  "B100_09_SigmoidAttnVRLM", "2nd from batch100"),
        ("fair_TSSEConvVR",     "B100_100_TSSEConvVRLM",   "3rd from batch100"),
        ("fair_TSVRBase",       "B100_25_TSVRBaseLM",       "Transformer-like baseline"),
    ]

    # Match baseline config: d=512, n=20, d_ff=2048 (~104M params)
    D_MODEL = 512
    N_LAYERS = 20
    D_FF = 2048

    data_cfg = DataConfig(dataset_path="./processed_data", seq_length=2048, num_samples=20000, cache_dir="./hf_cache")
    tokenizer = setup_tokenizer(data_cfg)
    print("Loading dataset...")
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)

    g = torch.Generator()
    g.manual_seed(42)
    loader_args = dict(
        batch_size=4, num_workers=2, pin_memory=True,
        persistent_workers=True, worker_init_fn=worker_init_fn, generator=g,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    all_results = []

    for name, arch_class, desc in experiments:
        result_file = RESULTS_DIR / f"{name}.json"
        if result_file.exists():
            print(f"\n  SKIP {name} (already done)")
            all_results.append(json.load(open(result_file)))
            continue

        print(f"\n{'='*70}")
        print(f" FAIR COMPARE: {name} — {desc}")
        print(f" d={D_MODEL}, L={N_LAYERS}, target={TARGET_TOKENS:,} tokens")
        print(f"{'='*70}")

        cfg = FrontierConfig(d_model=D_MODEL, n_layers=N_LAYERS, d_ff=D_FF, vocab_size=49152)

        try:
            set_seed(42)
            model = build_model(arch_class, cfg)
            n_params = model.count_parameters()
            print(f"  Parameters: {n_params:,} (baseline: 107,396,615)")

            result = train_tokens(model, train_loader, val_loader, TARGET_TOKENS, tokenizer.vocab_size)
            result["params"] = n_params
            result["arch"] = name
            result["arch_class"] = arch_class

            print(f"\n  RESULT: val_loss={result['val_loss']:.4f} | "
                  f"val_acc={result['val_accuracy']:.4f} | "
                  f"tokens={result['tokens_seen']:,} | "
                  f"time={result['train_time_sec']:.0f}s")

            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            all_results.append(result)

        except Exception as e:
            import traceback
            print(f"  CRASHED: {e}")
            traceback.print_exc()

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Final comparison
    valid = [r for r in all_results if "val_loss" in r]
    valid.sort(key=lambda x: x["val_loss"])

    baseline_val = 3.4486
    print(f"\n{'='*70}")
    print(f" FAIR COMPARISON (d=512, L=20, ~104M params, 12M tokens)")
    print(f" Original baseline: val_loss={baseline_val} (107M params, 12M tokens)")
    print(f"{'='*70}")
    for i, r in enumerate(valid):
        delta = r['val_loss'] - baseline_val
        marker = ">>>" if delta < -0.002 else "   "
        print(f" {marker}{i+1}. {r['arch']:<25s} val={r['val_loss']:.4f} ({delta:+.4f}) "
              f"params={r['params']:,} time={r['train_time_sec']:.0f}s "
              f"acc={r['val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
