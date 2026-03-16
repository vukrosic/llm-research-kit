"""
Scale-up comparison: top 3 architectures vs transformer baseline.
Larger model (d=640, 24 layers) with 8-minute training budget.
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
from frontier.architectures.registry import build_model, list_architectures
import frontier.architectures.batch100

from configs.dataset_config import DataConfig
from data.loader import setup_tokenizer
from train_llm import prepare_datasets, worker_init_fn
from optimizers.muon import Muon
from utils.helpers import set_seed

RESULTS_DIR = ROOT / "frontier_results" / "scaleup"


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

    est_steps = int(time_budget_sec / 0.2)
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TIME_BUDGET = 480  # 8 minutes

    # Scale-up configs: (name, arch_class, d_model, n_layers, d_ff, description)
    experiments = [
        # Top 3 from batch100
        ("scaleup_CosineAttnVR",    "B100_10_CosineAttnVRLM",   640, 24, 2560, "Best: cosine attn + value residual"),
        ("scaleup_SigmoidAttnVR",   "B100_09_SigmoidAttnVRLM",  640, 24, 2560, "2nd: sigmoid attn + value residual"),
        ("scaleup_TSSEConvVR",      "B100_100_TSSEConvVRLM",    640, 24, 2560, "3rd: TS SE conv + VR attn"),
        # Transformer baseline (pure GQA with conv, similar to winning pattern)
        ("scaleup_TSVRBase",        "B100_25_TSVRBaseLM",       640, 24, 2560, "Baseline: TS conv + VR standard GQA"),
        # Also try pure attention baseline
        ("scaleup_PureAttn",        "B100_30_PureAttnLM",       640, 24, 2560, "Baseline: pure GQA attention"),
    ]

    # Setup data
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

    for name, arch_class, d_model, n_layers, d_ff, desc in experiments:
        result_file = RESULTS_DIR / f"{name}.json"
        if result_file.exists():
            print(f"\n  SKIP {name} (already done)")
            all_results.append(json.load(open(result_file)))
            continue

        print(f"\n{'='*70}")
        print(f" SCALE-UP: {name}")
        print(f" {desc}")
        print(f" d={d_model}, L={n_layers}, d_ff={d_ff}, time={TIME_BUDGET}s")
        print(f"{'='*70}")

        cfg = FrontierConfig(d_model=d_model, n_layers=n_layers, d_ff=d_ff, vocab_size=49152)

        try:
            set_seed(42)
            model = build_model(arch_class, cfg)
            n_params = model.count_parameters()
            print(f"  Parameters: {n_params:,}")

            result = train_timed(model, train_loader, val_loader, TIME_BUDGET, tokenizer.vocab_size)
            result["params"] = n_params
            result["arch"] = name
            result["arch_class"] = arch_class
            result["d_model"] = d_model
            result["n_layers"] = n_layers

            print(f"\n  RESULT: val_loss={result['val_loss']:.4f} | "
                  f"val_acc={result['val_accuracy']:.4f} | "
                  f"tokens={result['tokens_seen']:,} | "
                  f"{result['tokens_per_sec']:,.0f} tok/s")

            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)

            all_results.append(result)

        except Exception as e:
            import traceback
            print(f"  CRASHED: {e}")
            traceback.print_exc()
            crash = {"arch": name, "status": "crashed", "error": str(e)}
            with open(result_file, "w") as f:
                json.dump(crash, f, indent=2)
            all_results.append(crash)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Print final comparison
    valid = [r for r in all_results if "val_loss" in r]
    valid.sort(key=lambda x: x["val_loss"])

    print(f"\n{'='*70}")
    print(f" SCALE-UP COMPARISON (d=640, L=24, 8min)")
    print(f"{'='*70}")
    print(f" {'Rank':>4} | {'Name':<30} | {'val_loss':>8} | {'tok/s':>8} | {'params':>10}")
    print(f" {'-'*4} | {'-'*30} | {'-'*8} | {'-'*8} | {'-'*10}")
    for i, r in enumerate(valid):
        print(f" {i+1:4d} | {r['arch']:<30} | {r['val_loss']:8.4f} | {r['tokens_per_sec']:8,.0f} | {r['params']:10,}")


if __name__ == "__main__":
    main()
