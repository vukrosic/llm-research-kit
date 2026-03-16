"""
Speed-Gated Novel Architecture Runner
=======================================
1. Load transformer baseline + 5 novel architectures
2. Speed benchmark: forward+backward on 10 batches
3. Discard anything >2x slower than transformer
4. Train survivors for 6M tokens
5. Update leaderboard and loop
"""

import sys
import os
import time
import json
import gc
import math
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
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)

from frontier.architectures.base import FrontierConfig, EmbeddingWithScale, LMHead
from frontier.architectures.registry import build_model, REGISTRY

# Import novel architectures to register them
import frontier.architectures.novel_v2
import frontier.architectures.novel_v2_fixed
import frontier.architectures.novel_v3
import frontier.architectures.novel_v4
import frontier.architectures.novel_v5
import frontier.architectures.novel_v6
import frontier.architectures.novel_v7
import frontier.architectures.novel_v8
import frontier.architectures.novel_v9
import frontier.architectures.novel_v10
import frontier.architectures.novel_v11
import frontier.architectures.novel_v12
import frontier.architectures.novel_v13
import frontier.architectures.novel_v14
import frontier.architectures.novel_v15
import frontier.architectures.novel_v16
import frontier.architectures.novel_v17
import frontier.architectures.novel_v18

from configs.dataset_config import DataConfig
from configs.llm_config import LLMConfig
from data.loader import setup_tokenizer
from train_llm import prepare_datasets, worker_init_fn
from training.evaluation import evaluate_model
from optimizers.muon import Muon
from utils.helpers import set_seed, format_time

OUTPUT_DIR = ROOT / "frontier_results"
LEADERBOARD_PATH = ROOT / "frontier" / "experiments" / "leaderboard.md"


# ─── Transformer baseline as FrontierModel ───

from frontier.architectures.registry import register_arch
from frontier.architectures.base import FrontierModel

@register_arch("TransformerBaseline", "transformer", "Standard transformer baseline for comparison")
class TransformerBaseline(FrontierModel):
    """Wraps the existing MinimalLLM as a FrontierModel."""
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        d = config.d_model
        n_heads = config.arch_config.get("n_heads", 8)
        n_kv = config.arch_config.get("n_kv_heads", 4)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)

        from models.layers import TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(d, n_heads, config.d_ff, config.max_seq_len, config.dropout, n_kv_heads=n_kv)
            for _ in range(config.n_layers)
        ])
        self.norm = torch.nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))

    @classmethod
    def arch_family(cls): return "transformer"
    def describe(self): return f"Transformer: {self.config.n_layers}L x {self.config.d_model}d"
    def sequence_mixing_complexity(self): return "O(n^2)"

import torch.nn as nn

# ─── Experiment definitions ───

EXPERIMENTS = [
    {
        "exp_id": "novel_differential_gqa_12M",
        "arch_class": "DifferentialGQALM",
        "arch_family": "novel",
        "n_layers": 14,
        "d_model": 640,
        "d_ff": 2176,
        "train_tokens": 12_000_000,
        "arch_config": {"n_heads": 8, "n_kv_heads": 4, "use_bias": True, "residual_scale": 1.0},
        "hypothesis": "Differential attention (noise-canceling via subtraction) + conv",
    },
    {
        "exp_id": "novel_decay_mask_gqa_12M",
        "arch_class": "DecayMaskGQALM",
        "arch_family": "novel",
        "n_layers": 14,
        "d_model": 640,
        "d_ff": 2176,
        "train_tokens": 12_000_000,
        "arch_config": {"n_heads": 8, "n_kv_heads": 4, "use_bias": True, "residual_scale": 1.0},
        "hypothesis": "Soft exponential decay instead of hard window cutoff (RetNet-inspired)",
    },
    {
        "exp_id": "novel_conv_ts_qknorm_12M",
        "arch_class": "ConvTSQKNormLM",
        "arch_family": "novel",
        "n_layers": 14,
        "d_model": 640,
        "d_ff": 2176,
        "train_tokens": 12_000_000,
        "arch_config": {"n_heads": 8, "n_kv_heads": 4, "use_bias": True, "residual_scale": 1.0},
        "hypothesis": "Token shift + QK-norm combined (both helped individually, never tested together)",
    },
    {
        "exp_id": "novel_soft_router_12M",
        "arch_class": "SoftRouterLM",
        "arch_family": "novel",
        "n_layers": 10,
        "d_model": 640,
        "d_ff": 2176,
        "train_tokens": 12_000_000,
        "arch_config": {"n_heads": 8, "n_kv_heads": 4, "use_bias": True, "residual_scale": 1.0},
        "hypothesis": "Per-token soft routing between conv and attn in EVERY layer (novel topology)",
    },
    {
        "exp_id": "novel_value_residual_12M",
        "arch_class": "ValueResidualLM",
        "arch_family": "novel",
        "n_layers": 14,
        "d_model": 640,
        "d_ff": 2176,
        "train_tokens": 12_000_000,
        "arch_config": {"n_heads": 8, "n_kv_heads": 4, "use_bias": True, "residual_scale": 1.0},
        "hypothesis": "Raw embedding fed into value projections across all attn layers (gradient highway)",
    },
]


def make_config(exp: dict, vocab_size: int = 49152) -> FrontierConfig:
    cfg = FrontierConfig()
    cfg.vocab_size = vocab_size
    cfg.d_model = exp.get("d_model", 512)
    cfg.n_layers = exp["n_layers"]
    cfg.d_ff = exp.get("d_ff", 2048)
    cfg.arch_family = exp["arch_family"]
    cfg.arch_config = exp["arch_config"]
    cfg.train_tokens = exp.get("train_tokens", 6_000_000)
    cfg.compile_model = False  # avoid compile overhead for speed test
    cfg.batch_size = 8
    cfg.max_seq_len = 2048
    return cfg


def speed_benchmark(model, device, batch_size=8, seq_len=2048, vocab_size=49152, n_steps=10):
    """Benchmark forward+backward speed. Returns ms/step."""
    model.train()
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(3):
        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(n_steps):
        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    elapsed = time.time() - start
    ms_per_step = (elapsed / n_steps) * 1000
    return ms_per_step


def setup_optimizer(model, config):
    muon_params, adamw_params = model.get_optimizer_groups()
    optimizers = []
    if muon_params:
        optimizers.append(Muon(muon_params, lr=config.muon_lr, momentum=config.muon_momentum))
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=config.adamw_lr,
                                             weight_decay=config.weight_decay, fused=True))
    return optimizers


def setup_schedulers(optimizers, config):
    tokens_per_step = config.batch_size * config.max_seq_len
    total_steps = config.train_tokens // tokens_per_step
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    schedulers = []
    for opt in optimizers:
        def lr_lambda(step, w=warmup_steps, t=total_steps):
            if step < w: return step / w
            progress = (step - w) / max(1, t - w)
            return max(0.1, 1.0 - progress)
        schedulers.append(torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda))
    return schedulers


def train_one(model, config, train_loader, val_loader, exp_id, output_dir):
    """Minimal training loop — no dependency on trainer.py's LLMConfig."""
    device = next(model.parameters()).device
    optimizers = setup_optimizer(model, config)
    schedulers = setup_schedulers(optimizers, config)

    tokens_per_step = config.batch_size * config.max_seq_len
    total_steps = config.train_tokens // tokens_per_step
    eval_at = {0, 50, 100, 150, 200, 300, total_steps}

    history = {"steps": [], "val_losses": [], "train_losses": []}
    best_val = float('inf')

    model.train()
    step = 0
    tokens_seen = 0
    train_loss_accum = 0.0

    print(f"\n  Training {exp_id} for {total_steps} steps ({config.train_tokens:,} tokens)...")
    t0 = time.time()

    while tokens_seen < config.train_tokens:
        for batch in train_loader:
            if tokens_seen >= config.train_tokens:
                break

            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            with autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                shift_labels = torch.full_like(y, -100)
                shift_labels[:, :-1] = y[:, 1:]
                loss = F.cross_entropy(logits.view(-1, config.vocab_size),
                                       shift_labels.view(-1), ignore_index=-100)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)
            for sched in schedulers:
                sched.step()

            train_loss_accum += loss.item()
            tokens_seen += x.numel()

            if step in eval_at or step % 100 == 0:
                # Quick eval
                model.eval()
                val_loss_sum, val_tokens = 0.0, 0
                with torch.no_grad():
                    for vi, vb in enumerate(val_loader):
                        if vi >= 50: break
                        vx = vb["input_ids"].to(device)
                        vy = vb["labels"].to(device)
                        with autocast('cuda', dtype=torch.bfloat16):
                            vlogits = model(vx)
                            sl = vy[:, 1:].contiguous()
                            vloss = F.cross_entropy(vlogits[:, :-1].contiguous().view(-1, config.vocab_size),
                                                     sl.view(-1))
                        val_loss_sum += vloss.item() * sl.numel()
                        val_tokens += sl.numel()

                vl = val_loss_sum / max(val_tokens, 1)
                tl = train_loss_accum / max(1, step % 100 + 1) if step > 0 else loss.item()
                history["steps"].append(step)
                history["val_losses"].append(vl)
                history["train_losses"].append(tl)
                best_val = min(best_val, vl)

                elapsed = time.time() - t0
                print(f"    step {step:4d}/{total_steps} | val_loss {vl:.4f} | train_loss {tl:.4f} | {elapsed:.0f}s")
                model.train()
                train_loss_accum = 0.0

            step += 1

    elapsed = time.time() - t0

    # Final eval
    model.eval()
    val_loss_sum, val_tokens, val_correct = 0.0, 0, 0
    with torch.no_grad():
        for vi, vb in enumerate(val_loader):
            if vi >= 100: break
            vx = vb["input_ids"].to(device)
            vy = vb["labels"].to(device)
            with autocast('cuda', dtype=torch.bfloat16):
                vlogits = model(vx)
                sl = vy[:, 1:].contiguous()
                vloss = F.cross_entropy(vlogits[:, :-1].contiguous().view(-1, config.vocab_size),
                                         sl.view(-1))
            val_loss_sum += vloss.item() * sl.numel()
            val_tokens += sl.numel()
            val_correct += (vlogits[:, :-1].argmax(-1) == sl).sum().item()

    final_val_loss = val_loss_sum / max(val_tokens, 1)
    final_val_acc = val_correct / max(val_tokens, 1)
    final_ppl = math.exp(min(final_val_loss, 20))

    history["steps"].append(step)
    history["val_losses"].append(final_val_loss)
    history["train_losses"].append(loss.item())

    result = {
        "exp_id": exp_id,
        "val_loss": final_val_loss,
        "val_accuracy": final_val_acc,
        "val_perplexity": final_ppl,
        "total_params": sum(p.numel() for p in model.parameters()),
        "training_time_seconds": elapsed,
        "total_steps": step,
        "tokens_seen": tokens_seen,
        "history": history,
    }

    # Save
    out_path = Path(output_dir) / exp_id
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  === {exp_id} DONE ===")
    print(f"  val_loss: {final_val_loss:.4f} | val_acc: {final_val_acc:.4f} | ppl: {final_ppl:.1f}")
    print(f"  params: {result['total_params']:,} | time: {elapsed:.0f}s")

    return result


def update_leaderboard(results):
    """Update the frontier leaderboard."""
    # Sort by val_loss
    results.sort(key=lambda r: r["val_loss"])

    # Find transformer baseline
    tf_result = next((r for r in results if "transformer" in r["exp_id"]), None)
    tf_loss = tf_result["val_loss"] if tf_result else results[0]["val_loss"]

    lines = [
        "# Frontier Architecture Leaderboard\n",
        f"**Active baseline**: Transformer — val_loss = {tf_loss:.4f}\n",
        "This leaderboard compares architectures across families. The goal: find something that beats the transformer.\n",
        "## Current Rankings\n",
        "| Rank | exp_id | Family | val_loss | Δ vs transformer | Params | Speed (ms/step) | Notes |",
        "|------|--------|--------|----------|------------------|--------|-----------------|-------|",
    ]

    for i, r in enumerate(results, 1):
        delta = r["val_loss"] - tf_loss
        delta_str = "baseline" if "transformer" in r["exp_id"] else f"{delta:+.4f}"
        params = f"{r['total_params']/1e6:.1f}M"
        speed = f"{r.get('speed_ms', 0):.0f}"
        family = r.get("arch_family", "?")
        lines.append(f"| {i} | {r['exp_id']} | {family} | {r['val_loss']:.4f} | {delta_str} | {params} | {speed} | |")

    lines.append("\n## History\n")
    lines.append(f"Last updated: {time.strftime('%Y-%m-%d %H:%M')}\n")

    with open(LEADERBOARD_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"\nLeaderboard updated at {LEADERBOARD_PATH}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return

    set_seed(42)

    # ─── Load shared dataset ───
    print("=" * 70)
    print("  FRONTIER NOVEL ARCHITECTURE BATCH")
    print("=" * 70)

    data_cfg = DataConfig(
        dataset_path="HuggingFaceTB/smollm-corpus",
        dataset_name="cosmopedia-v2",
        seq_length=2048,
        num_samples=12000,
        cache_dir=str(ROOT / "hf_cache"),
        streaming=True,
    )
    tokenizer = setup_tokenizer(data_cfg)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer, cache_dir=str(ROOT / "processed_data"))
    print(f"Train: {len(train_ds)} sequences, Val: {len(val_ds)} sequences")

    g = torch.Generator().manual_seed(42)
    loader_args = dict(batch_size=8, num_workers=2, pin_memory=True,
                       persistent_workers=True, worker_init_fn=worker_init_fn, generator=g)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    # ─── Phase 1: Speed benchmark ───
    print("\n" + "=" * 70)
    print("  PHASE 1: SPEED BENCHMARK")
    print("=" * 70)

    speed_results = {}
    for exp in EXPERIMENTS:
        exp_id = exp["exp_id"]
        cfg = make_config(exp, vocab_size)
        try:
            set_seed(42)
            model = build_model(exp["arch_class"], cfg).to(device, dtype=torch.bfloat16)
            params = sum(p.numel() for p in model.parameters())
            ms = speed_benchmark(model, device, vocab_size=vocab_size)
            speed_results[exp_id] = {"ms": ms, "params": params}
            print(f"  {exp_id:30s} | {ms:7.1f} ms/step | {params/1e6:.1f}M params")
            del model
        except Exception as e:
            print(f"  {exp_id:30s} | FAILED: {e}")
            speed_results[exp_id] = {"ms": float('inf'), "params": 0}
        gc.collect()
        torch.cuda.empty_cache()

    # ─── Speed gate: discard >2x slower than transformer ───
    tf_speed = speed_results.get("transformer_baseline_12M", {}).get("ms",
               speed_results.get("transformer_baseline_v2", {}).get("ms", 355.0))
    speed_limit = tf_speed * 2.0
    print(f"\n  Transformer speed: {tf_speed:.1f} ms/step")
    print(f"  Speed limit (2x): {speed_limit:.1f} ms/step")

    survivors = []
    for exp in EXPERIMENTS:
        ms = speed_results[exp["exp_id"]]["ms"]
        if ms <= speed_limit:
            print(f"  PASS: {exp['exp_id']} ({ms:.1f} ms)")
            survivors.append(exp)
        else:
            print(f"  FAIL: {exp['exp_id']} ({ms:.1f} ms) — TOO SLOW, discarded")

    if not survivors:
        print("ERROR: All architectures too slow! Exiting.")
        return

    # ─── Phase 2: Train survivors ───
    print("\n" + "=" * 70)
    print(f"  PHASE 2: TRAINING {len(survivors)} ARCHITECTURES (6M tokens each)")
    print("=" * 70)

    all_results = []
    for exp in survivors:
        exp_id = exp["exp_id"]
        cfg = make_config(exp, vocab_size)
        cfg.compile_model = True  # enable compile for actual training

        set_seed(42)
        model = build_model(exp["arch_class"], cfg).to(device, dtype=torch.bfloat16)

        try:
            model = torch.compile(model)
            print(f"\n  {exp_id}: torch.compile SUCCESS")
        except Exception as e:
            print(f"\n  {exp_id}: torch.compile FAILED ({e}), using eager")

        try:
            result = train_one(model, cfg, train_loader, val_loader, exp_id,
                               str(OUTPUT_DIR / f"{cfg.train_tokens}tok"))
            result["arch_family"] = exp["arch_family"]
            result["speed_ms"] = speed_results[exp_id]["ms"]
            result["hypothesis"] = exp["hypothesis"]
            all_results.append(result)
        except Exception as e:
            import traceback
            print(f"\n  CRASH: {exp_id}: {e}")
            traceback.print_exc()
            crash_result = {
                "exp_id": exp_id,
                "val_loss": 999.0,
                "val_accuracy": 0.0,
                "val_perplexity": 999.0,
                "total_params": speed_results[exp_id]["params"],
                "training_time_seconds": 0,
                "arch_family": exp["arch_family"],
                "speed_ms": speed_results[exp_id]["ms"],
                "error": str(e),
            }
            all_results.append(crash_result)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ─── Phase 3: Results ───
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    all_results.sort(key=lambda r: r["val_loss"])
    tf_result = next((r for r in all_results if "transformer" in r["exp_id"]), None)
    tf_loss = tf_result["val_loss"] if tf_result else all_results[0]["val_loss"]

    for r in all_results:
        delta = r["val_loss"] - tf_loss
        marker = "***BEATS TRANSFORMER***" if delta < -0.002 and "transformer" not in r["exp_id"] else ""
        print(f"  {r['exp_id']:30s} | val_loss {r['val_loss']:.4f} | Δ {delta:+.4f} | {r['speed_ms']:.0f}ms | {marker}")

    # ─── Update leaderboard ───
    update_leaderboard(all_results)

    # ─── Save full results ───
    summary_path = OUTPUT_DIR / "batch_summary.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
