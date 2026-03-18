#!/usr/bin/env python3
"""Experiment runner for LLM optimization — eager mode, no compilation overhead.

Usage:
    python optimization/run_experiments.py --queue optimization/queue.json
"""
import sys, os, json, time, gc, math, torch, traceback
import numpy as np, random

sys.path.insert(0, '/root/llm-research-kit')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.llm_config import LLMConfig
from configs.dataset_config import DataConfig
from training.trainer import train_model, setup_muon_optimizer, get_warmup_steps, build_lr_lambda
from models.llm import MinimalLLM
from utils.helpers import set_seed
from data.loader import setup_tokenizer
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda')

# ==============================================================
# Data cache
# ==============================================================
_DATASETS = None

def get_datasets():
    global _DATASETS
    if _DATASETS is not None:
        return _DATASETS
    data_cfg = DataConfig(
        dataset_path="auto", seq_length=2048,
        num_samples=100000, cache_dir="./hf_cache",
    )
    tokenizer = setup_tokenizer(data_cfg)
    from train_llm import prepare_datasets
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)
    _DATASETS = (train_ds, val_ds, tokenizer.vocab_size)
    return _DATASETS


def make_loaders(train_ds, val_ds, batch_size, seed=42):
    def winit(wid):
        np.random.seed(seed + wid)
        random.seed(seed + wid)
    g = torch.Generator().manual_seed(seed)
    kw = dict(num_workers=2, pin_memory=True,
              persistent_workers=True, worker_init_fn=winit)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, **kw),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw),
    )


# ==============================================================
# Single experiment
# ==============================================================
def run_experiment(exp_id, changes=None, train_seconds=5, seed=42):
    """Run one training experiment with a hard time limit."""
    changes = changes or {}

    # Build config
    config = LLMConfig()
    for k, v in changes.items():
        if hasattr(config, k):
            setattr(config, k, v)
    # Set train_tokens very high — time limit will stop training
    config.train_tokens = 100_000_000  # 100M, never reached
    config.max_train_seconds = train_seconds
    config.compile_model = False
    config.__post_init__()

    # Eval settings — no milestone evals, only final eval
    config.eval_milestones = ()
    config.eval_steps = 10
    config.eval_every = None
    config.log_every = 999999  # suppress mid-training logs for speed

    # Data
    train_ds, val_ds, vocab_size = get_datasets()
    config.vocab_size = vocab_size
    train_loader, val_loader = make_loaders(train_ds, val_ds, config.batch_size, seed)

    # Build model (eager, no compile)
    set_seed(seed)
    model = MinimalLLM(config).to(DEVICE, dtype=torch.bfloat16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {exp_id}: {total_params:,} params | batch={config.batch_size} | "
          f"time_limit={train_seconds}s")

    # Optimizers
    optimizers = setup_muon_optimizer(model, config)

    # Schedulers — estimate total steps from time limit
    tps_est = 50000  # ~50K tokens/sec in eager mode
    est_tokens = tps_est * train_seconds
    total_steps = max(1, int(est_tokens // (
        config.batch_size * config.max_seq_len * config.gradient_accumulation_steps
    )))
    warmup_steps = get_warmup_steps(config, total_steps)
    stype = getattr(config, 'schedule_type', 'constant')

    schedulers = []
    for opt in optimizers:
        fn = build_lr_lambda(stype, warmup_steps, total_steps)
        schedulers.append(torch.optim.lr_scheduler.LambdaLR(opt, fn))

    # Train
    set_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start = time.time()
    try:
        results = train_model(
            model=model, config=config,
            train_loader=train_loader, val_loader=val_loader,
            optimizers=optimizers, schedulers=schedulers,
            log_every=config.log_every,
        )
        wall = time.time() - start
        ttime = results['training_time']
        return {
            'exp_id': exp_id, 'status': 'done', 'seed': seed,
            'train_loss': results['train_loss'],
            'val_loss': results['final_metrics']['val_loss'],
            'val_acc': results['final_metrics'].get('val_accuracy', 0),
            'val_ppl': results['final_metrics'].get('val_perplexity', 0),
            'training_time': ttime, 'wall_time': wall,
            'steps': results['steps'],
            'tokens_seen': results['tokens_seen'],
            'tokens_per_second': results['tokens_seen'] / max(0.1, ttime),
            'train_seconds': train_seconds,
            'total_params': total_params,
            'changes': changes,
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {'exp_id': exp_id, 'status': 'oom', 'changes': changes,
                'error': 'CUDA OOM', 'wall_time': time.time() - start}
    except Exception as e:
        traceback.print_exc()
        return {'exp_id': exp_id, 'status': 'failed', 'changes': changes,
                'error': str(e), 'wall_time': time.time() - start}
    finally:
        del model, optimizers, schedulers, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()


# ==============================================================
# Batch runner
# ==============================================================
def update_status(data, path="optimization/status.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def run_batch(queue_file, results_dir="results"):
    with open(queue_file) as f:
        queue = json.load(f)

    print("Loading datasets...")
    get_datasets()
    print("Datasets ready.\n")

    results = []
    pending = [e for e in queue if e['status'] == 'pending']
    total = len(pending)

    for i, exp in enumerate(queue):
        if exp['status'] != 'pending':
            continue

        print(f"\n{'='*70}")
        print(f"[{len(results)+1}/{total}] {exp['exp_id']}: {exp.get('hypothesis','')}")
        print(f"{'='*70}")

        exp['status'] = 'running'
        update_status({
            'current_exp': exp['exp_id'],
            'hypothesis': exp.get('hypothesis', ''),
            'batch': exp.get('batch', '?'),
            'progress': f"{len(results)+1}/{total}",
            'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        })

        result = run_experiment(
            exp_id=exp['exp_id'],
            changes=exp.get('changes', {}),
            train_seconds=exp.get('train_seconds', 5),
            seed=exp.get('seed', 42),
        )

        exp['status'] = result['status']
        if result['status'] == 'done':
            exp['val_loss'] = round(result['val_loss'], 6)
            exp['train_loss'] = round(result['train_loss'], 6)
            exp['training_time'] = round(result['training_time'], 2)
            exp['wall_time'] = round(result.get('wall_time', 0), 2)
            exp['steps'] = result['steps']
            exp['tokens_per_second'] = round(result.get('tokens_per_second', 0))
        elif 'error' in result:
            exp['error'] = result['error']

        with open(queue_file, 'w') as f:
            json.dump(queue, f, indent=2)

        batch_dir = os.path.join(results_dir, exp.get('batch', 'misc'))
        os.makedirs(batch_dir, exist_ok=True)
        with open(os.path.join(batch_dir, f"{exp['exp_id']}.json"), 'w') as f:
            json.dump(result, f, indent=2, default=str)

        results.append(result)

        if result['status'] == 'done':
            print(f"\n  >> val={result['val_loss']:.4f} | "
                  f"train={result['train_loss']:.4f} | "
                  f"{result['training_time']:.1f}s | "
                  f"{result['steps']} steps | "
                  f"{result.get('tokens_per_second',0):.0f} tps")
        else:
            print(f"\n  >> FAILED: {result.get('error', 'unknown')}")

    update_status({
        'current_exp': None, 'batch': 'idle',
        'progress': 'complete',
        'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
    })

    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE: {len(results)} experiments")
    print(f"{'='*70}")
    done = [r for r in results if r['status'] == 'done']
    if done:
        best = min(done, key=lambda r: r['val_loss'])
        print(f"Best: {best['exp_id']} val_loss={best['val_loss']:.4f}")
    for r in results:
        if r['status'] == 'done':
            print(f"  {r['exp_id']:30s} val={r['val_loss']:.4f}  "
                  f"train={r['train_loss']:.4f}  {r['training_time']:.1f}s")
        else:
            print(f"  {r['exp_id']:30s} {r['status']}: {r.get('error','')}")

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--queue", default="optimization/queue.json")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()
    os.chdir('/root/llm-research-kit')
    run_batch(args.queue, args.results_dir)
