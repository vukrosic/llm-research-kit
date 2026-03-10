"""
Ablation Study Runner
=====================

Runs baseline + ablation experiments sequentially with identical data/seeds.
Each experiment gets its own output directory with metrics and plots.

Usage:
    # Quick smoke test (1M tokens each)
    python run_ablations.py --tokens 1000000
    
    # Full ablation study (50M tokens)
    python run_ablations.py --tokens 50000000

    # Run specific ablations only
    python run_ablations.py --tokens 1000000 --experiments baseline no_qk_norm

    # Skip compilation for faster debugging
    python run_ablations.py --tokens 1000000 --no-compile
"""

import argparse
import os
import sys
import json
import time
import math
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import autocast

from configs.ablation_configs import ABLATION_CONFIGS, get_ablation_config
from configs.dataset_config import DataConfig
from models.llm_ablation import MinimalLLMAblation
from optimizers.muon import Muon
from training.trainer import train_model, setup_muon_optimizer, warmup_compiled_kernels, EarlyStopping
from training.evaluation import evaluate_model
from utils.helpers import set_seed, format_time
from utils.logger import setup_logging
from data.loader import setup_tokenizer

# Reuse data loading from train_llm
from train_llm import prepare_datasets, worker_init_fn, print_system_info


def setup_muon_optimizer_ablation(model, config):
    """
    Setup Muon optimizer with ablation-aware ns_steps.
    Identical to the standard setup but reads muon_ns_steps from config.
    """
    muon_params = []
    adamw_params = []

    ns_steps = getattr(config, 'muon_ns_steps', 5)

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")
    print(f"  Muon polar express steps: {ns_steps}")

    muon_optimizer = Muon(
        muon_params, 
        lr=config.muon_lr, 
        momentum=config.muon_momentum,
        ns_steps=ns_steps,
    )
    adamw_optimizer = torch.optim.AdamW(
        adamw_params,
        lr=config.adamw_lr,
        weight_decay=config.weight_decay,
        fused=torch.cuda.is_available()
    )

    return [muon_optimizer, adamw_optimizer]


def warmup_compiled_kernels_ablation(model, config, train_loader, device, num_steps=3):
    """Warm up compiled kernels for ablation model."""
    print(f"🔥 Warming up kernels ({num_steps} steps)...")
    model.train()
    
    temp_optimizers = setup_muon_optimizer_ablation(model, config)
    warmup_iter = iter(train_loader)
    
    for _ in range(num_steps):
        try:
            batch = next(warmup_iter)
        except StopIteration:
            warmup_iter = iter(train_loader)
            batch = next(warmup_iter)
        
        if isinstance(batch, dict):
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)
        else:
            x, y = batch[0].to(device), batch[-1].to(device)
        
        if config.use_amp:
            with autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, config.vocab_size),
                    y[:, 1:].reshape(-1)
                )
            loss.backward()
        else:
            logits = model(x)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, config.vocab_size),
                y[:, 1:].reshape(-1)
            )
            loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        for opt in temp_optimizers:
            opt.step()
            opt.zero_grad()
    
    torch.cuda.synchronize()
    del temp_optimizers
    torch.cuda.empty_cache()
    print("✅ Kernels compiled and cached")


def run_single_experiment(
    config,
    train_loader,
    val_loader,
    output_dir: str,
    use_compile: bool = True,
):
    """Run a single ablation experiment and return results."""
    
    experiment_name = getattr(config, 'experiment_name', 'unknown')
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {experiment_name}")
    print(f"  Tokens: {config.train_tokens:,}")
    print(f"  Flags: embed_scale={getattr(config, 'use_embed_scale', True)}, "
          f"qk_norm={getattr(config, 'use_qk_norm', True)}, "
          f"muon_ns_steps={getattr(config, 'muon_ns_steps', 5)}")
    print(f"{'='*70}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_start = time.time()
    
    # 1. Initialize model with fixed seed
    set_seed(42)
    model = MinimalLLMAblation(config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")
    
    # 2. Save initial state
    initial_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # 3. Compile (if requested)
    if use_compile and config.compile_model:
        print("🚀 Compiling model with torch.compile...")
        orig_model = model
        try:
            model = torch.compile(model)
            print("✅ Model compiled successfully")
            
            warmup_compiled_kernels_ablation(model, config, train_loader, device, num_steps=3)
            orig_model.load_state_dict(initial_model_state)
            print("🔄 Model weights reset to initial state")
        except Exception as e:
            print(f"⚠️ Compilation failed: {e}")
            model = orig_model
            model.load_state_dict(initial_model_state)
    
    del initial_model_state
    torch.cuda.empty_cache()
    
    # 4. Create optimizers (ablation-aware)
    optimizers = setup_muon_optimizer_ablation(model, config)
    
    # 5. Create schedulers
    tokens_per_opt = config.batch_size * config.max_seq_len * config.gradient_accumulation_steps
    total_steps = config.train_tokens // tokens_per_opt
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))
    schedule_type = getattr(config, 'schedule_type', 'cosine')
    
    schedulers = []
    for optimizer in optimizers:
        if schedule_type == 'cosine':
            def lr_lambda(current_step, warmup=warmup_steps, total=total_steps):
                if current_step < warmup:
                    return current_step / warmup
                progress = (current_step - warmup) / max(1, total - warmup)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        elif schedule_type == 'linear':
            def lr_lambda(current_step, warmup=warmup_steps, total=total_steps):
                if current_step < warmup:
                    return current_step / warmup
                progress = (current_step - warmup) / max(1, total - warmup)
                return max(0.1, 1.0 - progress)
        else:
            def lr_lambda(current_step, warmup=warmup_steps):
                return current_step / warmup if current_step < warmup else 1.0
        
        schedulers.append(torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda))
    
    # 6. Reset RNG
    set_seed(42)
    
    setup_time = time.time() - setup_start
    print(f"⚙️ Setup complete in {setup_time:.2f}s")
    print("-" * 70)
    
    # 7. Train
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    results = train_model(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizers=optimizers,
        schedulers=schedulers,
        early_stopper=None,
        output_dir=None,
        extra_config={'experiment_name': experiment_name},
        log_every=getattr(config, 'log_every', 100),
    )
    
    total_training_time = results['training_time']
    total_wall_time = setup_time + total_training_time
    final_eval = results['final_metrics']
    metrics_history = results['metrics_history']
    
    # 8. Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics_data = {
        'experiment_name': experiment_name,
        'ablation_flags': {
            # original flags
            'use_embed_scale':  getattr(config, 'use_embed_scale', True),
            'use_qk_norm':      getattr(config, 'use_qk_norm', True),
            'muon_ns_steps':    getattr(config, 'muon_ns_steps', 5),
            # normalization
            'norm_type':        getattr(config, 'norm_type', 'rmsnorm'),
            'norm_position':    getattr(config, 'norm_position', 'pre'),
            'final_norm_type':  getattr(config, 'final_norm_type', 'rmsnorm'),
            # FFN
            'ffn_type':         getattr(config, 'ffn_type', 'standard'),
            'activation_type':  getattr(config, 'activation_type', 'squared_relu'),
            'd_ff':             getattr(config, 'd_ff', 2048),
            # attention
            'n_kv_heads':       getattr(config, 'n_kv_heads', 4),
            'use_rope':         getattr(config, 'use_rope', True),
            'rope_base':        getattr(config, 'rope_base', 10000.0),
            'use_bias':         getattr(config, 'use_bias', False),
            # block structure
            'parallel_block':   getattr(config, 'parallel_block', False),
            'residual_scale':   getattr(config, 'residual_scale', 1.0),
            # depth/width
            'n_layers':         getattr(config, 'n_layers', 22),
            'd_model':          getattr(config, 'd_model', 512),
            'n_heads':          getattr(config, 'n_heads', 8),
            # embeddings
            'use_learned_pos':  getattr(config, 'use_learned_pos', False),
            'tie_weights':      getattr(config, 'tie_weights', True),
            # init
            'init_scheme':      getattr(config, 'init_scheme', 'default'),
            # optimizer/regularization
            'muon_lr':          getattr(config, 'muon_lr', 0.024),
            'adamw_lr':         getattr(config, 'adamw_lr', 0.006),
            'muon_momentum':    getattr(config, 'muon_momentum', 0.95),
            'weight_decay':     getattr(config, 'weight_decay', 0.2),
            'grad_clip':        getattr(config, 'grad_clip', 1.0),
            'dropout':          getattr(config, 'dropout', 0.0),
            'schedule_type':    getattr(config, 'schedule_type', 'constant'),
            'warmup_ratio':     getattr(config, 'warmup_ratio', 0.0),
        },
        'final_metrics': final_eval,
        'setup_time_seconds': setup_time,
        'active_training_time_seconds': total_training_time,
        'total_wall_time_seconds': total_wall_time,
        'total_time_minutes': total_wall_time / 60,
        'actual_steps': results['steps'],
        'tokens_seen': results['tokens_seen'],
        'train_tokens': config.train_tokens,
        'history': metrics_history,
    }
    
    metrics_file = output_path / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"   📊 Metrics saved to {metrics_file}")
    
    # Save model checkpoint
    checkpoint_path = output_path / "model.pt"
    torch.save({
        'model_state_dict': results['model'].state_dict(),
        'config': config,
        'metrics': final_eval,
        'step': results['steps'],
    }, checkpoint_path)
    print(f"   💾 Model saved to {checkpoint_path}")
    
    # Print results
    print(f"\n{'─'*70}")
    print(f"  [{experiment_name}] RESULTS")
    print(f"{'─'*70}")
    print(f"  Setup Time:     {format_time(setup_time)}")
    print(f"  Training Time:  {format_time(total_training_time)}")
    print(f"  Val Loss:       {final_eval['val_loss']:.4f}")
    print(f"  Val Accuracy:   {final_eval['val_accuracy']:.4f}")
    print(f"  Val Perplexity: {final_eval['val_perplexity']:.2f}")
    print(f"{'─'*70}\n")
    
    return metrics_data


def generate_comparison_report(all_results, output_dir):
    """Generate a comparison summary of all experiments."""
    report_path = Path(output_dir) / "ablation_comparison.json"
    
    summary = []
    baseline_loss = None
    
    for result in all_results:
        name = result['experiment_name']
        val_loss = result['final_metrics']['val_loss']
        val_acc = result['final_metrics']['val_accuracy']
        val_ppl = result['final_metrics']['val_perplexity']
        train_time = result['active_training_time_seconds']
        
        if name == 'baseline':
            baseline_loss = val_loss
        
        summary.append({
            'experiment': name,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_perplexity': val_ppl,
            'training_time_min': train_time / 60,
        })
    
    # Add deltas if baseline exists
    if baseline_loss is not None:
        for entry in summary:
            entry['loss_delta_vs_baseline'] = entry['val_loss'] - baseline_loss
            entry['loss_delta_pct'] = (
                (entry['val_loss'] - baseline_loss) / baseline_loss * 100
                if baseline_loss > 0 else 0.0
            )
    
    report = {
        'summary': summary,
        'all_results': all_results,
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print table
    print("\n" + "=" * 80)
    print("  ABLATION STUDY COMPARISON")
    print("=" * 80)
    print(f"{'Experiment':<22} {'Val Loss':>10} {'Val Acc':>10} {'Val PPL':>10} {'Δ Loss':>10} {'Δ %':>8}")
    print("-" * 80)
    for entry in summary:
        delta = entry.get('loss_delta_vs_baseline', 0.0)
        delta_pct = entry.get('loss_delta_pct', 0.0)
        marker = "  (baseline)" if entry['experiment'] == 'baseline' else ""
        print(f"{entry['experiment']:<22} "
              f"{entry['val_loss']:>10.4f} "
              f"{entry['val_accuracy']:>10.4f} "
              f"{entry['val_perplexity']:>10.2f} "
              f"{delta:>+10.4f} "
              f"{delta_pct:>+7.2f}%"
              f"{marker}")
    print("=" * 80)
    print(f"\n📊 Full report saved to {report_path}\n")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Run Ablation Studies")
    parser.add_argument("--tokens", type=int, default=10_000,
                        help="Training tokens per experiment (default: 10k for hyper-fast smoke test)")
    parser.add_argument("--experiments", nargs="+", 
                        default=list(ABLATION_CONFIGS.keys()),
                        help=f"Experiments to run. Available: {list(ABLATION_CONFIGS.keys())}")
    parser.add_argument("--output_dir", type=str, default="./ablation_results",
                        help="Base output directory")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to preprocessed dataset")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile for faster debugging")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    
    args = parser.parse_args()
    
    logger = setup_logging(log_dir="./logs")
    print_system_info()
    
    print(f"🔬 ABLATION STUDY")
    print(f"   Tokens per experiment: {args.tokens:,}")
    print(f"   Experiments: {args.experiments}")
    print(f"   Output: {args.output_dir}")
    print(f"   Compile: {not args.no_compile}")
    print()
    
    # Validate experiment names
    for exp in args.experiments:
        if exp not in ABLATION_CONFIGS:
            print(f"❌ Unknown experiment: {exp}")
            print(f"   Available: {list(ABLATION_CONFIGS.keys())}")
            sys.exit(1)
    
    # Setup shared data (once for all experiments)
    # Use the first config to setup data parameters
    first_config = get_ablation_config(args.experiments[0], args.tokens)
    if args.batch_size:
        first_config.batch_size = args.batch_size
    
    # Calculate required docs
    avg_tokens_per_doc = 1000
    safety_factor = 2.0
    calc_num_docs = max(100, int((args.tokens / avg_tokens_per_doc) * safety_factor))
    
    data_cfg = DataConfig(
        dataset_path=args.dataset_path if args.dataset_path else "auto",
        seq_length=first_config.max_seq_len,
        num_samples=calc_num_docs,
        cache_dir="./hf_cache",
    )
    
    tokenizer = setup_tokenizer(data_cfg)
    
    print("📂 Loading shared dataset...")
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)
    
    # Create dataloaders
    import random
    import numpy as np
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    # Run experiments
    all_results = []
    total_start = time.time()
    
    for i, exp_name in enumerate(args.experiments):
        print(f"\n{'#'*70}")
        print(f"  RUNNING {i+1}/{len(args.experiments)}: {exp_name}")
        print(f"{'#'*70}")
        
        # Get config
        config = get_ablation_config(exp_name, args.tokens)
        config.vocab_size = tokenizer.vocab_size
        if args.batch_size:
            config.batch_size = args.batch_size
        
        # Set eval milestones based on token budget
        tokens_per_step = config.batch_size * config.max_seq_len * config.gradient_accumulation_steps
        est_steps = args.tokens // tokens_per_step
        
        if args.tokens <= 2_000_000:
            # Very short run — just eval at start and end
            config.eval_milestones = (0, est_steps // 4, est_steps // 2, 3 * est_steps // 4)
            config.log_every = max(10, est_steps // 20)
            config.eval_every = None
        elif args.tokens <= 8_000_000:
            config.eval_milestones = (0, 50, 100, 150, 200, 300, 400)
            config.log_every = 50
            config.eval_every = None
        elif args.tokens <= 20_000_000:
            config.eval_milestones = (0, 100, 250, 500, 750, 1000)
            config.log_every = 100
            config.eval_every = None
        elif args.tokens <= 100_000_000:
            config.eval_milestones = (0, 500, 1000, 2000, 3000, 4000, 5000)
            config.log_every = 250
            config.eval_every = None
        else:
            config.eval_milestones = (0, 1000, 5000, 10000, 20000, 30000, 40000, 50000)
            config.log_every = 1000
            config.eval_every = None
        
        # Create fresh dataloaders for each experiment (same data, same shuffle seed)
        g_exp = torch.Generator()
        g_exp.manual_seed(args.seed)
        
        loader_args = dict(
            batch_size=config.batch_size,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
            generator=g_exp,
        )
        train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
        
        # Output directory
        exp_output = os.path.join(args.output_dir, f"{args.tokens}tok", exp_name)
        
        # Run experiment
        result = run_single_experiment(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=exp_output,
            use_compile=not args.no_compile,
        )
        all_results.append(result)
        
        # Cleanup between experiments
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Generate comparison report
    report = generate_comparison_report(
        all_results, 
        os.path.join(args.output_dir, f"{args.tokens}tok")
    )
    
    total_time = time.time() - total_start
    print(f"\n⏱️ Total ablation study time: {format_time(total_time)}")
    print(f"   Ran {len(args.experiments)} experiments × {args.tokens:,} tokens each")


if __name__ == "__main__":
    main()
