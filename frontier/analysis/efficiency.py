"""
Efficiency Analysis
====================
Measure and compare FLOPs, memory, and throughput across architectures.
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


def measure_throughput(
    model: nn.Module,
    batch_size: int = 8,
    seq_len: int = 2048,
    vocab_size: int = 49152,
    n_warmup: int = 3,
    n_measure: int = 10,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Measure model throughput in tokens/second.

    Returns:
        dict with keys: tokens_per_sec, ms_per_batch, peak_memory_mb
    """
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(n_warmup):
            _ = model(x)
    torch.cuda.synchronize()

    # Reset peak memory
    torch.cuda.reset_peak_memory_stats()

    # Measure
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(n_measure):
            _ = model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    total_tokens = batch_size * seq_len * n_measure
    tokens_per_sec = total_tokens / elapsed
    ms_per_batch = (elapsed / n_measure) * 1000
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        "tokens_per_sec": tokens_per_sec,
        "ms_per_batch": ms_per_batch,
        "peak_memory_mb": peak_memory_mb,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def measure_training_throughput(
    model: nn.Module,
    batch_size: int = 8,
    seq_len: int = 2048,
    vocab_size: int = 49152,
    n_warmup: int = 2,
    n_measure: int = 5,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Measure training throughput (forward + backward) in tokens/second.
    """
    import torch.nn.functional as F

    model = model.to(device, dtype=torch.bfloat16)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(n_warmup):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Reset
    torch.cuda.reset_peak_memory_stats()

    # Measure
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_measure):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    total_tokens = batch_size * seq_len * n_measure
    tokens_per_sec = total_tokens / elapsed
    ms_per_step = (elapsed / n_measure) * 1000
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        "train_tokens_per_sec": tokens_per_sec,
        "ms_per_step": ms_per_step,
        "train_peak_memory_mb": peak_memory_mb,
    }


def compare_efficiency(models: Dict[str, nn.Module], **kwargs) -> Dict[str, Dict]:
    """Compare inference efficiency across multiple models."""
    results = {}
    for name, model in models.items():
        print(f"Measuring {name}...")
        try:
            results[name] = measure_throughput(model, **kwargs)
            results[name]["params"] = sum(p.numel() for p in model.parameters())
        except Exception as e:
            results[name] = {"error": str(e)}
        # Cleanup
        torch.cuda.empty_cache()
    return results
