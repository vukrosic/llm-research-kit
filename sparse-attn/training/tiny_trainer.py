from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.minimax_sparse_config import LLMConfig
from models.llm import MinimalLLM


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 8) -> dict:
    model.eval()
    losses = []
    correct = 0
    total = 0
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    val_loss = sum(losses) / max(1, len(losses))
    return {
        "val_loss": val_loss,
        "val_ppl": math.exp(min(val_loss, 20.0)),
        "val_acc": correct / max(1, total),
    }


def train_tiny_lm(
    config: LLMConfig,
    train_dataset,
    val_dataset,
    output_dir: str | Path,
    token_budget: int,
    batch_size: int,
    learning_rate: float,
    device_name: str = "auto",
    eval_every: int = 50,
    seed: int = 42,
    num_workers: int = 0,
) -> dict:
    torch.manual_seed(seed)
    device = resolve_device(device_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MinimalLLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=train_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    param_count = count_parameters(model)
    manifest = {
        "attention_impl": config.attention_impl,
        "parameters": param_count,
        "config": config.__dict__ | {"minimax_sparse": config.minimax_sparse.__dict__},
        "token_budget": token_budget,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(device),
        "seed": seed,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    history = []
    tokens_seen = 0
    step = 0
    start = time.time()
    pbar = tqdm(total=token_budget, desc=config.attention_impl, unit="tok")
    while tokens_seen < token_budget:
        for x, y in train_loader:
            if tokens_seen >= token_budget:
                break
            model.train()
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_tokens = x.numel()
            tokens_seen += batch_tokens
            step += 1
            pbar.update(min(batch_tokens, max(0, token_budget - (tokens_seen - batch_tokens))))

            if step == 1 or step % eval_every == 0 or tokens_seen >= token_budget:
                metrics = evaluate(model, val_loader, device)
                elapsed_seconds = time.time() - start
                record = {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "train_loss": loss.item(),
                    "elapsed_seconds": elapsed_seconds,
                    "tokens_per_second": tokens_seen / max(elapsed_seconds, 1e-9),
                    **metrics,
                }
                history.append(record)
                with (output_dir / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                pbar.set_postfix(loss=f"{loss.item():.3f}", val=f"{metrics['val_loss']:.3f}")
    pbar.close()
    torch.save(model.state_dict(), output_dir / "model.pt")
    final = history[-1] if history else {}
    (output_dir / "summary.json").write_text(json.dumps(final, indent=2) + "\n")
    return {"manifest": manifest, "final": final, "output_dir": str(output_dir)}
