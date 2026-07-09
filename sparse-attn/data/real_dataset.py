from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import AutoTokenizer


@dataclass
class PreparedCorpus:
    train: Dataset
    val: Dataset
    vocab_size: int
    tokenizer_name: str
    tokens_available: int


class TokenBlockDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        if tokens.ndim != 1:
            raise ValueError("tokens must be a flat tensor")
        usable = (tokens.numel() // (seq_len + 1)) * (seq_len + 1)
        if usable <= seq_len + 1:
            raise ValueError("not enough tokens for one sequence")
        self.tokens = tokens[:usable].view(-1, seq_len + 1).contiguous()

    def __len__(self) -> int:
        return self.tokens.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.tokens[index]
        return row[:-1].long(), row[1:].long()


def _tokens_from_dataset(path: Path, tokenizer_name: str, token_budget: int, seq_len: int) -> tuple[torch.Tensor, int]:
    ds = load_from_disk(str(path))
    if hasattr(ds, "keys") and "train" in ds:
        ds = ds["train"]

    needed = token_budget + max(4096, token_budget // 5) + 2 * (seq_len + 1)
    chunks: list[torch.Tensor] = []
    total = 0

    if "input_ids" in ds.column_names:
        for example in ds:
            ids = example["input_ids"]
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            chunk = torch.tensor(ids, dtype=torch.long).flatten()
            chunks.append(chunk)
            total += chunk.numel()
            if total >= needed:
                break
    elif "text" in ds.column_names:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        texts = []
        for example in ds:
            texts.append(example["text"])
            if len(texts) >= 256:
                ids = tokenizer("\n\n".join(texts), add_special_tokens=True)["input_ids"]
                chunk = torch.tensor(ids, dtype=torch.long)
                chunks.append(chunk)
                total += chunk.numel()
                texts = []
                if total >= needed:
                    break
        if texts and total < needed:
            ids = tokenizer("\n\n".join(texts), add_special_tokens=True)["input_ids"]
            chunk = torch.tensor(ids, dtype=torch.long)
            chunks.append(chunk)
            total += chunk.numel()
    else:
        raise ValueError(f"Dataset must contain input_ids or text. Columns: {ds.column_names}")

    if not chunks:
        raise ValueError(f"No tokens found in {path}")
    return torch.cat(chunks)[:needed], total


def prepare_real_corpus(
    dataset_path: str | Path,
    seq_len: int,
    token_budget: int,
    tokenizer_name: str = "HuggingFaceTB/SmolLM2-135M",
    val_fraction: float = 0.1,
) -> PreparedCorpus:
    path = Path(dataset_path)
    tokens, tokens_available = _tokens_from_dataset(path, tokenizer_name, token_budget, seq_len)
    split = int(tokens.numel() * (1.0 - val_fraction))
    split = (split // (seq_len + 1)) * (seq_len + 1)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]
    if val_tokens.numel() < 2 * (seq_len + 1):
        val_tokens = train_tokens[-min(train_tokens.numel(), 8 * (seq_len + 1)) :]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return PreparedCorpus(
        train=TokenBlockDataset(train_tokens, seq_len),
        val=TokenBlockDataset(val_tokens, seq_len),
        vocab_size=tokenizer.vocab_size,
        tokenizer_name=tokenizer_name,
        tokens_available=tokens_available,
    )

