import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.minimax_sparse_attention import block_max_pool_topk


def run_probe(seq_len: int, block_size: int, top_k: int, groups: int, index_dim: int, trials: int) -> dict:
    hits = 0
    total = 0
    num_blocks = seq_len // block_size
    for seed in range(trials):
        torch.manual_seed(seed)
        queries = torch.randn(1, seq_len, groups, index_dim) * 0.01
        keys = torch.randn(1, seq_len, index_dim) * 0.01

        target_query = torch.randint(block_size, seq_len, (1,)).item()
        max_block = target_query // block_size
        target_block = torch.randint(0, max_block + 1, (1,)).item()
        direction = torch.randn(index_dim)
        direction = direction / direction.norm()
        queries[:, target_query, :, :] = direction
        keys[:, target_block * block_size : (target_block + 1) * block_size, :] = direction

        selection = block_max_pool_topk(queries, keys, block_size=block_size, top_k=top_k)
        selected = selection.indices[0, target_query]
        mask = selection.mask[0, target_query]
        hits += ((selected == target_block) & mask).any(dim=-1).sum().item()
        total += groups

    return {
        "seq_len": seq_len,
        "block_size": block_size,
        "top_k": top_k,
        "groups": groups,
        "index_dim": index_dim,
        "trials": trials,
        "recall": hits / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic top-k block retrieval probe.")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--index_dim", type=int, default=32)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("runs/block_retrieval_probe.json"))
    args = parser.parse_args()

    result = run_probe(args.seq_len, args.block_size, args.top_k, args.groups, args.index_dim, args.trials)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
