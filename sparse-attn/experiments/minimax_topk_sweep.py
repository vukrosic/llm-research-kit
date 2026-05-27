import argparse
import json
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.minimax_sparse_attention import DenseGQAAttention, MiniMaxSparseAttention


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def time_forward(module: torch.nn.Module, x: torch.Tensor, warmup: int, repeats: int) -> float:
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            module(x)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            module(x)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMax sparse attention top-k/block-size sweep.")
    parser.add_argument("--seq_lens", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--top_ks", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--block_sizes", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("runs/minimax_topk_sweep/manifest.jsonl"))
    args = parser.parse_args()

    device = resolve_device(args.device)
    max_seq_len = max(args.seq_lens)
    dense = DenseGQAAttention(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=max_seq_len,
        dropout=0.0,
    ).to(device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for seq_len in args.seq_lens:
            x = torch.randn(args.batch_size, seq_len, args.d_model, device=device)
            dense_seconds = time_forward(dense, x, args.warmup, args.repeats)
            dense_record = {
                "attention": "dense",
                "seq_len": seq_len,
                "seconds": dense_seconds,
                "device": str(device),
            }
            f.write(json.dumps(dense_record) + "\n")
            print(dense_record)

            for block_size in args.block_sizes:
                for top_k in args.top_ks:
                    sparse = MiniMaxSparseAttention(
                        d_model=args.d_model,
                        n_heads=args.n_heads,
                        n_kv_heads=args.n_kv_heads,
                        max_seq_len=max_seq_len,
                        block_size=block_size,
                        top_k=top_k,
                        dropout=0.0,
                    ).to(device)
                    sparse_seconds = time_forward(sparse, x, args.warmup, args.repeats)
                    density = min(top_k * block_size, seq_len) / seq_len
                    record = {
                        "attention": "minimax_sparse",
                        "seq_len": seq_len,
                        "block_size": block_size,
                        "top_k": top_k,
                        "theoretical_density": density,
                        "seconds": sparse_seconds,
                        "dense_seconds": dense_seconds,
                        "speedup_vs_dense": dense_seconds / sparse_seconds if sparse_seconds else None,
                        "device": str(device),
                    }
                    f.write(json.dumps(record) + "\n")
                    print(record)


if __name__ == "__main__":
    main()
