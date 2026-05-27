import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_tiny_real_compare import build_config
from models.llm import MinimalLLM
from models.minimax_sparse_attention import MiniMaxSparseAttention
from training.tiny_trainer import count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether dense and sparse paths really differ.")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=49152)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, default=Path("runs/attention_diagnostics.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dense = MinimalLLM(build_config("dense", args.vocab_size, args.seq_len)).eval()
    torch.manual_seed(args.seed)
    sparse = MinimalLLM(build_config("minimax_sparse", args.vocab_size, args.seq_len)).eval()
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))

    with torch.no_grad():
        dense_logits = dense(x)
        sparse_logits = sparse(x)

    first_sparse_attn = next(
        module for module in sparse.modules() if isinstance(module, MiniMaxSparseAttention)
    )
    hidden = sparse.token_embedding(x) * (sparse.config.d_model ** 0.5)
    normalized = sparse.transformer_blocks[0].norm1(hidden)
    with torch.no_grad():
        _, debug = first_sparse_attn(normalized, return_debug=True)

    selected_tokens = debug.selected_token_mask.float().sum(dim=-1)
    possible_dense_tokens = torch.arange(1, args.seq_len + 1).view(1, args.seq_len, 1).expand_as(selected_tokens)
    density_vs_dense_causal = (selected_tokens / possible_dense_tokens).mean().item()

    report = {
        "dense_params": count_parameters(dense),
        "sparse_params": count_parameters(sparse),
        "logit_mean_abs_diff": (dense_logits - sparse_logits).abs().mean().item(),
        "logit_max_abs_diff": (dense_logits - sparse_logits).abs().max().item(),
        "sparse_selected_block_shape": list(debug.selected_block_indices.shape),
        "sparse_mean_selected_tokens": selected_tokens.mean().item(),
        "sparse_density_vs_dense_causal": density_vs_dense_causal,
        "first_token_indices": debug.selected_block_indices[0, 0].tolist(),
        "first_token_mask": debug.selected_block_mask[0, 0].tolist(),
        "last_token_indices": debug.selected_block_indices[0, -1].tolist(),
        "last_token_mask": debug.selected_block_mask[0, -1].tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
