from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TopKBlockSelection:
    indices: torch.Tensor
    mask: torch.Tensor
    block_scores: torch.Tensor


@dataclass
class SparseAttentionDebug:
    selected_block_indices: torch.Tensor
    selected_block_mask: torch.Tensor
    selected_token_positions: torch.Tensor
    selected_token_mask: torch.Tensor
    block_scores: torch.Tensor


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("rotary dimension must be even")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.cos_cached.size(0):
            raise ValueError(f"sequence length {seq_len} exceeds rotary cache")
        cos = torch.repeat_interleave(
            self.cos_cached[:seq_len].to(device=x.device, dtype=x.dtype),
            repeats=2,
            dim=-1,
        ).view(1, seq_len, 1, -1)
        sin = torch.repeat_interleave(
            self.sin_cached[:seq_len].to(device=x.device, dtype=x.dtype),
            repeats=2,
            dim=-1,
        ).view(1, seq_len, 1, -1)
        return (x * cos) + (_rotate_half(x) * sin)


def block_max_pool_topk(
    index_queries: torch.Tensor,
    index_keys: torch.Tensor,
    block_size: int,
    top_k: int,
    pooling: str = "max",
) -> TopKBlockSelection:
    """
    Causal block-max top-k selection from the MiniMax diagram.

    index_queries: [batch, seq, gqa_groups, index_dim]
    index_keys: [batch, seq, index_dim]
    """
    if index_queries.ndim != 4:
        raise ValueError("index_queries must have shape [batch, seq, groups, dim]")
    if index_keys.ndim != 3:
        raise ValueError("index_keys must have shape [batch, seq, dim]")
    if index_queries.size(0) != index_keys.size(0) or index_queries.size(1) != index_keys.size(1):
        raise ValueError("index query/key batch and sequence dimensions must match")
    if index_queries.size(-1) != index_keys.size(-1):
        raise ValueError("index query/key dimensions must match")

    batch_size, seq_len, groups, index_dim = index_queries.shape
    num_blocks = math.ceil(seq_len / block_size)
    device = index_queries.device
    dtype_min = torch.finfo(index_queries.dtype).min

    if top_k == 0:
        empty_indices = torch.empty(batch_size, seq_len, groups, 0, dtype=torch.long, device=device)
        empty_mask = torch.empty(batch_size, seq_len, groups, 0, dtype=torch.bool, device=device)
        block_scores = index_queries.new_full((batch_size, seq_len, groups, num_blocks), dtype_min)
        return TopKBlockSelection(empty_indices, empty_mask, block_scores)

    token_scores = torch.einsum("btgd,bsd->btgs", index_queries, index_keys)
    token_scores = token_scores / math.sqrt(index_dim)

    query_pos = torch.arange(seq_len, device=device).view(seq_len, 1)
    key_pos = torch.arange(seq_len, device=device).view(1, seq_len)
    causal_token_mask = key_pos <= query_pos
    token_scores = token_scores.masked_fill(
        ~causal_token_mask.view(1, seq_len, 1, seq_len),
        dtype_min,
    )

    valid_mask = causal_token_mask.view(1, seq_len, 1, seq_len)
    padded_len = num_blocks * block_size
    pad_len = padded_len - seq_len
    if pad_len:
        token_scores = F.pad(token_scores, (0, pad_len), value=dtype_min)
        valid_mask = F.pad(valid_mask, (0, pad_len), value=False)

    token_scores = token_scores.view(batch_size, seq_len, groups, num_blocks, block_size)
    valid_mask = valid_mask.view(1, seq_len, 1, num_blocks, block_size)
    if pooling == "max":
        block_scores = token_scores.amax(dim=-1)
    elif pooling == "mean":
        masked = token_scores.masked_fill(~valid_mask, 0.0)
        counts = valid_mask.sum(dim=-1).clamp(min=1)
        block_scores = masked.sum(dim=-1) / counts
        block_scores = block_scores.masked_fill(valid_mask.sum(dim=-1) == 0, dtype_min)
    elif pooling == "logsumexp":
        block_scores = torch.logsumexp(token_scores, dim=-1)
    else:
        raise ValueError("pooling must be one of 'max', 'mean', or 'logsumexp'")
    actual_top_k = min(top_k, num_blocks)
    top_scores, top_indices = torch.topk(block_scores, k=actual_top_k, dim=-1)
    top_mask = top_scores > (dtype_min / 2)
    return TopKBlockSelection(top_indices, top_mask, block_scores)


class DenseGQAAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        n_kv_heads: int | None = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        if n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.head_dim = d_model // n_heads
        self.groups_per_kv = n_heads // self.n_kv_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.rotary = Rotary(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        q = self.rotary(self.q_norm(q)).transpose(1, 2)
        k = self.rotary(self.k_norm(k))
        if self.n_kv_heads != self.n_heads:
            k = torch.repeat_interleave(k, self.groups_per_kv, dim=2)
            v = torch.repeat_interleave(v, self.groups_per_kv, dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).reshape(batch_size, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(y)


class MiniMaxSparseAttention(nn.Module):
    """
    MiniMax-style sparse GQA attention.

    The index branch emits one top-k block list per query token and KV group.
    Each normal Q head then attends over only the selected blocks belonging to
    its GQA group. Future tokens inside the selected current block are masked.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        block_size: int = 16,
        top_k: int = 8,
        index_dim: int | None = None,
        pooling: str = "max",
        router_source: str = "separate",
        dropout: float = 0.0,
        n_kv_heads: int | None = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        if n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.head_dim = d_model // n_heads
        self.heads_per_group = n_heads // self.n_kv_heads
        self.block_size = block_size
        self.top_k = top_k
        self.index_dim = index_dim or self.head_dim
        self.pooling = pooling
        self.router_source = router_source
        self.dropout = dropout

        if self.router_source not in {"separate", "group_mean_q"}:
            raise ValueError("router_source must be 'separate' or 'group_mean_q'")
        if self.pooling not in {"max", "mean", "logsumexp"}:
            raise ValueError("pooling must be one of 'max', 'mean', or 'logsumexp'")
        if self.router_source == "group_mean_q" and self.index_dim != self.head_dim:
            raise ValueError("group_mean_q router_source requires index_dim == head_dim")

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.index_q_proj = (
            None
            if self.router_source == "group_mean_q"
            else nn.Linear(d_model, self.n_kv_heads * self.index_dim, bias=False)
        )
        self.index_k_proj = nn.Linear(d_model, self.index_dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.index_q_norm = nn.RMSNorm(self.index_dim)
        self.index_k_norm = nn.RMSNorm(self.index_dim)
        self.rotary = Rotary(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        q = self.rotary(self.q_norm(q))
        k = self.rotary(self.k_norm(k))

        if self.router_source == "group_mean_q":
            index_q = q.view(
                batch_size,
                seq_len,
                self.n_kv_heads,
                self.heads_per_group,
                self.head_dim,
            ).mean(dim=3)
        else:
            index_q = self.index_q_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.index_dim)
        index_q = self.index_q_norm(index_q)
        index_k = self.index_k_norm(self.index_k_proj(x))
        selection = block_max_pool_topk(index_q, index_k, self.block_size, self.top_k, pooling=self.pooling)

        selected_k, selected_v, selected_positions, selected_mask = self._gather_selected_kv(
            k,
            v,
            selection.indices,
            selection.mask,
        )

        if selected_k.size(3) == 0:
            sparse_output = q.new_zeros(batch_size, seq_len, self.n_heads, self.head_dim)
        else:
            q_grouped = q.view(
                batch_size,
                seq_len,
                self.n_kv_heads,
                self.heads_per_group,
                self.head_dim,
            )
            scores = torch.einsum("btghd,btgld->btghl", q_grouped, selected_k)
            scores = scores / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~selected_mask.unsqueeze(3), torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            weights = F.dropout(weights, p=self.dropout, training=self.training)
            sparse_output = torch.einsum("btghl,btgld->btghd", weights, selected_v)
            sparse_output = sparse_output.reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        y = sparse_output.reshape(batch_size, seq_len, self.n_heads * self.head_dim)
        y = self.o_proj(y)
        if not return_debug:
            return y

        return y, SparseAttentionDebug(
            selected_block_indices=selection.indices,
            selected_block_mask=selection.mask,
            selected_token_positions=selected_positions,
            selected_token_mask=selected_mask,
            block_scores=selection.block_scores,
        )

    def _gather_selected_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indices: torch.Tensor,
        block_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, groups, head_dim = k.shape
        selected_blocks = block_indices.size(-1)
        device = k.device
        if selected_blocks == 0:
            empty_kv = k.new_empty(batch_size, seq_len, groups, 0, head_dim)
            empty_pos = torch.empty(batch_size, seq_len, groups, 0, dtype=torch.long, device=device)
            empty_mask = torch.empty(batch_size, seq_len, groups, 0, dtype=torch.bool, device=device)
            return empty_kv, empty_kv, empty_pos, empty_mask

        offsets = torch.arange(self.block_size, device=device)
        positions = block_indices.unsqueeze(-1) * self.block_size + offsets.view(1, 1, 1, 1, self.block_size)
        query_pos = torch.arange(seq_len, device=device).view(1, seq_len, 1, 1, 1)
        token_mask = (positions < seq_len) & (positions <= query_pos) & block_mask.unsqueeze(-1)
        safe_positions = positions.clamp(min=0, max=seq_len - 1)

        batch_index = torch.arange(batch_size, device=device).view(batch_size, 1, 1, 1, 1)
        group_index = torch.arange(groups, device=device).view(1, 1, groups, 1, 1)
        selected_k = k[batch_index, safe_positions, group_index]
        selected_v = v[batch_index, safe_positions, group_index]

        selected_k = selected_k.reshape(batch_size, seq_len, groups, selected_blocks * self.block_size, head_dim)
        selected_v = selected_v.reshape(batch_size, seq_len, groups, selected_blocks * self.block_size, head_dim)
        positions = positions.reshape(batch_size, seq_len, groups, selected_blocks * self.block_size)
        token_mask = token_mask.reshape(batch_size, seq_len, groups, selected_blocks * self.block_size)
        return selected_k, selected_v, positions, token_mask
