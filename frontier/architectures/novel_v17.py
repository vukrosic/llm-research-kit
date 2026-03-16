"""
Novel Architectures — Batch 17 (Everything Combined + Novel)
=============================================================
Current best: 14L x 640d + GQA (3.5419).

Strategy:
1. KitchenSinkLM: Combine ALL winning ideas — 14L x 640d, token-shift conv,
   GQA w=256, growing windows in last few layers
2. TokenShiftLargerLM: 14L x 640d with token-shift conv + GQA (merge best two)
3. ConvQKNormLM: Add QK-norm to attention layers (proven in transformer ablations)
4. WiderStillLM: 12L x 704d + GQA — push width with GQA
5. ConvGatedLinAttnLM: Replace windowed attention with gated linear attention
   in late layers — O(n) attention replacement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from frontier.architectures.base import (
    FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
)
from frontier.architectures.registry import register_arch


class SwiGLU(nn.Module):
    def __init__(self, d, d_ff, bias=True):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=bias)
        self.up = nn.Linear(d, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d, bias=bias)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)


class MultiHeadConvMixer(nn.Module):
    def __init__(self, d_model, n_heads=8, max_kernel=65):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        kernel_sizes = [min(2**(i+1) + 1, max_kernel) for i in range(n_heads)]
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.head_convs = nn.ModuleList([
            nn.Conv1d(self.d_head, self.d_head, ks, padding=ks - 1, groups=self.d_head)
            for ks in kernel_sizes
        ])
        self.gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head)
        head_outs = []
        for h in range(self.n_heads):
            vh = v[:, :, h]
            out = self.head_convs[h](vh.transpose(1, 2))[:, :, :L].transpose(1, 2)
            head_outs.append(F.silu(out))
        combined = torch.cat(head_outs, dim=-1)
        return self.out(combined * torch.sigmoid(self.gate(x)))


class TokenShiftConvMixer(nn.Module):
    def __init__(self, d_model, n_heads=8, max_kernel=65):
        super().__init__()
        self.mix_weight = nn.Parameter(torch.ones(d_model) * 0.5)
        self.conv = MultiHeadConvMixer(d_model, n_heads, max_kernel)

    def forward(self, x):
        w = torch.sigmoid(self.mix_weight)
        shifted = F.pad(x[:, :-1], (0, 0, 1, 0))
        mixed = w * x + (1 - w) * shifted
        return self.conv(mixed)


class WindowedCausalGQA(nn.Module):
    def __init__(self, d_model, n_heads=8, n_kv_heads=4, window=256):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads
        self.heads_per_kv = n_heads // n_kv_heads
        self.window = window
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.scale = self.d_head ** -0.5

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.d_head).transpose(1, 2)
        k = k.repeat_interleave(self.heads_per_kv, dim=1)
        v = v.repeat_interleave(self.heads_per_kv, dim=1)
        W = self.window
        output = torch.zeros_like(q)
        for i in range(0, L, W):
            end = min(i + W, L)
            start_k = max(0, i - W)
            qi = q[:, :, i:end]
            ki = k[:, :, start_k:end]
            vi = v[:, :, start_k:end]
            attn = torch.matmul(qi, ki.transpose(-1, -2)) * self.scale
            q_pos = torch.arange(i, end, device=x.device)
            k_pos = torch.arange(start_k, end, device=x.device)
            mask = q_pos.unsqueeze(-1) < k_pos.unsqueeze(0)
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = F.softmax(attn, dim=-1)
            output[:, :, i:end] = torch.matmul(attn, vi)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(output)


class WindowedCausalGQAWithQKNorm(nn.Module):
    """GQA with QK-normalization (proven to help in transformer ablations)."""
    def __init__(self, d_model, n_heads=8, n_kv_heads=4, window=256):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads
        self.heads_per_kv = n_heads // n_kv_heads
        self.window = window
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.d_head)
        self.k_norm = nn.RMSNorm(self.d_head)
        self.scale = self.d_head ** -0.5

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.d_head)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.d_head)
        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.repeat_interleave(self.heads_per_kv, dim=1)
        v = v.repeat_interleave(self.heads_per_kv, dim=1)
        W = self.window
        output = torch.zeros_like(q)
        for i in range(0, L, W):
            end = min(i + W, L)
            start_k = max(0, i - W)
            qi = q[:, :, i:end]
            ki = k[:, :, start_k:end]
            vi = v[:, :, start_k:end]
            attn = torch.matmul(qi, ki.transpose(-1, -2)) * self.scale
            q_pos = torch.arange(i, end, device=x.device)
            k_pos = torch.arange(start_k, end, device=x.device)
            mask = q_pos.unsqueeze(-1) < k_pos.unsqueeze(0)
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = F.softmax(attn, dim=-1)
            output[:, :, i:end] = torch.matmul(attn, vi)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(output)


class GatedLinearAttention(nn.Module):
    """
    O(n) attention: Q, K, V projected, then:
    y_t = (sum_{s<=t} gate_s * K_s^T * V_s) @ Q_t
    Computed via cumulative sum — no explicit attention matrix.
    Uses ELU+1 kernel for positivity.
    """
    def __init__(self, d_model, n_heads=8, d_head_state=32):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_state = d_head_state  # reduced state dim for memory

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head_state, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, n_heads, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head)  # (B, L, H, d)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_state)  # (B, L, H, s)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head)  # (B, L, H, d)

        # ELU+1 kernel for positivity
        q = F.elu(q, alpha=1.0) + 1.0
        k = F.elu(k, alpha=1.0) + 1.0

        # Gate
        gate = torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)  # (B, L, H, 1)

        # Compute KV outer product and cumsum (chunked for memory)
        # KV: (B, L, H, s, d) — too large! Use chunk processing instead.
        CHUNK = 128
        outputs = []
        # Running state: (B, H, s, d)
        state = torch.zeros(B, self.n_heads, self.d_state, self.d_head,
                           device=x.device, dtype=x.dtype)

        for i in range(0, L, CHUNK):
            end = min(i + CHUNK, L)
            q_c = q[:, i:end]  # (B, chunk, H, d)
            k_c = k[:, i:end]  # (B, chunk, H, s)
            v_c = v[:, i:end]  # (B, chunk, H, d)
            g_c = gate[:, i:end]  # (B, chunk, H, 1)

            chunk_len = end - i
            chunk_out = []
            for t in range(chunk_len):
                # Update state: state = gate * state + k_t^T @ v_t
                kv = torch.einsum('bhs,bhd->bhsd', k_c[:, t], v_c[:, t])
                state = g_c[:, t].unsqueeze(-1) * state + kv
                # Query state
                out_t = torch.einsum('bhsd,bhd->bhs', state, q_c[:, t])
                # Project back to d_head by using a simple sum over state dim
                # Actually we need (B, H, d) output. Fix: query should give (B, H, d)
                # out = state @ q -> but dimensions wrong. Let's fix:
                # state: (B, H, s, d), q: (B, H, d) -> (B, H, s) via einsum
                # Then we need to combine. Actually:
                # y = Q @ (K^T V cumsum) = sum_s q_s * state_{s,:}
                out_t = torch.einsum('bhsd,bhs->bhd', state, q_c[:, t])
                chunk_out.append(out_t)

            chunk_out = torch.stack(chunk_out, dim=1)  # (B, chunk, H, d)
            outputs.append(chunk_out)

        output = torch.cat(outputs, dim=1)  # (B, L, H, d)
        # Normalize by sum of gates (for stability)
        output = output.contiguous().view(B, L, D)
        return self.out(output)


class FlexBlock(nn.Module):
    def __init__(self, d, d_ff, mixer, rs=1.0):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = mixer
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, bias=True)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


# ═══════════════════════════════════════════════════════
# 1. KITCHEN SINK — All winning ideas combined
# ═══════════════════════════════════════════════════════

@register_arch("KitchenSinkLM", "novel", "14L x 640d: token-shift conv + GQA growing windows")
class KitchenSinkLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        n_kv = ac.get("n_kv_heads", 4)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers  # 14
        split = int(n * 0.5)  # 7 conv, 7 GQA

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                attn_idx = i - split
                window = 64 * (2 ** min(attn_idx, 3))
                window = min(window, 512)
                mixer = WindowedCausalGQA(d, nh, n_kv, window=window)
            else:
                mixer = TokenShiftConvMixer(d, nh)
            blocks.append(FlexBlock(d, config.d_ff, mixer, rs=rs))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_init_weights)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"KitchenSink: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 2. TOKEN SHIFT LARGER — Merge two best ideas
# ═══════════════════════════════════════════════════════

@register_arch("TokenShiftLargerLM", "novel", "14L x 640d token-shift conv + GQA w=256")
class TokenShiftLargerLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        n_kv = ac.get("n_kv_heads", 4)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        split = int(n * 0.5)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                mixer = WindowedCausalGQA(d, nh, n_kv, window=256)
            else:
                mixer = TokenShiftConvMixer(d, nh)
            blocks.append(FlexBlock(d, config.d_ff, mixer, rs=rs))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_init_weights)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"TokenShiftLarger: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 3. CONV + QK-NORM GQA
# ═══════════════════════════════════════════════════════

@register_arch("ConvQKNormGQALM", "novel", "14L x 640d conv + QK-normed GQA")
class ConvQKNormGQALM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        n_kv = ac.get("n_kv_heads", 4)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        split = int(n * 0.5)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                mixer = WindowedCausalGQAWithQKNorm(d, nh, n_kv, window=256)
            else:
                mixer = MultiHeadConvMixer(d, nh)
            blocks.append(FlexBlock(d, config.d_ff, mixer, rs=rs))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_init_weights)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"ConvQKNormGQA: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 4. WIDER STILL + GQA — 12L x 704d
# ═══════════════════════════════════════════════════════

@register_arch("WiderStillGQALM", "novel", "12L x 704d + GQA w=256")
class WiderStillGQALM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model  # 704
        nh = ac.get("n_heads", 8)
        n_kv = ac.get("n_kv_heads", 4)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        split = int(n * 0.5)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                mixer = WindowedCausalGQA(d, nh, n_kv, window=256)
            else:
                mixer = MultiHeadConvMixer(d, nh)
            blocks.append(FlexBlock(d, config.d_ff, mixer, rs=rs))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_init_weights)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"WiderStillGQA: {self.config.n_layers}L x {self.config.d_model}d"
    def sequence_mixing_complexity(self): return "O(n*w)"


# ═══════════════════════════════════════════════════════
# 5. CONV + GATED LINEAR ATTENTION — O(n) global mixing
# ═══════════════════════════════════════════════════════

@register_arch("ConvGatedLinAttnLM", "novel", "Conv early + gated linear attention late (O(n))")
class ConvGatedLinAttnLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        split = int(n * 0.5)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                mixer = GatedLinearAttention(d, nh, d_head_state=32)
            else:
                mixer = MultiHeadConvMixer(d, nh)
            blocks.append(FlexBlock(d, config.d_ff, mixer, rs=rs))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_init_weights)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"ConvGatedLinAttn: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n)"
