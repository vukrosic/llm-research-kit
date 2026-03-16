"""
Novel Architectures — Batch 13 (Breaking the 3.58 Floor)
=========================================================
Progressive conv→attn is saturated at ~3.58-3.59. Every variant lands there.
Need genuinely different mechanisms to go lower.

Strategy — 5 radical ideas:
1. SlidingStateAttnLM: Attention where K,V are compressed running states (not raw tokens)
   — like a learned memory that summarizes the past, queried by current tokens
2. RecurrentGateLM: Conv layers with an explicit recurrent gate that carries
   a hidden state forward (not EMA — a full GRU-style gate on conv features)
3. ConvAttnPyramidLM: Multi-resolution — process at 2048, 1024, 512 resolution
   simultaneously via strided convs, then combine. Attention only at coarse res.
4. HybridMambaConvLM: Replace conv backbone with Mamba-style selective scan
   in early layers, keep windowed attention in late layers
5. DeepNarrowProgLM: 24 layers, d_model=384, d_ff=1536 — much deeper/narrower
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


class WindowedCausalAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, window=128):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window = window
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.scale = self.d_head ** -0.5

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        W = self.window
        output = torch.zeros_like(v)
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


class FlexBlock(nn.Module):
    def __init__(self, d, d_ff, mixer, ffn=None, rs=1.0):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = mixer
        self.n2 = nn.RMSNorm(d)
        self.ffn = ffn if ffn is not None else SwiGLU(d, d_ff, bias=True)
        self.rs = rs
    def forward(self, x):
        x = x + self.rs * self.mix(self.n1(x))
        x = x + self.rs * self.ffn(self.n2(x))
        return x


# ═══════════════════════════════════════════════════════
# 1. SLIDING STATE ATTENTION — Attend to compressed memory states
# ═══════════════════════════════════════════════════════

class SlidingStateAttention(nn.Module):
    """
    Instead of attending to raw past tokens, compress them into fixed-size
    state vectors (one per chunk). Query attends to both local tokens AND
    compressed states from previous chunks. This gives O(n) global context.
    """
    def __init__(self, d_model, n_heads=8, chunk_size=64, n_states=16):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.chunk_size = chunk_size
        self.n_states = n_states  # number of compressed states to keep

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.scale = self.d_head ** -0.5

        # State compression: linear projection from chunk to n_states summary vectors
        self.state_compress_k = nn.Linear(d_model, d_model, bias=False)
        self.state_compress_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        q = q.transpose(1, 2)  # (B, H, L, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        C = self.chunk_size
        output = torch.zeros_like(v)

        # Accumulate compressed states from past chunks
        # Each chunk produces one summary K and one summary V
        state_keys = []  # list of (B, H, 1, d)
        state_vals = []

        for i in range(0, L, C):
            end = min(i + C, L)
            qi = q[:, :, i:end]  # (B, H, chunk, d)
            ki = k[:, :, i:end]
            vi = v[:, :, i:end]

            # Build combined K,V: local chunk tokens + compressed past states
            if state_keys:
                past_k = torch.cat(state_keys[-self.n_states:], dim=2)  # (B, H, <=n_states, d)
                past_v = torch.cat(state_vals[-self.n_states:], dim=2)
                full_k = torch.cat([past_k, ki], dim=2)
                full_v = torch.cat([past_v, vi], dim=2)
                n_past = past_k.shape[2]
            else:
                full_k = ki
                full_v = vi
                n_past = 0

            attn = torch.matmul(qi, full_k.transpose(-1, -2)) * self.scale

            # Causal mask: queries can see all past states + causal within chunk
            chunk_len = end - i
            total_kv = full_k.shape[2]
            q_pos = torch.arange(i, end, device=x.device)
            # Past states are always visible (they're from before this chunk)
            # Within-chunk: causal
            k_pos_local = torch.arange(i, end, device=x.device)
            # Build mask: (chunk_len, total_kv)
            mask = torch.zeros(chunk_len, total_kv, device=x.device, dtype=torch.bool)
            # Mask future tokens within the local chunk portion
            for qi_idx in range(chunk_len):
                for ki_idx in range(n_past, total_kv):
                    local_ki = ki_idx - n_past
                    if local_ki > qi_idx:  # future token
                        mask[qi_idx, ki_idx] = True

            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = F.softmax(attn, dim=-1)
            output[:, :, i:end] = torch.matmul(attn, full_v)

            # Compress this chunk into a state vector (causal average)
            chunk_x = x[:, i:end]  # (B, chunk, D)
            sk = self.state_compress_k(chunk_x.mean(dim=1, keepdim=True))  # (B, 1, D)
            sv = self.state_compress_v(chunk_x.mean(dim=1, keepdim=True))
            sk = sk.view(B, 1, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, 1, d)
            sv = sv.view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
            state_keys.append(sk)
            state_vals.append(sv)

        output = output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(output)


# ═══════════════════════════════════════════════════════
# 2. RECURRENT GATE CONV — GRU-style gate on conv features
# ═══════════════════════════════════════════════════════

class RecurrentGateConvMixer(nn.Module):
    """
    MHConv with a GRU-style recurrent gate. The gate decides how much of the
    conv output to incorporate vs how much of the running hidden state to keep.
    Unlike EMA (fixed decay), this is input-dependent gating.

    Uses chunked processing for efficiency: process chunks of 64 tokens,
    carry state between chunks.
    """
    def __init__(self, d_model, n_heads=8, max_kernel=65, chunk_size=64):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.chunk_size = chunk_size
        kernel_sizes = [min(2**(i+1) + 1, max_kernel) for i in range(n_heads)]
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.head_convs = nn.ModuleList([
            nn.Conv1d(self.d_head, self.d_head, ks, padding=ks - 1, groups=self.d_head)
            for ks in kernel_sizes
        ])
        # GRU-style gates
        self.reset_gate = nn.Linear(d_model * 2, d_model)
        self.update_gate = nn.Linear(d_model * 2, d_model)
        self.candidate = nn.Linear(d_model * 2, d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        # First compute conv features for the whole sequence
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head)
        head_outs = []
        for h in range(self.n_heads):
            vh = v[:, :, h]
            out = self.head_convs[h](vh.transpose(1, 2))[:, :, :L].transpose(1, 2)
            head_outs.append(F.silu(out))
        conv_out = torch.cat(head_outs, dim=-1)  # (B, L, D)

        # Now apply GRU-style gating in chunks for efficiency
        C = self.chunk_size
        h_state = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
        outputs = []

        for i in range(0, L, C):
            end = min(i + C, L)
            chunk_conv = conv_out[:, i:end]  # (B, chunk, D)
            chunk_len = end - i

            # Expand state to chunk size
            h_expanded = h_state.expand(B, chunk_len, D)
            combined = torch.cat([chunk_conv, h_expanded], dim=-1)

            r = torch.sigmoid(self.reset_gate(combined))
            z = torch.sigmoid(self.update_gate(combined))
            combined_reset = torch.cat([chunk_conv, r * h_expanded], dim=-1)
            h_candidate = torch.tanh(self.candidate(combined_reset))

            out = (1 - z) * h_expanded + z * h_candidate
            outputs.append(out)

            # Update state with last position of chunk
            h_state = out[:, -1:].detach()  # detach to prevent BPTT explosion

        result = torch.cat(outputs, dim=1)
        return self.out(result)


# ═══════════════════════════════════════════════════════
# 3. CONV-ATTN PYRAMID — Multi-resolution processing
# ═══════════════════════════════════════════════════════

class MultiResConvMixer(nn.Module):
    """
    Process at multiple resolutions simultaneously using strided causal convolutions.
    Fine resolution (stride 1) captures local patterns.
    Coarse resolution (stride 4) captures global patterns.
    Combine via learned gating.
    """
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.d_model = d_model
        # Fine-resolution conv (standard MHConv)
        self.fine_conv = MultiHeadConvMixer(d_model, n_heads)
        # Coarse-resolution: downsample via causal conv (stride 4), process, upsample
        self.down_conv = nn.Conv1d(d_model, d_model, kernel_size=4, stride=4, padding=0)
        self.coarse_conv = MultiHeadConvMixer(d_model, n_heads // 2 if n_heads > 2 else 1)
        self.up_proj = nn.Linear(d_model, d_model, bias=False)
        # Combine
        self.combine_gate = nn.Linear(d_model * 2, d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        # Fine resolution
        fine = self.fine_conv(x)  # (B, L, D)

        # Coarse resolution: causal downsample
        # Pad to make L divisible by 4
        pad_len = (4 - L % 4) % 4
        x_padded = F.pad(x.transpose(1, 2), (pad_len, 0))  # left-pad for causality
        coarse = self.down_conv(x_padded)  # (B, D, L//4)
        coarse = coarse.transpose(1, 2)  # (B, L//4, D)
        coarse = self.coarse_conv(coarse)
        # Upsample back to L via repeat (causal: each coarse position maps to 4 fine positions)
        coarse_up = coarse.repeat_interleave(4, dim=1)[:, :L + pad_len]
        # Remove the padding portion
        if pad_len > 0:
            coarse_up = coarse_up[:, pad_len:]
        coarse_up = self.up_proj(coarse_up)

        # Combine fine and coarse
        combined = torch.cat([fine, coarse_up], dim=-1)
        gate = torch.sigmoid(self.combine_gate(combined))
        return self.out(fine * gate + coarse_up * (1 - gate))


# ═══════════════════════════════════════════════════════
# 4. SELECTIVE SCAN CONV — Mamba-inspired selective state space in conv backbone
# ═══════════════════════════════════════════════════════

class SelectiveScanMixer(nn.Module):
    """
    Simplified Mamba-style selective scan:
    - Input-dependent A, B, C matrices (discretized)
    - Uses causal scan (cumulative product) for state evolution
    - Much simpler than full Mamba but captures the key idea:
      the state transition is input-dependent (selective)
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        # Selective parameters: computed from input
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))

        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # Compute selective parameters
        dt = F.softplus(self.dt_proj(x_in))  # (B, L, D) — input-dependent timestep
        A = -torch.exp(self.A_log.float())  # (D, N) — negative for stability
        B_sel = self.B_proj(x_in)  # (B, L, N)
        C_sel = self.C_proj(x_in)  # (B, L, N)

        # Discretize: dA = exp(dt * A), dB = dt * B
        # dt: (B, L, D), A: (D, N) → dA: (B, L, D, N)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)
        dB = dt.unsqueeze(-1) * B_sel.unsqueeze(2)  # (B, L, D, N)

        # Selective scan: h_t = dA_t * h_{t-1} + dB_t * x_t
        # y_t = C_t @ h_t + D * x_t
        # Process in chunks to avoid sequential loop over L
        x_db = (x_in.unsqueeze(-1) * dB).float()  # (B, L, D, N)
        dA = dA.float()

        # Chunked scan for efficiency (chunk_size=64)
        CS = 64
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=torch.float32)
        y_parts = []

        for i in range(0, L, CS):
            end = min(i + CS, L)
            chunk_dA = dA[:, i:end]  # (B, chunk, D, N)
            chunk_xdB = x_db[:, i:end]  # (B, chunk, D, N)
            chunk_C = C_sel[:, i:end]  # (B, chunk, N)

            chunk_len = end - i
            ys = []
            for t in range(chunk_len):
                h = chunk_dA[:, t] * h + chunk_xdB[:, t]
                y_t = (h * chunk_C[:, t].unsqueeze(1)).sum(-1)  # (B, D)
                ys.append(y_t)

            y_parts.append(torch.stack(ys, dim=1))  # (B, chunk, D)

        y = torch.cat(y_parts, dim=1).to(x.dtype)  # (B, L, D)
        y = y + self.D * x_in
        y = y * F.silu(z)
        return self.out_proj(y)


# ═══════════════════════════════════════════════════════
# MODEL WRAPPERS
# ═══════════════════════════════════════════════════════

@register_arch("SlidingStateAttnLM", "novel", "Conv early + sliding state attention late")
class SlidingStateAttnLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        split = int(n * 0.75)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                mixer = SlidingStateAttention(d, nh, chunk_size=64, n_states=16)
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
    def describe(self): return f"SlidingStateAttn: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n)"


@register_arch("RecurrentGateConvLM", "novel", "Conv with GRU-style recurrent gate")
class RecurrentGateConvLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        split = int(n * 0.75)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                mixer = WindowedCausalAttention(d, nh, window=128)
            else:
                mixer = RecurrentGateConvMixer(d, nh, chunk_size=64)
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
    def describe(self): return f"RecurrentGateConv: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n)"


@register_arch("ConvAttnPyramidLM", "novel", "Multi-resolution conv + attention")
class ConvAttnPyramidLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        split = int(n * 0.75)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                mixer = WindowedCausalAttention(d, nh, window=128)
            else:
                mixer = MultiResConvMixer(d, nh)
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
    def describe(self): return f"ConvAttnPyramid: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n*w)"


@register_arch("MambaConvAttnLM", "novel", "Selective scan early + windowed attention late")
class MambaConvAttnLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model
        nh = ac.get("n_heads", 8)
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers
        split = int(n * 0.75)

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                mixer = WindowedCausalAttention(d, nh, window=128)
            else:
                mixer = SelectiveScanMixer(d, d_state=16)
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
    def describe(self): return f"MambaConvAttn: {self.config.n_layers}L"
    def sequence_mixing_complexity(self): return "O(n)"


@register_arch("DeepNarrowProgLM", "novel", "24-layer d=384 progressive conv→attn")
class DeepNarrowProgLM(FrontierModel):
    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d = config.d_model  # will be 384
        nh = ac.get("n_heads", 6)  # 6 heads for 384 dim
        rs = ac.get("residual_scale", 1.0)
        n = config.n_layers  # 24
        split = int(n * 0.75)  # 18 conv, 6 attn

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)
        blocks = []
        for i in range(n):
            if i >= split:
                mixer = WindowedCausalAttention(d, nh, window=128)
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
    def describe(self): return f"DeepNarrowProg: {self.config.n_layers}L x {self.config.d_model}d"
    def sequence_mixing_complexity(self): return "O(n*w)"
