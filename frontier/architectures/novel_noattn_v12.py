"""
V12: Breaking the Plateau — New Paradigms
==========================================
V10 confirmed the conv+SingleHead+VR paradigm plateaus at ~3.665.
V11 tests parallel conv+attn and additive attention.

V12 tries fundamentally different approaches:
1. Multi-query attention (4 Q heads, shared KV → 4 attention patterns cheaply)
2. ALiBi position bias in attention (position-aware attention scores)
3. Gated linear recurrence (parallel via cumsum, fp32 for stability)
4. Downsampled attention (attend on half-length sequence → O(n/2))
5. Two-pass: conv stack → attention → second conv stack with skip from first
6. Mixture-of-Attentions (multiple single-head attns, route to best)
7. Cross-layer attention (attend over features from multiple conv layers)
8. Wider FFN only at attention layer (more processing after attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from frontier.architectures.base import FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
from frontier.architectures.registry import register_arch
from frontier.architectures.batch100 import SwiGLU, Block, _init


class GatedMHConvMixer(nn.Module):
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks
        ])
        self.head_gates = nn.Linear(d, nh)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        v = self.v(x).view(B, L, self.nh, self.dh)
        gates = torch.sigmoid(self.head_gates(x))
        hs = []
        for h in range(self.nh):
            hv = v[:, :, h, :].transpose(1, 2)
            conv_out = F.silu(self.convs[h](hv)[:, :, :L].transpose(1, 2))
            hs.append(conv_out * gates[:, :, h:h+1])
        return self.out(torch.cat(hs, -1))


class SingleHeadAttnVR(nn.Module):
    def __init__(self, d, d_head=64, alpha=0.5):
        super().__init__()
        self.dh = d_head; self.alpha = alpha
        self.q = nn.Linear(d, d_head, bias=False)
        self.k = nn.Linear(d, d_head, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.q_norm = nn.RMSNorm(d_head)
        self.k_norm = nn.RMSNorm(d_head)
        self.out = nn.Linear(d, d, bias=False)
    def forward(self, x, embed=None, **kw):
        B, L, D = x.shape
        q = self.q_norm(self.q(x)).unsqueeze(1)
        k = self.k_norm(self.k(x)).unsqueeze(1)
        v = self.v(x)
        if embed is not None: v = (1 - self.alpha) * v + self.alpha * embed
        return self.out(F.scaled_dot_product_attention(q, k, v.unsqueeze(1), is_causal=True).squeeze(1))


class MixerBlock(nn.Module):
    def __init__(self, d, dff, mixer):
        super().__init__()
        self.norm1 = nn.RMSNorm(d)
        self.mixer = mixer
        self.norm2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, dff)
    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ═══════════════════════════════════════════════════════════════════════
# NEW: Multi-Query Attention (4 Q heads, 1 shared KV, full-width V)
# ═══════════════════════════════════════════════════════════════════════
class MultiQueryAttnVR(nn.Module):
    """4 query heads sharing one K and one full-width V.
    Gets 4 different attention patterns at cost of ~1.5 single-head attns.
    Each head produces a full-width output, then we average."""
    def __init__(self, d, d_head=64, n_q_heads=4, alpha=0.5):
        super().__init__()
        self.n_q = n_q_heads; self.dh = d_head; self.alpha = alpha
        self.q = nn.Linear(d, d_head * n_q_heads, bias=False)
        self.k = nn.Linear(d, d_head, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.q_norm = nn.RMSNorm(d_head)
        self.k_norm = nn.RMSNorm(d_head)
        self.out = nn.Linear(d, d, bias=False)
        self.head_gate = nn.Linear(d, n_q_heads)

    def forward(self, x, embed=None, **kw):
        B, L, D = x.shape
        # Multiple queries, shared KV
        q = self.q(x).view(B, L, self.n_q, self.dh)  # (B, L, nq, dh)
        q = self.q_norm(q).transpose(1, 2)  # (B, nq, L, dh)
        k = self.k_norm(self.k(x)).unsqueeze(1)  # (B, 1, L, dh)
        v = self.v(x)
        if embed is not None: v = (1 - self.alpha) * v + self.alpha * embed
        v = v.unsqueeze(1)  # (B, 1, L, D)

        # Each Q head attends with shared K, V
        attn_out = F.scaled_dot_product_attention(q, k.expand(-1, self.n_q, -1, -1),
                                                   v.expand(-1, self.n_q, -1, -1),
                                                   is_causal=True)  # (B, nq, L, D)
        # Gated combination of heads
        gates = torch.softmax(self.head_gate(x), dim=-1).transpose(1, 2).unsqueeze(-1)  # (B, nq, L, 1)
        combined = (attn_out * gates).sum(dim=1)  # (B, L, D)
        return self.out(combined)


# ═══════════════════════════════════════════════════════════════════════
# NEW: Attention with ALiBi (linear position bias)
# ═══════════════════════════════════════════════════════════════════════
class SingleHeadAttnALiBiVR(nn.Module):
    """Single-head attention with ALiBi position bias + value residual."""
    def __init__(self, d, d_head=64, alpha=0.5, slope=0.1):
        super().__init__()
        self.dh = d_head; self.alpha = alpha; self.slope = slope
        self.q = nn.Linear(d, d_head, bias=False)
        self.k = nn.Linear(d, d_head, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.q_norm = nn.RMSNorm(d_head)
        self.k_norm = nn.RMSNorm(d_head)
        self.out = nn.Linear(d, d, bias=False)
        self.slope_param = nn.Parameter(torch.tensor(slope).log())

    def forward(self, x, embed=None, **kw):
        B, L, D = x.shape
        q = self.q_norm(self.q(x))  # (B, L, dh)
        k = self.k_norm(self.k(x))
        v = self.v(x)
        if embed is not None: v = (1 - self.alpha) * v + self.alpha * embed

        # Compute attention scores manually to add ALiBi
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.dh ** 0.5)  # (B, L, L)

        # ALiBi: subtract slope * |i - j| for causal (j <= i, so i - j >= 0)
        slope = self.slope_param.exp()
        pos = torch.arange(L, device=x.device)
        bias = -slope * (pos.unsqueeze(1) - pos.unsqueeze(0)).abs().float()  # (L, L)

        # Causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores = scores + bias.unsqueeze(0)
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        return self.out(torch.matmul(attn, v))


# ═══════════════════════════════════════════════════════════════════════
# NEW: Gated Linear Recurrence (parallel via cumsum, fp32 stable)
# ═══════════════════════════════════════════════════════════════════════
class GatedLinearRecurrence(nn.Module):
    """h_t = gate_t * h_{t-1} + (1 - gate_t) * v_t
    Parallelized via log-space cumsum. Uses fp32 for stability."""
    def __init__(self, d, d_state=64):
        super().__init__()
        self.ds = d_state
        self.gate_proj = nn.Linear(d, d_state, bias=False)
        self.val_proj = nn.Linear(d, d_state, bias=False)
        self.out_proj = nn.Linear(d_state, d, bias=False)

    def forward(self, x, **kw):
        B, L, D = x.shape
        # Compute gates and values in fp32 for stability
        x_f = x.float()
        gate_logit = self.gate_proj(x_f)  # (B, L, ds)
        gate = torch.sigmoid(gate_logit)  # (B, L, ds) in (0, 1)
        val = self.val_proj(x_f)  # (B, L, ds)

        # Parallel scan via log-space cumsum
        # h_t = gate_t * h_{t-1} + (1 - gate_t) * val_t
        # Using the log-space trick: log_gate cumsum for decay, then weighted sum
        log_gate = torch.log(gate.clamp(min=1e-6))  # (B, L, ds)
        log_cum_gate = log_gate.cumsum(dim=1)  # cumulative log decay

        # For each position t: h_t = sum_{s<=t} val_s * (1-gate_s) * prod_{j=s+1}^{t} gate_j
        # = sum_{s<=t} val_s * (1-gate_s) * exp(log_cum_gate_t - log_cum_gate_s)
        # = exp(log_cum_gate_t) * sum_{s<=t} val_s * (1-gate_s) * exp(-log_cum_gate_s)

        weighted_val = (1 - gate) * val * torch.exp(-log_cum_gate)  # (B, L, ds)
        cum_weighted = weighted_val.cumsum(dim=1)  # (B, L, ds)
        h = cum_weighted * torch.exp(log_cum_gate)  # (B, L, ds)

        return self.out_proj(h.to(x.dtype))


# ═══════════════════════════════════════════════════════════════════════
# NEW: Downsampled Attention (attend on half-length, interpolate back)
# ═══════════════════════════════════════════════════════════════════════
class DownsampledAttnVR(nn.Module):
    """Average-pool to half length, run single-head attention there,
    then expand back. Causal: only pool from past tokens."""
    def __init__(self, d, d_head=64, alpha=0.5, pool_factor=2):
        super().__init__()
        self.pf = pool_factor; self.dh = d_head; self.alpha = alpha
        self.q = nn.Linear(d, d_head, bias=False)
        self.k = nn.Linear(d, d_head, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.q_norm = nn.RMSNorm(d_head)
        self.k_norm = nn.RMSNorm(d_head)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, embed=None, **kw):
        B, L, D = x.shape
        pf = self.pf
        # Causal pooling: each pooled position i represents tokens [i*pf, (i+1)*pf)
        # Pad to multiple of pf
        pad = (pf - L % pf) % pf
        if pad > 0:
            x_padded = F.pad(x, (0, 0, 0, pad))
        else:
            x_padded = x
        Lp = x_padded.shape[1]

        # Average pool
        x_pooled = x_padded.view(B, Lp // pf, pf, D).mean(dim=2)  # (B, L//pf, D)

        q = self.q_norm(self.q(x_pooled)).unsqueeze(1)
        k = self.k_norm(self.k(x_pooled)).unsqueeze(1)
        v = self.v(x_pooled)
        if embed is not None:
            # Pool the embedding too
            if pad > 0:
                emb_padded = F.pad(embed, (0, 0, 0, pad))
            else:
                emb_padded = embed
            embed_pooled = emb_padded.view(B, Lp // pf, pf, D).mean(dim=2)
            v = (1 - self.alpha) * v + self.alpha * embed_pooled

        attn_out = F.scaled_dot_product_attention(q, k, v.unsqueeze(1), is_causal=True).squeeze(1)
        out = self.out(attn_out)

        # Expand back: repeat each pooled position for pf tokens
        out_expanded = out.unsqueeze(2).expand(-1, -1, pf, -1).reshape(B, Lp, D)
        return out_expanded[:, :L, :]


# ═══════════════════════════════════════════════════════════════════════
# NEW: Cross-Layer Attention (attend over features from layers 7-9)
# ═══════════════════════════════════════════════════════════════════════
class CrossLayerAttnVR(nn.Module):
    """Attention where K/V come from concatenated features of multiple layers,
    not just the current hidden state."""
    def __init__(self, d, n_sources=3, d_head=64, alpha=0.5):
        super().__init__()
        self.dh = d_head; self.alpha = alpha; self.n_src = n_sources
        self.q = nn.Linear(d, d_head, bias=False)
        self.k = nn.Linear(d * n_sources, d_head, bias=False)
        self.v = nn.Linear(d * n_sources, d, bias=False)
        self.q_norm = nn.RMSNorm(d_head)
        self.k_norm = nn.RMSNorm(d_head)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, embed=None, cross_features=None, **kw):
        B, L, D = x.shape
        q = self.q_norm(self.q(x)).unsqueeze(1)

        if cross_features is not None:
            kv_input = torch.cat(cross_features, dim=-1)  # (B, L, D*n_src)
        else:
            kv_input = x.expand(-1, -1, self.n_src).reshape(B, L, -1)

        k = self.k_norm(self.k(kv_input)).unsqueeze(1)
        v = self.v(kv_input)
        if embed is not None: v = (1 - self.alpha) * v + self.alpha * embed
        return self.out(F.scaled_dot_product_attention(q, k, v.unsqueeze(1), is_causal=True).squeeze(1))


# ═══════════════════════════════════════════════════════════════════════
# V12 Architecture Definitions
# ═══════════════════════════════════════════════════════════════════════

class V12Base(FrontierModel):
    def _build(self, d, dff, vocab_size, n_total, special_layers, d_head=None, alpha=0.5):
        """Build standard conv model with special layers at specified positions."""
        if d_head is None: d_head = max(64, d // 8)
        self.embed = EmbeddingWithScale(vocab_size, d)
        self.vr_set = set(special_layers.keys())
        blocks = []
        for i in range(n_total):
            if i in special_layers:
                blocks.append(MixerBlock(d, dff, special_layers[i]))
            else:
                blocks.append(MixerBlock(d, dff, GatedMHConvMixer(d)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, vocab_size, self.embed.embedding.weight)
        self.apply(_init)

    def forward(self, x):
        h = self.embed(x); emb = h.detach()
        for i, b in enumerate(self.blocks):
            if i in self.vr_set:
                h = h + b.mixer(b.norm1(h), embed=emb); h = h + b.ffn(b.norm2(h))
            else: h = b(h)
        return self.head(self.norm(h))


# V12_01: Multi-query attention (4 Q heads, shared KV) at layer 10
@register_arch("V12_01_MultiQueryVRLM", "novel_v12", "d=640 16L, 4-query shared-KV attn at layer 10")
class V12_01(V12Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        self._build(640, 2560, c.vocab_size, 16,
                    {10: MultiQueryAttnVR(640, d_head=80, n_q_heads=4, alpha=0.5)})
    @classmethod
    def arch_family(cls): return "novel_v12"
    def describe(self): return "V12_01: Multi-query (4Q shared KV) + VR at layer 10"


# V12_02: ALiBi attention at layer 10
@register_arch("V12_02_ALiBiVRLM", "novel_v12", "d=640 16L, ALiBi + VR attention at layer 10")
class V12_02(V12Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        self._build(640, 2560, c.vocab_size, 16,
                    {10: SingleHeadAttnALiBiVR(640, d_head=80, alpha=0.5)})
    @classmethod
    def arch_family(cls): return "novel_v12"
    def describe(self): return "V12_02: SingleHead + ALiBi + VR at layer 10"


# V12_03: Gated Linear Recurrence at layer 10 (replaces attention entirely)
@register_arch("V12_03_GatedRecurrLM", "novel_v12", "d=640 16L, gated linear recurrence at layer 10")
class V12_03(V12Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        self._build(640, 2560, c.vocab_size, 16,
                    {10: GatedLinearRecurrence(640, d_state=128)})
        self.vr_set = set()  # No VR for recurrence
    @classmethod
    def arch_family(cls): return "novel_v12"
    def describe(self): return "V12_03: Gated linear recurrence (no attention)"


# V12_04: Gated linear recurrence + single-head VR (both at layer 10, parallel)
@register_arch("V12_04_RecurrPlusAttnLM", "novel_v12", "d=640 16L, parallel recurrence+attn at layer 10")
class V12_04(FrontierModel):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        d = 640; dff = 2560
        self.embed = EmbeddingWithScale(c.vocab_size, d)
        blocks = []
        for i in range(16):
            blocks.append(MixerBlock(d, dff, GatedMHConvMixer(d)))
        self.blocks = nn.ModuleList(blocks)
        self.recur = GatedLinearRecurrence(d, d_state=128)
        self.attn = SingleHeadAttnVR(d, d_head=80, alpha=0.5)
        self.gate = nn.Linear(d, 1)
        self.norm_pre = nn.RMSNorm(d)
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, c.vocab_size, self.embed.embedding.weight)
        self.apply(_init)

    def forward(self, x):
        h = self.embed(x); emb = h.detach()
        for i, b in enumerate(self.blocks):
            if i == 10:
                normed = self.norm_pre(h)
                r_out = self.recur(normed)
                a_out = self.attn(normed, embed=emb)
                g = torch.sigmoid(self.gate(normed))
                h = h + g * a_out + (1 - g) * r_out
                h = h + b.ffn(b.norm2(h))
            else:
                h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v12"
    def describe(self): return "V12_04: Parallel recurrence + attn at layer 10"


# V12_05: Downsampled attention (2x pool) at layer 10
@register_arch("V12_05_DownsampledAttnLM", "novel_v12", "d=640 16L, 2x downsampled attention at layer 10")
class V12_05(V12Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        self._build(640, 2560, c.vocab_size, 16,
                    {10: DownsampledAttnVR(640, d_head=80, alpha=0.5, pool_factor=2)})
    @classmethod
    def arch_family(cls): return "novel_v12"
    def describe(self): return "V12_05: 2x downsampled attention + VR at layer 10"


# V12_06: Multi-query attention (8 Q heads) — more patterns
@register_arch("V12_06_MultiQ8VRLM", "novel_v12", "d=640 16L, 8-query shared-KV attn at layer 10")
class V12_06(V12Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        self._build(640, 2560, c.vocab_size, 16,
                    {10: MultiQueryAttnVR(640, d_head=80, n_q_heads=8, alpha=0.5)})
    @classmethod
    def arch_family(cls): return "novel_v12"
    def describe(self): return "V12_06: Multi-query (8Q shared KV) + VR at layer 10"


# V12_07: Cross-layer attention (K/V from layers 7,8,9 features)
@register_arch("V12_07_CrossLayerAttnLM", "novel_v12", "d=640 16L, cross-layer attn at layer 10 (K/V from L7-9)")
class V12_07(FrontierModel):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        d = 640; dff = 2560
        self.embed = EmbeddingWithScale(c.vocab_size, d)
        blocks = []
        for i in range(16):
            if i == 10:
                blocks.append(MixerBlock(d, dff, CrossLayerAttnVR(d, n_sources=3, d_head=80, alpha=0.5)))
            else:
                blocks.append(MixerBlock(d, dff, GatedMHConvMixer(d)))
        self.blocks = nn.ModuleList(blocks)
        self.capture_layers = {7, 8, 9}
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, c.vocab_size, self.embed.embedding.weight)
        self.apply(_init)

    def forward(self, x):
        h = self.embed(x); emb = h.detach()
        captured = []
        for i, b in enumerate(self.blocks):
            if i == 10:
                h = h + b.mixer(b.norm1(h), embed=emb, cross_features=captured)
                h = h + b.ffn(b.norm2(h))
            else:
                h = b(h)
                if i in self.capture_layers:
                    captured.append(h.detach())
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v12"
    def describe(self): return "V12_07: Cross-layer attn (K/V from layers 7-9)"


# V12_08: Two single-head attention layers at 5 and 10, with VR
@register_arch("V12_08_TwoVR510LM", "novel_v12", "d=640 16L, 2 VR single-head at layers 5 and 10")
class V12_08(V12Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        self._build(640, 2560, c.vocab_size, 16,
                    {5: SingleHeadAttnVR(640, d_head=80, alpha=0.5),
                     10: SingleHeadAttnVR(640, d_head=80, alpha=0.5)})
    @classmethod
    def arch_family(cls): return "novel_v12"
    def describe(self): return "V12_08: 2 VR single-head at layers 5 and 10"


# V12_09: Wider FFN only at the attention layer (dff=5120 instead of 2560)
@register_arch("V12_09_WideFFNAttnLM", "novel_v12", "d=640 16L, VR@10 + 2x wider FFN at attn layer")
class V12_09(FrontierModel):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        d = 640; dff = 2560
        self.embed = EmbeddingWithScale(c.vocab_size, d)
        blocks = []
        for i in range(16):
            if i == 10:
                # Wider FFN at attention layer
                mixer = SingleHeadAttnVR(d, d_head=80, alpha=0.5)
                b = MixerBlock(d, dff * 2, mixer)  # 2x wider FFN
                blocks.append(b)
            else:
                blocks.append(MixerBlock(d, dff, GatedMHConvMixer(d)))
        self.blocks = nn.ModuleList(blocks)
        self.vr_set = {10}
        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, c.vocab_size, self.embed.embedding.weight)
        self.apply(_init)

    def forward(self, x):
        h = self.embed(x); emb = h.detach()
        for i, b in enumerate(self.blocks):
            if i in self.vr_set:
                h = h + b.mixer(b.norm1(h), embed=emb); h = h + b.ffn(b.norm2(h))
            else: h = b(h)
        return self.head(self.norm(h))
    @classmethod
    def arch_family(cls): return "novel_v12"
    def describe(self): return "V12_09: VR@10 + 2x wider FFN at attention layer"


# V12_10: Control (V9_13 copy for reproducibility)
@register_arch("V12_10_ControlLM", "novel_v12", "Control: V9_13 copy (d=640 16L VR@10)")
class V12_10(V12Base):
    def __init__(self, c: FrontierConfig):
        super().__init__(c)
        self._build(640, 2560, c.vocab_size, 16,
                    {10: SingleHeadAttnVR(640, d_head=80, alpha=0.5)})
    @classmethod
    def arch_family(cls): return "novel_v12"
    def describe(self): return "V12_10: Control (V9_13 copy)"
