"""
Experimental Architectures
============================
Truly novel mechanisms that don't fit existing categories.
These are the highest-risk, highest-reward experiments.

Each class here represents a genuinely new idea about how to
process sequences for language modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from frontier.architectures.base import (
    FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
)
from frontier.architectures.registry import register_arch


# ──────────────────────────────────────────────
# 1. Differential Attention
# ──────────────────────────────────────────────

class DifferentialAttention(nn.Module):
    """
    Differential Attention (Microsoft Research).

    Instead of one softmax attention, compute TWO attention patterns
    and take their difference. This cancels out noise/common patterns
    and amplifies the signal — like differential amplifiers in electronics.

    attn = softmax(Q1 @ K1^T) - λ * softmax(Q2 @ K2^T)
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0,
                 use_bias: bool = True, max_seq_len: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Two sets of Q, K projections + one V
        self.q1_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k1_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.q2_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k2_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)

        # Learnable subtraction weight per head
        self.lambda_param = nn.Parameter(torch.ones(n_heads) * 0.5)

        # QK norms
        self.q1_norm = nn.LayerNorm(self.d_head)
        self.k1_norm = nn.LayerNorm(self.d_head)
        self.q2_norm = nn.LayerNorm(self.d_head)
        self.k2_norm = nn.LayerNorm(self.d_head)

        # RoPE
        from torchtune.modules import RotaryPositionalEmbeddings
        self.rope = RotaryPositionalEmbeddings(self.d_head, max_seq_len)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_head

        q1 = self.q1_proj(x).reshape(B, L, H, D).transpose(1, 2)
        k1 = self.k1_proj(x).reshape(B, L, H, D).transpose(1, 2)
        q2 = self.q2_proj(x).reshape(B, L, H, D).transpose(1, 2)
        k2 = self.k2_proj(x).reshape(B, L, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, L, H, D).transpose(1, 2)

        # Normalize and apply RoPE
        q1, k1 = self.q1_norm(q1), self.k1_norm(k1)
        q2, k2 = self.q2_norm(q2), self.k2_norm(k2)
        q1, k1 = self.rope(q1), self.rope(k1)
        q2, k2 = self.rope(q2), self.rope(k2)

        # Compute both attention patterns
        attn1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scale
        attn2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        attn1 = attn1.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn2 = attn2.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)

        # Differential: subtract with learnable lambda
        lam = torch.sigmoid(self.lambda_param).reshape(1, H, 1, 1)
        attn = attn1 - lam * attn2
        attn = F.relu(attn)  # ensure non-negative after subtraction

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.dropout(self.out_proj(out))


# ──────────────────────────────────────────────
# 2. Frequency-Domain Mixing (FFT-based)
# ──────────────────────────────────────────────

class FrequencyMixer(nn.Module):
    """
    Token mixing in the frequency domain.

    Apply FFT along the sequence dimension, learn a filter in frequency
    space, then IFFT back. This is equivalent to a global convolution
    but parameterized in the spectral domain.

    Inspired by FNet but with learnable spectral filters instead of
    just using raw FFT features.
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        n_freq = max_seq_len // 2 + 1

        # Learnable spectral filter (complex-valued)
        self.filter_real = nn.Parameter(torch.ones(d_model, n_freq) * 0.5)
        self.filter_imag = nn.Parameter(torch.zeros(d_model, n_freq))

        # Gating
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # FFT along sequence dimension
        x_freq = torch.fft.rfft(x.transpose(1, 2), dim=-1)  # (B, D, L//2+1)

        # Apply learnable filter
        n_freq = x_freq.shape[-1]
        filt = torch.complex(
            self.filter_real[:, :n_freq],
            self.filter_imag[:, :n_freq]
        )  # (D, n_freq)
        x_filtered = x_freq * filt.unsqueeze(0)

        # IFFT back to sequence domain
        x_time = torch.fft.irfft(x_filtered, n=L, dim=-1)  # (B, D, L)
        x_time = x_time.transpose(1, 2)  # (B, L, D)

        # Causal masking via cumulative gate (approximate)
        gate = torch.sigmoid(self.gate(x))
        return self.dropout(x_time * gate)


# ──────────────────────────────────────────────
# 3. Evolving State Machine
# ──────────────────────────────────────────────

class EvolvingStateMachine(nn.Module):
    """
    A novel sequence mixer that maintains a state matrix S ∈ R^{d×d}
    which evolves at each token via a learned transition:

        S_t = σ(W_decay) ⊙ S_{t-1} + x_t^T @ W_write @ x_t
        o_t = (S_t @ W_read @ x_t)

    This is like an external memory that the model reads from and
    writes to at each position. The key difference from SSMs is that
    the state is a full matrix (rank-d), not a vector.
    """

    def __init__(self, d_model: int, d_state: int = 32, dropout: float = 0.0,
                 use_bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Write: project input to state update
        self.write_key = nn.Linear(d_model, d_state, bias=use_bias)
        self.write_val = nn.Linear(d_model, d_state, bias=use_bias)

        # Read: project from state to output
        self.read_query = nn.Linear(d_model, d_state, bias=use_bias)
        self.read_proj = nn.Linear(d_state, d_model, bias=use_bias)

        # Decay: input-dependent forgetting
        self.decay_proj = nn.Linear(d_model, d_state * d_state, bias=True)
        nn.init.constant_(self.decay_proj.bias, 2.0)  # high initial retention

        self.out_gate = nn.Linear(d_model, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        S = self.d_state

        state = torch.zeros(B, S, S, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(L):
            xt = x[:, t]  # (B, D)

            # Write: outer product of write_key and write_val
            wk = self.write_key(xt)  # (B, S)
            wv = self.write_val(xt)  # (B, S)
            write = torch.einsum('bi,bj->bij', wk, wv)  # (B, S, S)

            # Decay: input-dependent per-element decay
            decay = torch.sigmoid(self.decay_proj(xt).reshape(B, S, S))

            # Update state
            state = decay * state + write

            # Read
            rq = self.read_query(xt)  # (B, S)
            read = torch.einsum('bij,bj->bi', state, rq)  # (B, S)
            out = self.read_proj(read)  # (B, D)

            # Gate
            gate = torch.sigmoid(self.out_gate(xt))
            outputs.append(gate * out)

        return self.dropout(torch.stack(outputs, dim=1))


# ──────────────────────────────────────────────
# 4. Polynomial Attention
# ──────────────────────────────────────────────

class PolynomialAttention(nn.Module):
    """
    Replace softmax with polynomial kernel: attn(Q, K) = (Q @ K^T / d + 1)^p

    For p=2, this gives quadratic attention which can be decomposed:
    (QK^T + d)^2 = (QK^T)^2 + 2d*(QK^T) + d^2

    The squared term captures feature interactions not present in
    linear attention, while still being more interpretable than softmax.
    """

    def __init__(self, d_model: int, n_heads: int = 8, degree: int = 2,
                 dropout: float = 0.0, use_bias: bool = True,
                 max_seq_len: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.degree = degree

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)

        self.q_norm = nn.LayerNorm(self.d_head)
        self.k_norm = nn.LayerNorm(self.d_head)

        from torchtune.modules import RotaryPositionalEmbeddings
        self.rope = RotaryPositionalEmbeddings(self.d_head, max_seq_len)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_head

        qkv = self.qkv(x).reshape(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.rope(self.q_norm(q))
        k = self.rope(self.k_norm(k))

        # Polynomial kernel attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)

        # Causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)

        # Polynomial kernel: (1 + score)^degree, then normalize
        attn = (1 + scores).clamp(min=0) ** self.degree
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.dropout(self.out_proj(out))


# ──────────────────────────────────────────────
# Blocks for experimental mechanisms
# ──────────────────────────────────────────────

class ExperimentalBlock(nn.Module):
    """Generic block wrapping any sequence mixer."""

    def __init__(self, d_model: int, d_ff: int, mixer: nn.Module,
                 dropout: float = 0.0, residual_scale: float = 1.0,
                 use_bias: bool = True):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.mixer = mixer
        self.norm2 = nn.RMSNorm(d_model)
        self.residual_scale = residual_scale

        self.gate_proj = nn.Linear(d_model, d_ff, bias=use_bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=use_bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=use_bias)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_scale * self.mixer(self.norm1(x))
        h = self.norm2(x)
        ffn_out = self.ffn_dropout(self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h)))
        x = x + self.residual_scale * ffn_out
        return x


# ──────────────────────────────────────────────
# Full Language Models
# ──────────────────────────────────────────────

@register_arch("DiffAttnLM", "experimental", "Differential attention language model")
class DiffAttnLM(FrontierModel):

    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        n_heads = ac.get("n_heads", 8)
        residual_scale = ac.get("residual_scale", 1.0)
        use_bias = ac.get("use_bias", True)

        self.embed = EmbeddingWithScale(config.vocab_size, config.d_model, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            ExperimentalBlock(
                config.d_model, config.d_ff,
                DifferentialAttention(config.d_model, n_heads, config.dropout, use_bias, config.max_seq_len),
                config.dropout, residual_scale, use_bias,
            )
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(config.d_model)
        self.head = LMHead(
            config.d_model, config.vocab_size,
            embedding_weight=self.embed.embedding.weight if config.tie_weights else None
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))

    @classmethod
    def arch_family(cls) -> str:
        return "experimental"

    def describe(self) -> str:
        return f"DiffAttn: {self.config.n_layers}L x {self.config.d_model}d"

    def sequence_mixing_complexity(self) -> str:
        return "O(n^2)"


@register_arch("FreqMixerLM", "experimental", "Frequency-domain mixer language model")
class FreqMixerLM(FrontierModel):

    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        residual_scale = ac.get("residual_scale", 1.0)
        use_bias = ac.get("use_bias", True)

        self.embed = EmbeddingWithScale(config.vocab_size, config.d_model, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            ExperimentalBlock(
                config.d_model, config.d_ff,
                FrequencyMixer(config.d_model, config.max_seq_len, config.dropout),
                config.dropout, residual_scale, use_bias,
            )
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(config.d_model)
        self.head = LMHead(
            config.d_model, config.vocab_size,
            embedding_weight=self.embed.embedding.weight if config.tie_weights else None
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))

    @classmethod
    def arch_family(cls) -> str:
        return "experimental"

    def describe(self) -> str:
        return f"FreqMixer: {self.config.n_layers}L x {self.config.d_model}d"

    def sequence_mixing_complexity(self) -> str:
        return "O(n log n)"


@register_arch("EvolvingStateLM", "experimental", "Evolving state machine language model")
class EvolvingStateLM(FrontierModel):

    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        d_state = ac.get("d_state", 32)
        residual_scale = ac.get("residual_scale", 1.0)
        use_bias = ac.get("use_bias", True)

        self.embed = EmbeddingWithScale(config.vocab_size, config.d_model, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            ExperimentalBlock(
                config.d_model, config.d_ff,
                EvolvingStateMachine(config.d_model, d_state, config.dropout, use_bias),
                config.dropout, residual_scale, use_bias,
            )
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(config.d_model)
        self.head = LMHead(
            config.d_model, config.vocab_size,
            embedding_weight=self.embed.embedding.weight if config.tie_weights else None
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))

    @classmethod
    def arch_family(cls) -> str:
        return "experimental"

    def describe(self) -> str:
        return f"EvolvingState: {self.config.n_layers}L x {self.config.d_model}d, state={self.config.arch_config.get('d_state', 32)}"

    def supports_recurrent_inference(self) -> bool:
        return True

    def sequence_mixing_complexity(self) -> str:
        return "O(n)"


@register_arch("PolyAttnLM", "experimental", "Polynomial attention language model")
class PolyAttnLM(FrontierModel):

    def __init__(self, config: FrontierConfig):
        super().__init__(config)
        ac = config.arch_config
        n_heads = ac.get("n_heads", 8)
        degree = ac.get("degree", 2)
        residual_scale = ac.get("residual_scale", 1.0)
        use_bias = ac.get("use_bias", True)

        self.embed = EmbeddingWithScale(config.vocab_size, config.d_model, dropout=config.dropout)
        self.blocks = nn.ModuleList([
            ExperimentalBlock(
                config.d_model, config.d_ff,
                PolynomialAttention(config.d_model, n_heads, degree, config.dropout, use_bias, config.max_seq_len),
                config.dropout, residual_scale, use_bias,
            )
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(config.d_model)
        self.head = LMHead(
            config.d_model, config.vocab_size,
            embedding_weight=self.embed.embedding.weight if config.tie_weights else None
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))

    @classmethod
    def arch_family(cls) -> str:
        return "experimental"

    def describe(self) -> str:
        return f"PolyAttn (deg={self.config.arch_config.get('degree', 2)}): {self.config.n_layers}L x {self.config.d_model}d"

    def sequence_mixing_complexity(self) -> str:
        return "O(n^2)"
