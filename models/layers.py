import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from configs.llm_config import LLMConfig
from .components import SquaredReLUFeedForward


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(
            dim=dim, max_seq_len=max_seq_len, base=10000
        )

    def forward(self, x_BTHD: torch.Tensor):
        # x_BTHD shape: [B, T, H, D] - need to convert to [B, T, H, D] for torchtune
        # torchtune expects [batch, seq_len, num_heads, head_dim]
        # Our input is already [B, T, H, D] which matches torchtune's expectation
        return self.rope(x_BTHD)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        config: LLMConfig,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.max_seq_len = config.max_seq_len
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_heads
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.d_k = config.d_model // config.n_heads
        
        # ============ MERGED QKVO PROJECTION ============
        # Instead of 4 separate Linear layers, use single merged projection
        q_size = config.d_model
        kv_size = self.n_kv_heads * self.d_k
        o_size = config.d_model
        
        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_size = q_size + 2 * kv_size  # Q + K + V sizes
        
        # Single parameter tensor for all projections
        # Shape: [Q_size + K_size + V_size + O_size, d_model]
        self.qkvo_proj = nn.Parameter(
            torch.empty(q_size + 2 * kv_size + o_size, config.d_model)
        )
        
        # Initialize all weights with std=0.02
        with torch.no_grad():
            torch.nn.init.normal_(self.qkvo_proj, mean=0.0, std=0.02)
        # ================================================
        
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            if config.shared_qk_norm:
                self.qk_shared_norm = nn.RMSNorm(self.d_k)
            else:
                self.q_norm = nn.RMSNorm(self.d_k)
                self.k_norm = nn.RMSNorm(self.d_k)
            
            if config.use_qk_bias:
                self.q_bias = nn.Parameter(torch.zeros(self.n_heads, self.d_k))
                self.k_bias = nn.Parameter(torch.zeros(self.n_kv_heads, self.d_k))
        
        if config.use_per_head_scaling:
            self.head_scale = nn.Parameter(torch.ones(self.n_heads, 1, 1))

        self.qk_norm_after_rope = config.qk_norm_after_rope
        self.qk_norm_k_only = config.qk_norm_k_only
        
        self.rotary = Rotary(self.d_k, config.max_seq_len)
        self.dropout = config.dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # ============ MERGED QKV PROJECTION ============
        # Single matmul instead of 3 separate projections
        qkv = F.linear(x, self.qkvo_proj[:self.qkv_size])
        
        # Split the result into Q, K, V
        Q, K, V = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # ================================================
        
        # Reshape to multi-head format
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        V = V.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        
        # Apply Norm before RoPE (default) or after RoPE
        def apply_norm(q, k):
            if not self.use_qk_norm:
                return q, k
            
            # Key only norm
            if self.qk_norm_k_only:
                if hasattr(self, 'qk_shared_norm'):
                    k = self.qk_shared_norm(k)
                else:
                    k = self.k_norm(k)
            # Both Q/K norm
            else:
                if hasattr(self, 'qk_shared_norm'):
                    q = self.qk_shared_norm(q)
                    k = self.qk_shared_norm(k)
                else:
                    q = self.q_norm(q)
                    k = self.k_norm(k)
            
            # Apply Bias if enabled
            if hasattr(self, 'q_bias'):
                q = q + self.q_bias
                k = k + self.k_bias
            
            return q, k

        if not self.qk_norm_after_rope:
            Q, K = apply_norm(Q, K)
        
        Q = self.rotary(Q)
        K = self.rotary(K)

        if self.qk_norm_after_rope:
            Q, K = apply_norm(Q, K)
        
        # Repeat K/V for GQA if needed
        if self.n_kv_heads != self.n_heads:
            K = torch.repeat_interleave(K, self.num_key_value_groups, dim=2)
            V = torch.repeat_interleave(V, self.num_key_value_groups, dim=2)
        
        # Transpose for attention
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        
        # Apply head scaling if enabled
        if hasattr(self, 'head_scale'):
            # sqrt(scale) so that Q*K has linear head_scale
            # We use Q * head_scale and let SDPA handle the 1/sqrt(dk)
            Q = Q * self.head_scale

        # Compute attention
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.d_model
        )
        
        # ============ MERGED O PROJECTION ============
        # Use the last part of qkvo_proj for output projection
        return F.linear(attn_output, self.qkvo_proj[self.qkv_size:])


class TransformerBlock(nn.Module):
    """Standard transformer block with dense feed-forward"""

    def __init__(
        self,
        config: LLMConfig,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(config)
        self.feed_forward = SquaredReLUFeedForward(config.d_model, config.d_ff, config.dropout)

        # Normalization layers
        self.norm1 = nn.RMSNorm(config.d_model)
        self.norm2 = nn.RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Self-attention
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # Feed-forward
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x
