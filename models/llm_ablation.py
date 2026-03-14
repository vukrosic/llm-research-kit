"""
MinimalLLMAblation
==================
Extended ablation model that wires up ALL new config flags from the swarm
of 40 architecture experiments.
"""

import torch
import torch.nn as nn
import math

from models.layers_ablation import (
    TransformerBlockAblation,
    ParallelTransformerBlock,
    build_norm,
)


class MinimalLLMAblation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ── Config flags with safe defaults ──────────────────────────────
        self.use_embed_scale  = getattr(config, 'use_embed_scale', True)
        use_qk_norm           = getattr(config, 'use_qk_norm', True)
        activation_type       = getattr(config, 'activation_type', 'squared_relu')
        rope_base             = getattr(config, 'rope_base', 10000.0)
        norm_type             = getattr(config, 'norm_type', 'rmsnorm')
        norm_position         = getattr(config, 'norm_position', 'pre')
        ffn_type              = getattr(config, 'ffn_type', 'standard')
        use_rope              = getattr(config, 'use_rope', True)
        use_bias              = getattr(config, 'use_bias', False)
        parallel_block        = getattr(config, 'parallel_block', False)
        use_learned_pos       = getattr(config, 'use_learned_pos', False)
        tie_weights           = getattr(config, 'tie_weights', True)
        init_scheme           = getattr(config, 'init_scheme', 'default')  # 'default' | 'depth_scaled' | 'gpt2' | 'small_embed'
        residual_scale        = getattr(config, 'residual_scale', 1.0)
        final_norm_type       = getattr(config, 'final_norm_type', norm_type)
        n_kv_heads            = getattr(config, 'n_kv_heads', None)

        # ── Embeddings ────────────────────────────────────────────────────
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        if use_learned_pos:
            self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        else:
            self.pos_embedding = None

        # ── New feature flags ─────────────────────────────────────────────
        value_norm        = getattr(config, 'value_norm', False)
        layer_scale_init  = getattr(config, 'layer_scale_init', None)
        stochastic_depth  = getattr(config, 'stochastic_depth', 0.0)
        # Gen9/10 novel mechanisms
        cosine_attn       = getattr(config, 'cosine_attn', False)
        q_rope_only       = getattr(config, 'q_rope_only', False)
        alibi             = getattr(config, 'alibi', False)
        gated_residual    = getattr(config, 'gated_residual', False)
        gate_init         = getattr(config, 'gate_init', 0.0)
        gate_per_channel  = getattr(config, 'gate_per_channel', False)

        # ── Transformer Blocks ────────────────────────────────────────────
        block_kwargs = dict(
            d_model         = config.d_model,
            n_heads         = config.n_heads,
            d_ff            = config.d_ff,
            max_seq_len     = config.max_seq_len,
            dropout         = config.dropout,
            n_kv_heads      = n_kv_heads,
            use_qk_norm     = use_qk_norm,
            activation_type = activation_type,
            rope_base       = rope_base,
            norm_type       = norm_type,
            ffn_type        = ffn_type,
            use_rope        = use_rope,
            use_bias        = use_bias,
            qk_norm_type    = getattr(config, 'qk_norm_type', 'rmsnorm'),
            use_q_norm      = getattr(config, 'use_q_norm', True),
            use_k_norm      = getattr(config, 'use_k_norm', True),
            attn_scale      = getattr(config, 'attn_scale', 1.0),
            attn_window_size= getattr(config, 'attn_window_size', None),
            attn_softcap    = getattr(config, 'attn_softcap', None),
            attn_activation = getattr(config, 'attn_activation', 'softmax'),
            use_shared_qkv  = getattr(config, 'use_shared_qkv', False),
            hilo_fraction   = getattr(config, 'hilo_fraction', None),
            kv_pool_factor  = getattr(config, 'kv_pool_factor', None),
            poly_order      = getattr(config, 'poly_order', None),
            value_norm      = value_norm,
            layer_scale_init= layer_scale_init,
            cosine_attn       = cosine_attn,
            q_rope_only       = q_rope_only,
            alibi             = alibi,
            gated_residual    = gated_residual,
            gate_init         = gate_init,
            gate_per_channel  = gate_per_channel,
        )

        n_layers = config.n_layers
        if parallel_block:
            self.transformer_blocks = nn.ModuleList([
                ParallelTransformerBlock(
                    **block_kwargs,
                    stochastic_depth_rate=stochastic_depth * (i / max(n_layers - 1, 1)),
                )
                for i in range(n_layers)
            ])
        else:
            self.transformer_blocks = nn.ModuleList([
                TransformerBlockAblation(
                    **block_kwargs,
                    norm_position=norm_position,
                    residual_scale=residual_scale,
                    stochastic_depth_rate=stochastic_depth * (i / max(n_layers - 1, 1)),
                )
                for i in range(n_layers)
            ])

        # ── Final norm ────────────────────────────────────────────────────
        self.norm = build_norm(final_norm_type, config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # ── LM Head ──────────────────────────────────────────────────────
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # ── Weight initialization ─────────────────────────────────────────
        self._init_scheme = init_scheme
        self._n_layers    = config.n_layers
        self.apply(self._init_weights)

    # ─────────────────────────────────────────────────────────────────────
    def _init_weights(self, module):
        scheme = self._init_scheme

        if isinstance(module, nn.Embedding):
            if scheme == 'small_embed':
                std = 0.002
            else:
                std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

        elif isinstance(module, nn.Linear):
            if scheme == 'gpt2':
                # GPT-2: output projections scaled by 1/sqrt(2*n_layers)
                std = 0.02 / math.sqrt(2 * self._n_layers)
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            elif scheme == 'depth_scaled':
                # Depth-scaled: each layer's init std is 1/sqrt(layer_depth)
                # We use a simple global 1/sqrt(n_layers) proxy
                std = 0.02 / math.sqrt(self._n_layers)
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        # nn.Parameter (e.g. qkvo_proj) are already initialized in the module

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, x):
        tok_emb = self.token_embedding(x)

        if self.use_embed_scale:
            tok_emb = tok_emb * math.sqrt(self.config.d_model)

        if self.pos_embedding is not None:
            positions = torch.arange(x.size(1), device=x.device)
            tok_emb = tok_emb + self.pos_embedding(positions)

        x = self.position_dropout(tok_emb)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        return self.lm_head(x)
