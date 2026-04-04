# Architecture Notes

## LLM Architecture

- **Model**: 88M parameters default (configurable)
- **Layers**: 22 Transformer blocks
- **Hidden Dimension**: 512
- **Feed-Forward Dimension**: 2048
- **Attention**: 8 query heads, 4 KV heads (Grouped Query Attention)
- **Positional Encoding**: Rotary (RoPE)
- **Normalization**: Pre-norm RMSNorm
- **Activation**: Squared ReLU (Primer-style)
- **Vocab Size**: 49,152
- **Sequence Length**: 2048 tokens

## Key Components

- Weight tying between token embeddings and LM head
- Fused QKVO projection
- QK-Normalization for training stability
- Muon optimizer support (orthogonal updates)