import unittest

import torch

from configs.llm_config import FiveMillionConfig, LLMConfig, MiniMaxSparseConfig
from models.llm import MinimalLLM
from models.minimax_sparse_attention import MiniMaxSparseAttention, block_max_pool_topk


class BlockSelectionTest(unittest.TestCase):
    def test_block_max_pool_topk_is_causal(self):
        queries = torch.ones(1, 5, 2, 4)
        keys = torch.ones(1, 5, 4)
        selection = block_max_pool_topk(queries, keys, block_size=2, top_k=2)

        indices = selection.indices[0]
        masks = selection.mask[0]
        for token_idx in range(5):
            valid = indices[token_idx][masks[token_idx]]
            self.assertTrue(torch.all(valid <= token_idx // 2))

    def test_block_max_pool_retrieves_planted_block(self):
        torch.manual_seed(0)
        queries = torch.zeros(1, 8, 1, 4)
        keys = torch.zeros(1, 8, 4)
        queries[:, 6, 0, 0] = 8.0
        keys[:, 2:4, 0] = 8.0

        selection = block_max_pool_topk(queries, keys, block_size=2, top_k=1)
        self.assertEqual(selection.indices[0, 6, 0, 0].item(), 1)


class MiniMaxSparseAttentionTest(unittest.TestCase):
    def test_forward_backward_on_cpu(self):
        torch.manual_seed(1)
        attention = MiniMaxSparseAttention(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=16,
            block_size=4,
            top_k=2,
            index_dim=8,
            dropout=0.0,
        )
        hidden = torch.randn(2, 13, 32, requires_grad=True)

        output, debug = attention(hidden, return_debug=True)
        self.assertEqual(output.shape, hidden.shape)
        self.assertEqual(debug.selected_block_indices.shape, (2, 13, 2, 2))

        loss = output.square().mean()
        loss.backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_attention_cannot_read_future_tokens(self):
        torch.manual_seed(2)
        attention = MiniMaxSparseAttention(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=16,
            block_size=4,
            top_k=4,
            index_dim=8,
            dropout=0.0,
        )
        attention.eval()
        hidden = torch.randn(1, 12, 32)
        changed_future = hidden.clone()
        changed_future[:, 7:] = torch.randn_like(changed_future[:, 7:]) * 20.0

        original = attention(hidden)
        modified = attention(changed_future)
        torch.testing.assert_close(original[:, :7], modified[:, :7], atol=1e-5, rtol=1e-5)

    def test_minimal_llm_uses_minimax_sparse_attention(self):
        config = LLMConfig(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            n_layers=1,
            d_ff=64,
            max_seq_len=12,
            vocab_size=128,
            attention_impl="minimax_sparse",
            minimax_sparse=MiniMaxSparseConfig(block_size=3, top_k=2, index_dim=8),
        )
        model = MinimalLLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, config.max_seq_len))

        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, config.max_seq_len, config.vocab_size))

    def test_5m_sparse_config_keeps_baseline_budget(self):
        config = FiveMillionConfig(attention_impl="minimax_sparse")
        self.assertEqual(config.train_tokens, 8_000_000)
        self.assertEqual(config.max_seq_len, 2048)
        self.assertEqual(config.batch_size, 8)


if __name__ == "__main__":
    unittest.main()
