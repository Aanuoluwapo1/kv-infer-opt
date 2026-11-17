"""Tests for NaiveKVCache, StaticRingKVCache, and vLLM-style PagedKVCache parity and CoW."""

from __future__ import annotations

import pytest
import torch

from kv_infer_opt import (
    LlamaConfig,
    MicroLlama,
    NaiveKVCache,
    PagedKVCache,
    StaticRingKVCache,
)


class TestPagedKVAndCacheParity:
    """Verify exact numerical parity across Naive, Ring, and PagedKV backends plus CoW forking."""

    def test_prefill_and_multi_step_decode_exact_parity(self) -> None:
        torch.manual_seed(42)
        cfg = LlamaConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
        model = MicroLlama(cfg)
        model.eval()

        naive = NaiveKVCache(
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            max_seq_len=128,
        )
        ring = StaticRingKVCache(
            num_layers=cfg.num_hidden_layers,
            batch_size=1,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            max_seq_len=128,
        )
        paged = PagedKVCache(
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            block_size=8,
            max_blocks=32,
        )

        prompt = torch.randint(4, 250, (1, 19), dtype=torch.long)
        with torch.no_grad():
            logits_naive = model(prompt, kv_cache=naive)
            logits_ring = model(prompt, kv_cache=ring)
            logits_paged = model(prompt, kv_cache=paged)

            torch.testing.assert_close(logits_naive, logits_ring, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(logits_naive, logits_paged, atol=1e-5, rtol=1e-5)

            # Decode 10 tokens step-by-step across block boundaries (19 -> 29 crosses block 24)
            for step in range(10):
                next_tok = torch.tensor([[10 + step]], dtype=torch.long)
                out_n = model(next_tok, kv_cache=naive)
                out_r = model(next_tok, kv_cache=ring)
                out_p = model(next_tok, kv_cache=paged)
                torch.testing.assert_close(out_n, out_r, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(out_n, out_p, atol=1e-5, rtol=1e-5)

        assert paged.get_seq_len(0) == 29
        # 29 tokens with block_size=8 uses ceil(29/8)=4 physical blocks (32 slots -> 3 unused slots)
        assert len(paged.get_block_table(0)) == 4
        assert paged.fragmentation_ratio() == pytest.approx(3.0 / 32.0)
        assert paged.allocated_bytes() < naive.allocated_bytes()

    def test_copy_on_write_fork_and_block_reclamation(self) -> None:
        torch.manual_seed(99)
        paged = PagedKVCache(
            num_layers=2,
            num_kv_heads=2,
            head_dim=8,
            block_size=4,
            max_blocks=16,
        )
        # Prefill 6 tokens into seq_id=0 -> occupies 2 blocks (block 0 full, block 1 has 2/4 slots)
        k_init = torch.randn(1, 2, 6, 8)
        v_init = torch.randn(1, 2, 6, 8)
        for layer in range(2):
            paged.append(layer, k_init, v_init, seq_ids=[0])

        assert paged.block_manager.num_allocated_blocks == 2
        # Fork seq 0 -> seq 1 (zero-copy: still only 2 physical blocks allocated!)
        paged.fork_sequence(src_seq_id=0, dst_seq_id=1)
        assert paged.block_manager.num_allocated_blocks == 2
        assert paged.get_block_table(0) == paged.get_block_table(1)

        # Appending 1 token to seq 1 triggers Copy-on-Write ONLY on the shared tail block
        k_step = torch.randn(1, 2, 1, 8)
        v_step = torch.randn(1, 2, 1, 8)
        for layer in range(2):
            paged.append(layer, k_step, v_step, seq_ids=[1])

        assert paged.cow_copies == 1
        assert paged.block_manager.num_allocated_blocks == 3
        assert paged.get_block_table(0)[0] == paged.get_block_table(1)[0]
        assert paged.get_block_table(0)[1] != paged.get_block_table(1)[1]

        # Freeing both sequences returns all 3 blocks to the free pool
        paged.free_sequence(0)
        paged.free_sequence(1)
        assert paged.block_manager.num_allocated_blocks == 0
        assert paged.block_manager.num_free_blocks == 16

    def test_block_manager_exhaustion_raises_memory_error(self) -> None:
        paged = PagedKVCache(
            num_layers=1,
            num_kv_heads=2,
            head_dim=8,
            block_size=4,
            max_blocks=2,
        )
        # 2 blocks = 8 tokens capacity; requesting 9 tokens must raise MemoryError cleanly
        k_too_large = torch.randn(1, 2, 9, 8)
        v_too_large = torch.randn(1, 2, 9, 8)
        with pytest.raises(MemoryError, match="out of physical KV blocks"):
            paged.append(0, k_too_large, v_too_large, seq_ids=[0])
