"""Tests for SGLang-style RadixPrefixCache: prefix matching, edge splitting, and LRU eviction."""

from __future__ import annotations

from kv_infer_opt import BlockManager, RadixPrefixCache


class TestRadixPrefixCache:
    """Verify radix tree insertion, node splitting, reference locking, and LRU eviction."""

    def test_prefix_matching_and_node_splitting(self) -> None:
        bm = BlockManager(max_blocks=32, block_size=4)
        radix = RadixPrefixCache(block_manager=bm, block_size=4)

        # Insert 12-token prompt (3 blocks: [0, 1, 2])
        prompt_a = [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33]
        blocks_a = [bm.allocate() for _ in range(3)]
        inserted_a = radix.insert(prompt_a, blocks_a)
        for b in blocks_a:
            bm.free(b)

        assert inserted_a == 12
        assert radix.cached_token_count() == 12

        # Query prompt_b sharing first 8 tokens (2 blocks) then diverging
        prompt_b = [10, 11, 12, 13, 20, 21, 22, 23, 90, 91, 92, 93]
        match_b = radix.match_prefix(prompt_b)
        assert match_b.num_matched_tokens == 8
        assert match_b.matched_blocks == blocks_a[:2]

        # Insert divergent suffix block for prompt_b
        new_blk = bm.allocate()
        inserted_b = radix.insert(prompt_b, match_b.matched_blocks + [new_blk])
        bm.free(new_blk)
        assert inserted_b == 4
        assert radix.cached_block_count() == 4

    def test_lru_eviction_and_lock_ref_protection(self) -> None:
        bm = BlockManager(max_blocks=16, block_size=4)
        radix = RadixPrefixCache(block_manager=bm, block_size=4)

        # Insert Branch 1 (older)
        seq_old = [1, 2, 3, 4, 5, 6, 7, 8]
        blks_old = [bm.allocate() for _ in range(2)]
        radix.insert(seq_old, blks_old)
        for b in blks_old:
            bm.free(b)

        # Insert Branch 2 (newer)
        seq_new = [50, 51, 52, 53, 54, 55, 56, 57]
        blks_new = [bm.allocate() for _ in range(2)]
        radix.insert(seq_new, blks_new)
        for b in blks_new:
            bm.free(b)

        assert bm.num_allocated_blocks == 4

        # Evict 2 blocks -> must evict seq_old (least recently used) and keep seq_new
        freed = radix.evict(num_blocks_needed=2)
        assert freed == 2
        assert bm.num_allocated_blocks == 2
        assert radix.match_prefix(seq_old).num_matched_tokens == 0
        assert radix.match_prefix(seq_new).num_matched_tokens == 8

        # Locking seq_new prevents eviction even when requested
        match_new = radix.match_prefix(seq_new)
        radix.inc_lock_ref(match_new.last_node)
        assert radix.evict(num_blocks_needed=2) == 0
        radix.dec_lock_ref(match_new.last_node)
        assert radix.evict(num_blocks_needed=2) == 2
        assert bm.num_allocated_blocks == 0

    def test_empty_single_token_and_duplicate_insertions(self) -> None:
        bm = BlockManager(max_blocks=16, block_size=4)
        radix = RadixPrefixCache(block_manager=bm, block_size=4)

        # 1. Empty token list and sub-block (<4 tokens) sequences return 0 matched/inserted
        assert radix.insert([], []) == 0
        assert radix.match_prefix([]).num_matched_tokens == 0
        assert radix.insert([42], [0]) == 0
        assert radix.match_prefix([42]).num_matched_tokens == 0

        # 2. Duplicate identical block-aligned prompt does not double-insert or leak blocks
        seq = [10, 11, 12, 13, 14, 15, 16, 17]
        blks = [bm.allocate() for _ in range(2)]
        assert radix.insert(seq, blks) == 8
        assert radix.insert(seq, blks) == 0
        for b in blks:
            bm.free(b)
        assert radix.cached_block_count() == 2
        assert bm.num_allocated_blocks == 2
