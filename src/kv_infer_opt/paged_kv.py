"""KV-Cache backends: Naive torch.cat, Static Ring Buffer, and vLLM-style PagedAttention."""

from __future__ import annotations

from typing import Protocol

import torch


class KVCacheProtocol(Protocol):
    """Unified interface for Transformer key-value cache backends."""

    def append(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        seq_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new key/value states `(B, H_kv, T_new, D)` and return full `(B, H_kv, T_total, D)`."""
        ...

    def get_seq_len(self, seq_id: int = 0) -> int:
        """Return the current cached sequence length for `seq_id`."""
        ...

    def allocated_bytes(self) -> int:
        """Return total physical memory bytes currently reserved by the cache."""
        ...

    def fragmentation_ratio(self) -> float:
        """Return internal/external memory fragmentation fraction in `[0.0, 1.0]`."""
        ...

    def reset(self) -> None:
        """Clear all cached key-value states."""
        ...


class NaiveKVCache:
    """Baseline KV-Cache that dynamically reallocates and concatenates (`torch.cat`) on every step.

    Simulates naive autoregressive decoding where every token step allocates a brand-new
    contiguous tensor of size `T + 1` and copies the previous `T` tokens, plus over-reserves
    worst-case `max_seq_len` contiguous memory in standard static serving reservations.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 512,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        self.batch_size = int(batch_size)
        self.dtype = dtype
        self.k_cache: list[torch.Tensor | None] = [None] * self.num_layers
        self.v_cache: list[torch.Tensor | None] = [None] * self.num_layers
        self.realloc_copies_bytes: int = 0

    def append(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        seq_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del seq_ids
        prev_k = self.k_cache[layer_idx]
        prev_v = self.v_cache[layer_idx]
        if prev_k is None or prev_v is None:
            k_full = k_new.detach().clone()
            v_full = v_new.detach().clone()
        else:
            # Naive serving without pre-allocated blocks must allocate a fresh contiguous buffer
            # and copy the entire historical prefix on every single token decode step
            k_full = torch.cat([prev_k.clone(), k_new.detach().clone()], dim=2)
            v_full = torch.cat([prev_v.clone(), v_new.detach().clone()], dim=2)
            self.realloc_copies_bytes += 2 * k_full.numel() * k_full.element_size()

        self.k_cache[layer_idx] = k_full
        self.v_cache[layer_idx] = v_full
        return k_full, v_full

    def get_seq_len(self, seq_id: int = 0) -> int:
        del seq_id
        first = self.k_cache[0]
        return int(first.shape[2]) if first is not None else 0

    def allocated_bytes(self) -> int:
        """Naive serving pre-reserves worst-case `max_seq_len` per active sequence slot."""
        elem_bytes = torch.tensor([], dtype=self.dtype).element_size()
        return (
            2
            * self.num_layers
            * self.batch_size
            * self.num_kv_heads
            * self.max_seq_len
            * self.head_dim
            * elem_bytes
        )

    def active_token_bytes(self) -> int:
        total = 0
        for k, v in zip(self.k_cache, self.v_cache, strict=True):
            if k is not None and v is not None:
                total += (k.numel() + v.numel()) * k.element_size()
        return total

    def fragmentation_ratio(self) -> float:
        alloc = self.allocated_bytes()
        if alloc == 0:
            return 0.0
        used = self.active_token_bytes()
        return max(0.0, 1.0 - (float(used) / float(alloc)))

    def reset(self) -> None:
        self.k_cache = [None] * self.num_layers
        self.v_cache = [None] * self.num_layers
        self.realloc_copies_bytes = 0


class StaticRingKVCache:
    """Pre-allocated contiguous ring/slab KV-Cache (`max_seq_len` fixed buffer).

    Eliminates per-step `torch.cat` reallocation via in-place slice writes, but still
    suffers high internal fragmentation when sequences are shorter than `max_seq_len`.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 512,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_layers = int(num_layers)
        self.batch_size = int(batch_size)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        self.dtype = dtype
        self.device = torch.device(device)

        shape = (
            self.num_layers,
            self.batch_size,
            self.num_kv_heads,
            self.max_seq_len,
            self.head_dim,
        )
        self.k_buffer = torch.zeros(shape, dtype=dtype, device=self.device)
        self.v_buffer = torch.zeros(shape, dtype=dtype, device=self.device)
        self.seq_len: int = 0

    def append(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        seq_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del seq_ids
        t_new = int(k_new.shape[2])
        start = self.seq_len
        end = start + t_new
        if end > self.max_seq_len:
            raise RuntimeError(
                f"StaticRingKVCache overflow: requested {end} tokens > max_seq_len={self.max_seq_len}"
            )
        self.k_buffer[layer_idx, : k_new.shape[0], :, start:end, :] = k_new.detach()
        self.v_buffer[layer_idx, : v_new.shape[0], :, start:end, :] = v_new.detach()
        if layer_idx == self.num_layers - 1:
            self.seq_len = end
        return (
            self.k_buffer[layer_idx, : k_new.shape[0], :, :end, :],
            self.v_buffer[layer_idx, : v_new.shape[0], :, :end, :],
        )

    def get_seq_len(self, seq_id: int = 0) -> int:
        del seq_id
        return self.seq_len

    def allocated_bytes(self) -> int:
        return (self.k_buffer.numel() + self.v_buffer.numel()) * self.k_buffer.element_size()

    def fragmentation_ratio(self) -> float:
        if self.max_seq_len == 0:
            return 0.0
        return max(0.0, 1.0 - (float(self.seq_len) / float(self.max_seq_len)))

    def reset(self) -> None:
        self.seq_len = 0
        self.k_buffer.zero_()
        self.v_buffer.zero_()


class BlockManager:
    """Physical KV block pool allocator with reference counting and Copy-on-Write (CoW)."""

    def __init__(self, max_blocks: int, block_size: int = 16) -> None:
        if max_blocks <= 0 or block_size <= 0:
            raise ValueError("max_blocks and block_size must be positive integers.")
        self.max_blocks = int(max_blocks)
        self.block_size = int(block_size)
        self.free_blocks: list[int] = list(range(self.max_blocks - 1, -1, -1))
        self.ref_counts: dict[int, int] = dict.fromkeys(range(self.max_blocks), 0)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        return self.max_blocks - len(self.free_blocks)

    def allocate(self) -> int:
        """Pop a free physical block ID and initialize its reference count to 1."""
        if not self.free_blocks:
            raise MemoryError(
                f"PagedKV BlockManager out of physical KV blocks (max_blocks={self.max_blocks})."
            )
        block_id = self.free_blocks.pop()
        self.ref_counts[block_id] = 1
        return block_id

    def incref(self, block_id: int) -> None:
        """Increment reference count on an allocated physical block (for prefix sharing / CoW)."""
        if self.ref_counts.get(block_id, 0) <= 0:
            self.free_blocks.remove(block_id)
            self.ref_counts[block_id] = 1
        else:
            self.ref_counts[block_id] += 1

    def free(self, block_id: int) -> bool:
        """Decrement reference count and return block to free pool if count reaches 0."""
        count = self.ref_counts.get(block_id, 0)
        if count <= 0:
            return False
        count -= 1
        self.ref_counts[block_id] = count
        if count == 0:
            self.free_blocks.append(block_id)
            return True
        return False

    def reset(self) -> None:
        self.free_blocks = list(range(self.max_blocks - 1, -1, -1))
        self.ref_counts = dict.fromkeys(range(self.max_blocks), 0)


class PagedKVCache:
    """vLLM-Style PagedAttention KV-Cache Engine with logical-to-physical Block Tables.

    Partitions KV cache memory into fixed-size physical blocks of `block_size` tokens:
      `k_pool / v_pool shape = (num_layers, max_blocks, num_kv_heads, block_size, head_dim)`

    Features:
    - On-demand physical block allocation (`block_size = 16`), eliminating contiguous
      over-reservation and restricting internal fragmentation to at most `< block_size` tokens
      on the final partially filled tail block of each sequence.
    - Copy-on-Write (`fork_sequence(src_seq_id, dst_seq_id)`) sharing physical blocks with
      zero tensor copying until a shared tail block is mutated.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_blocks: int = 256,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.block_size = int(block_size)
        self.max_blocks = int(max_blocks)
        self.dtype = dtype
        self.device = torch.device(device)

        self.block_manager = BlockManager(max_blocks=self.max_blocks, block_size=self.block_size)
        pool_shape = (
            self.num_layers,
            self.max_blocks,
            self.num_kv_heads,
            self.block_size,
            self.head_dim,
        )
        self.k_pool = torch.zeros(pool_shape, dtype=dtype, device=self.device)
        self.v_pool = torch.zeros(pool_shape, dtype=dtype, device=self.device)

        # Per-sequence logical block tables and token counts
        self.block_tables: dict[int, list[int]] = {}
        self.seq_lens: dict[int, int] = {}
        self._step_start_lens: dict[int, int] = {}
        self.cow_copies: int = 0

    def _ensure_seq(self, seq_id: int) -> None:
        if seq_id not in self.block_tables:
            self.block_tables[seq_id] = []
            self.seq_lens[seq_id] = 0

    def fork_sequence(self, src_seq_id: int, dst_seq_id: int) -> None:
        """Zero-copy fork of `src_seq_id`'s block table into `dst_seq_id` using Copy-on-Write."""
        if src_seq_id not in self.block_tables:
            raise KeyError(f"Unknown source sequence ID {src_seq_id}")
        if dst_seq_id in self.block_tables:
            self.free_sequence(dst_seq_id)

        src_blocks = list(self.block_tables[src_seq_id])
        for blk in src_blocks:
            self.block_manager.incref(blk)
        self.block_tables[dst_seq_id] = src_blocks
        self.seq_lens[dst_seq_id] = self.seq_lens[src_seq_id]

    def mount_prefix_blocks(self, seq_id: int, prefix_blocks: list[int], prefix_len: int) -> None:
        """Attach pre-populated physical blocks (e.g. from `RadixPrefixCache`) to `seq_id`."""
        if seq_id in self.block_tables:
            self.free_sequence(seq_id)
        for blk in prefix_blocks:
            self.block_manager.incref(blk)
        self.block_tables[seq_id] = list(prefix_blocks)
        self.seq_lens[seq_id] = int(prefix_len)

    def free_sequence(self, seq_id: int) -> None:
        """Release all physical blocks held by `seq_id`."""
        if seq_id not in self.block_tables:
            return
        for blk in self.block_tables[seq_id]:
            self.block_manager.free(blk)
        del self.block_tables[seq_id]
        del self.seq_lens[seq_id]
        self._step_start_lens.pop(seq_id, None)

    def _prepare_sequence_blocks_for_step(self, seq_id: int, t_new: int) -> int:
        """Ensure `seq_id` has writable physical blocks for appending `t_new` tokens."""
        self._ensure_seq(seq_id)
        start_len = self.seq_lens[seq_id]
        table = self.block_tables[seq_id]

        for offset in range(t_new):
            pos = start_len + offset
            logical_idx = pos // self.block_size
            if logical_idx >= len(table):
                new_blk = self.block_manager.allocate()
                table.append(new_blk)
            else:
                phys_blk = table[logical_idx]
                # Copy-on-Write if writing into a shared block (ref_count > 1)
                if self.block_manager.ref_counts.get(phys_blk, 0) > 1:
                    new_blk = self.block_manager.allocate()
                    self.k_pool[:, new_blk].copy_(self.k_pool[:, phys_blk])
                    self.v_pool[:, new_blk].copy_(self.v_pool[:, phys_blk])
                    self.block_manager.free(phys_blk)
                    table[logical_idx] = new_blk
                    self.cow_copies += 1

        self.seq_lens[seq_id] = start_len + t_new
        return start_len

    def append(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        seq_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write `k_new, v_new` of shape `(B, H_kv, T_new, D)` into paged physical blocks
        and gather the contiguous view `(B, H_kv, T_total, D)` for attention computation.
        """
        bsz, _h_kv, t_new, _d = k_new.shape
        active_seq_ids = seq_ids if seq_ids is not None else list(range(bsz))
        if len(active_seq_ids) != bsz:
            raise ValueError(f"Expected {bsz} seq_ids, got {len(active_seq_ids)}")

        if layer_idx == 0:
            for sid in active_seq_ids:
                self._step_start_lens[sid] = self._prepare_sequence_blocks_for_step(sid, t_new)

        gathered_k: list[torch.Tensor] = []
        gathered_v: list[torch.Tensor] = []

        for b_idx, sid in enumerate(active_seq_ids):
            start_len = self._step_start_lens[sid]
            total_len = self.seq_lens[sid]
            table = self.block_tables[sid]

            if t_new == 1:
                phys_blk = table[start_len // self.block_size]
                slot = start_len % self.block_size
                self.k_pool[layer_idx, phys_blk, :, slot, :] = k_new[b_idx, :, 0, :].detach()
                self.v_pool[layer_idx, phys_blk, :, slot, :] = v_new[b_idx, :, 0, :].detach()
            else:
                positions = torch.arange(
                    start_len, start_len + t_new, dtype=torch.long, device=self.device
                )
                blk_indices = [table[int(p) // self.block_size] for p in positions]
                blk_tensor = torch.tensor(blk_indices, dtype=torch.long, device=self.device)
                slot_tensor = positions % self.block_size
                self.k_pool[layer_idx, blk_tensor, :, slot_tensor, :] = (
                    k_new[b_idx].detach().transpose(0, 1)
                )
                self.v_pool[layer_idx, blk_tensor, :, slot_tensor, :] = (
                    v_new[b_idx].detach().transpose(0, 1)
                )

            # Gather logical blocks up to total_len
            num_logical = (total_len + self.block_size - 1) // self.block_size
            active_blocks = table[:num_logical]
            # (num_logical, H_kv, block_size, D) -> (H_kv, num_logical * block_size, D)
            k_seq = (
                self.k_pool[layer_idx, active_blocks]
                .permute(1, 0, 2, 3)
                .reshape(self.num_kv_heads, num_logical * self.block_size, self.head_dim)[
                    :, :total_len, :
                ]
            )
            v_seq = (
                self.v_pool[layer_idx, active_blocks]
                .permute(1, 0, 2, 3)
                .reshape(self.num_kv_heads, num_logical * self.block_size, self.head_dim)[
                    :, :total_len, :
                ]
            )
            gathered_k.append(k_seq)
            gathered_v.append(v_seq)

        return torch.stack(gathered_k, dim=0), torch.stack(gathered_v, dim=0)

    def get_seq_len(self, seq_id: int = 0) -> int:
        return self.seq_lens.get(seq_id, 0)

    def get_block_table(self, seq_id: int = 0) -> list[int]:
        return list(self.block_tables.get(seq_id, []))

    def allocated_bytes(self) -> int:
        """Return physical bytes occupied by currently allocated blocks across all layers."""
        elem_bytes = self.k_pool.element_size()
        num_alloc = self.block_manager.num_allocated_blocks
        return (
            2
            * self.num_layers
            * num_alloc
            * self.num_kv_heads
            * self.block_size
            * self.head_dim
            * elem_bytes
        )

    def active_token_bytes(self) -> int:
        """Return exact bytes corresponding to valid tokens stored across unique sequences."""
        elem_bytes = self.k_pool.element_size()
        total_tokens = sum(self.seq_lens.values())
        return 2 * self.num_layers * total_tokens * self.num_kv_heads * self.head_dim * elem_bytes

    def fragmentation_ratio(self) -> float:
        """Compute internal fragmentation within allocated pages (`wasted_slots / total_slots`)."""
        total_slots = sum(len(tbl) * self.block_size for tbl in self.block_tables.values())
        if total_slots == 0:
            return 0.0
        used_slots = sum(self.seq_lens.values())
        return max(0.0, 1.0 - (float(used_slots) / float(total_slots)))

    def reset(self) -> None:
        self.block_manager.reset()
        self.block_tables.clear()
        self.seq_lens.clear()
        self._step_start_lens.clear()
        self.cow_copies = 0
        self.k_pool.zero_()
        self.v_pool.zero_()
