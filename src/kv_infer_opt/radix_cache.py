"""SGLang-Style RadixAttention Prefix Tree Cache for automatic KV-Cache reuse."""

from __future__ import annotations

from dataclasses import dataclass, field

from kv_infer_opt.paged_kv import BlockManager


@dataclass
class RadixNode:
    """A node in the RadixPrefixCache compressed trie.

    Each edge is keyed by `key[0]` (first token ID of the edge segment) and stores:
    - `key`: Tuple of block-aligned token IDs `(L = k * block_size)` along this edge.
    - `blocks`: Physical KV block IDs storing the computed key/value states for `key`.
    - `lock_ref`: Reference count of active in-flight requests currently locking this node.
    - `last_access_tick`: Monotonic logical timestamp for Least-Recently-Used (LRU) eviction.
    """

    key: tuple[int, ...] = ()
    blocks: list[int] = field(default_factory=list)
    parent: RadixNode | None = None
    children: dict[tuple[int, ...], RadixNode] = field(default_factory=dict)
    lock_ref: int = 0
    last_access_tick: int = 0


@dataclass(frozen=True)
class PrefixMatchResult:
    """Result of matching a prompt token sequence against `RadixPrefixCache`."""

    matched_tokens: tuple[int, ...]
    matched_blocks: list[int]
    last_node: RadixNode

    @property
    def num_matched_tokens(self) -> int:
        return len(self.matched_tokens)


class RadixPrefixCache:
    """SGLang-Style Radix Tree mapping token prefixes to physical `PagedKVCache` block IDs.

    Features:
    - Block-aligned compressed radix tree (`block_size` granularity) so matched prefixes
      mount directly into `PagedKVCache` block tables with zero tensor copying.
    - Automatic edge splitting when two prompts share a partial block-aligned prefix.
    - Reference counting (`inc_lock_ref` / `dec_lock_ref`) protecting active sequences
      from eviction.
    - Monotonic LRU leaf eviction (`evict(num_blocks_needed)`) freeing the oldest unlocked
      leaf blocks when the physical KV pool is under memory pressure.
    """

    def __init__(self, block_manager: BlockManager, block_size: int = 16) -> None:
        self.block_manager = block_manager
        self.block_size = int(block_size)
        self.root = RadixNode(key=(), blocks=[], parent=None)
        self._clock: int = 0
        self.total_queries: int = 0
        self.total_matched_tokens: int = 0
        self.total_requested_tokens: int = 0

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    @staticmethod
    def _common_prefix_len(a: tuple[int, ...], b: tuple[int, ...], block_size: int) -> int:
        limit = min(len(a), len(b))
        matched = 0
        while matched < limit and a[matched] == b[matched]:
            matched += 1
        # Align to block_size boundary so physical blocks map 1-to-1
        return (matched // block_size) * block_size

    def _edge_key(self, tokens: tuple[int, ...]) -> tuple[int, ...]:
        """Use the first full block of `tokens` as the unique child edge key."""
        return tokens[: self.block_size]

    def _split_node(self, child: RadixNode, split_len: int) -> RadixNode:
        """Split `child` at `split_len` tokens (`split_len` is a multiple of `block_size`)."""
        assert 0 < split_len < len(child.key)
        assert split_len % self.block_size == 0
        blk_split = split_len // self.block_size

        parent = child.parent
        assert parent is not None

        mid_node = RadixNode(
            key=child.key[:split_len],
            blocks=child.blocks[:blk_split],
            parent=parent,
            children={},
            lock_ref=child.lock_ref,
            last_access_tick=child.last_access_tick,
        )

        # Re-link parent -> mid_node -> child
        parent.children[self._edge_key(mid_node.key)] = mid_node
        child.key = child.key[split_len:]
        child.blocks = child.blocks[blk_split:]
        child.parent = mid_node
        mid_node.children[self._edge_key(child.key)] = child
        return mid_node

    def match_prefix(self, token_ids: list[int] | tuple[int, ...]) -> PrefixMatchResult:
        """Find the longest block-aligned prefix of `token_ids` cached in the radix tree."""
        tokens = tuple(int(t) for t in token_ids)
        aligned_len = (len(tokens) // self.block_size) * self.block_size
        aligned_tokens = tokens[:aligned_len]

        now = self._tick()
        self.root.last_access_tick = now
        self.total_queries += 1
        self.total_requested_tokens += len(tokens)

        if not aligned_tokens:
            return PrefixMatchResult(matched_tokens=(), matched_blocks=[], last_node=self.root)

        node = self.root
        matched_tokens: list[int] = []
        matched_blocks: list[int] = []
        remaining = aligned_tokens

        while remaining:
            edge_k = self._edge_key(remaining)
            if edge_k not in node.children:
                break
            child = node.children[edge_k]
            c_len = self._common_prefix_len(child.key, remaining, self.block_size)
            if c_len == 0:
                break

            if c_len < len(child.key):
                child = self._split_node(child, c_len)

            child.last_access_tick = now
            matched_tokens.extend(child.key)
            matched_blocks.extend(child.blocks)
            remaining = remaining[c_len:]
            node = child

        self.total_matched_tokens += len(matched_tokens)
        return PrefixMatchResult(
            matched_tokens=tuple(matched_tokens),
            matched_blocks=matched_blocks,
            last_node=node,
        )

    def insert(
        self,
        token_ids: list[int] | tuple[int, ...],
        block_ids: list[int],
    ) -> int:
        """Insert a block-aligned prefix of `token_ids` and its corresponding physical `block_ids`.

        Increments reference counts in `BlockManager` for newly cached suffix blocks so
        they persist in the pool after the originating request finishes.

        Returns:
            Number of newly inserted tokens added to the radix tree.
        """
        tokens = tuple(int(t) for t in token_ids)
        num_full_blocks = min(len(tokens) // self.block_size, len(block_ids))
        aligned_len = num_full_blocks * self.block_size
        if aligned_len == 0:
            return 0

        aligned_tokens = tokens[:aligned_len]
        aligned_blocks = list(block_ids[:num_full_blocks])
        now = self._tick()
        self.root.last_access_tick = now

        node = self.root
        remaining_tokens = aligned_tokens
        remaining_blocks = aligned_blocks

        while remaining_tokens:
            edge_k = self._edge_key(remaining_tokens)
            if edge_k not in node.children:
                break
            child = node.children[edge_k]
            c_len = self._common_prefix_len(child.key, remaining_tokens, self.block_size)
            if c_len == 0:
                break

            if c_len < len(child.key):
                child = self._split_node(child, c_len)

            child.last_access_tick = now
            blk_step = c_len // self.block_size
            remaining_tokens = remaining_tokens[c_len:]
            remaining_blocks = remaining_blocks[blk_step:]
            node = child

        if remaining_tokens:
            for blk in remaining_blocks:
                self.block_manager.incref(blk)
            new_leaf = RadixNode(
                key=remaining_tokens,
                blocks=list(remaining_blocks),
                parent=node,
                children={},
                lock_ref=0,
                last_access_tick=now,
            )
            node.children[self._edge_key(remaining_tokens)] = new_leaf
            return len(remaining_tokens)

        return 0

    def inc_lock_ref(self, node: RadixNode) -> None:
        """Pin `node` and all its ancestors so active blocks cannot be evicted mid-request."""
        curr: RadixNode | None = node
        while curr is not None and curr is not self.root:
            curr.lock_ref += 1
            curr = curr.parent

    def dec_lock_ref(self, node: RadixNode) -> None:
        """Unpin `node` and all its ancestors when a request completes."""
        curr: RadixNode | None = node
        while curr is not None and curr is not self.root:
            curr.lock_ref = max(0, curr.lock_ref - 1)
            curr = curr.parent

    def _collect_evictable_leaves(self) -> list[RadixNode]:
        leaves: list[RadixNode] = []
        stack: list[RadixNode] = list(self.root.children.values())
        while stack:
            node = stack.pop()
            if not node.children:
                if node.lock_ref == 0:
                    leaves.append(node)
            else:
                stack.extend(node.children.values())
        return leaves

    def evict(self, num_blocks_needed: int) -> int:
        """Evict unlocked leaf nodes in Least-Recently-Used (LRU) order until at least
        `num_blocks_needed` physical KV blocks have been released.

        Returns:
            Total number of physical blocks evicted from the tree.
        """
        freed_blocks = 0
        while freed_blocks < num_blocks_needed:
            leaves = self._collect_evictable_leaves()
            if not leaves:
                break
            # Pick leaf with smallest last_access_tick
            victim = min(leaves, key=lambda n: n.last_access_tick)
            parent = victim.parent
            assert parent is not None
            for blk in victim.blocks:
                self.block_manager.free(blk)
                freed_blocks += 1
            del parent.children[self._edge_key(victim.key)]
            victim.parent = None

        return freed_blocks

    def cached_block_count(self) -> int:
        """Return total number of physical KV blocks currently indexed in the radix tree."""
        total = 0
        stack: list[RadixNode] = list(self.root.children.values())
        while stack:
            node = stack.pop()
            total += len(node.blocks)
            stack.extend(node.children.values())
        return total

    def cached_token_count(self) -> int:
        return self.cached_block_count() * self.block_size

    @property
    def hit_rate(self) -> float:
        if self.total_requested_tokens == 0:
            return 0.0
        return float(self.total_matched_tokens) / float(self.total_requested_tokens)

    def format_tree(self) -> list[str]:
        """Render an ASCII visualization of all nodes in the radix prefix tree."""
        lines: list[str] = [f"<Root> (cached_blocks={self.cached_block_count()})"]

        def _walk(node: RadixNode, prefix: str, is_last: bool) -> None:
            connector = "`-- " if is_last else "|-- "
            tok_preview = list(node.key[:4])
            suffix = "..." if len(node.key) > 4 else ""
            lines.append(
                f"{prefix}{connector}tokens={tok_preview}{suffix} "
                f"(len={len(node.key)}, blocks={node.blocks}, "
                f"lock_ref={node.lock_ref}, lru_tick={node.last_access_tick})"
            )
            child_prefix = prefix + ("    " if is_last else "|   ")
            children = list(node.children.values())
            for idx, child in enumerate(children):
                _walk(child, child_prefix, idx == len(children) - 1)

        root_children = list(self.root.children.values())
        for idx, c in enumerate(root_children):
            _walk(c, "", idx == len(root_children) - 1)
        return lines
