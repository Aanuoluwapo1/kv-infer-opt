"""Command-line interface for kv-infer-opt: comparative benchmarks and radix tree inspection."""

from __future__ import annotations

import argparse
import sys

import torch

from kv_infer_opt.engine import run_comparative_benchmark
from kv_infer_opt.paged_kv import BlockManager
from kv_infer_opt.radix_cache import RadixPrefixCache
from kv_infer_opt.tokenizer_bridge import HFTokenizerAdapter


def _cmd_benchmark(args: argparse.Namespace) -> int:
    metrics = run_comparative_benchmark(
        num_requests=args.requests,
        system_prompt_len=args.system_len,
        user_query_len=args.query_len,
        decode_tokens=args.decode_tokens,
        seed=args.seed,
    )

    print("=" * 118)
    print(" KV-INFER-OPT: COMPARATIVE LLM INFERENCE, PAGED/RADIX KV-CACHE & QUANTIZATION BENCHMARK")
    print("=" * 118)
    header = (
        f"{'Strategy':<35} | {'Weights (MB)':>12} | {'Peak KV (KB)':>12} | "
        f"{'Frag (%)':>8} | {'TTFT (ms)':>9} | {'Decode (tok/s)':>14} | {'Cosine Sim':>10}"
    )
    print(header)
    print("-" * 118)
    for m in metrics:
        print(
            f"{m.strategy_name:<35} | {m.weight_mb:12.2f} | {m.peak_kv_kb:12.1f} | "
            f"{m.fragmentation_pct:7.1f}% | {m.ttft_ms:9.2f} | {m.decode_tok_per_sec:14.1f} | "
            f"{m.logit_cosine_sim:10.6f}"
        )
    print("=" * 118)
    return 0


def _cmd_inspect_radix(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    tokenizer = HFTokenizerAdapter.build_local_bpe(vocab_size=1024)
    block_size = 4
    bm = BlockManager(max_blocks=32, block_size=block_size)
    radix = RadixPrefixCache(block_manager=bm, block_size=block_size)

    print("=" * 96)
    print(
        f" KV-INFER-OPT: SGLANG-STYLE RADIX ATTENTION PREFIX TREE & LRU EVICTION DEMO (block_size={block_size})"
    )
    print("=" * 96)

    sys_prompt = "You are a helpful systems engineering assistant specializing in LLM inference."
    sys_ids = tokenizer.encode(sys_prompt, add_bos=True)[:16]  # 16 tokens = 4 blocks

    q1_ids = sys_ids + tokenizer.encode("PagedAttention block tables", add_bos=False)[:8]
    q2_ids = sys_ids + tokenizer.encode("RadixAttention LRU eviction", add_bos=False)[:8]
    q3_ids = tokenizer.encode(
        "Unrelated standalone prompt about GGUF Q4_0 block quantization scales.", add_bos=True
    )[:12]

    for idx, (label, seq) in enumerate(
        [
            ("Request 1 (System Prompt + PagedAttention Query)", q1_ids),
            ("Request 2 (Shared System Prompt + Radix Query)", q2_ids),
            ("Request 3 (Distinct Prompt - GGUF Quantization)", q3_ids),
        ],
        start=1,
    ):
        match = radix.match_prefix(seq)
        missing_tokens = len(seq) - match.num_matched_tokens
        needed_blocks = missing_tokens // block_size
        new_blocks = [bm.allocate() for _ in range(needed_blocks)]
        all_blocks = match.matched_blocks + new_blocks
        radix.insert(seq, all_blocks)
        # Release temporary request allocation (radix.insert incremented refcount for cached blocks)
        for blk in new_blocks:
            bm.free(blk)

        print(
            f"[{idx}] {label}\n"
            f"    Total Prompt Tokens : {len(seq)} | Matched Prefix Tokens : {match.num_matched_tokens} "
            f"| Saved Prefill Compute : {(match.num_matched_tokens / len(seq)) * 100.0:.1f}%"
        )

    print("\n--- Radix Tree Structure Before Eviction ---")
    for line in radix.format_tree():
        print("  " + line)

    evicted = radix.evict(num_blocks_needed=2)
    print(f"\n--- After LRU Eviction (requested=2 blocks, evicted={evicted} blocks) ---")
    for line in radix.format_tree():
        print("  " + line)
    print("=" * 96)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kv-infer-opt",
        description="High-Throughput Transformer Inference, Paged/Radix KV-Cache & Quantization Engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_bench = sub.add_parser(
        "benchmark",
        help="Run end-to-end comparative benchmark across Naive, Ring, PagedKV, Radix, INT8, and Q4_0.",
    )
    p_bench.add_argument("--requests", type=int, default=6)
    p_bench.add_argument("--system-len", type=int, default=64)
    p_bench.add_argument("--query-len", type=int, default=16)
    p_bench.add_argument("--decode-tokens", type=int, default=16)
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.set_defaults(func=_cmd_benchmark)

    p_radix = sub.add_parser(
        "inspect-radix",
        help="Demonstrate SGLang-style RadixPrefixCache prefix hits, edge splitting, and LRU eviction.",
    )
    p_radix.add_argument("--seed", type=int, default=42)
    p_radix.set_defaults(func=_cmd_inspect_radix)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
