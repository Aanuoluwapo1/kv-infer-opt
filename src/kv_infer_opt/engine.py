"""High-throughput LLM inference engine and comparative benchmark harness."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from kv_infer_opt.model import LlamaConfig, MicroLlama
from kv_infer_opt.paged_kv import (
    KVCacheProtocol,
    NaiveKVCache,
    PagedKVCache,
    StaticRingKVCache,
)
from kv_infer_opt.quantization import (
    compute_logit_fidelity,
    model_weight_bytes,
    quantize_model,
)
from kv_infer_opt.radix_cache import RadixPrefixCache


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Performance, memory, and numerical accuracy metrics for a single serving configuration."""

    strategy_name: str
    weight_mb: float
    peak_kv_kb: float
    fragmentation_pct: float
    ttft_ms: float
    decode_tok_per_sec: float
    prefix_hit_rate_pct: float
    logit_cosine_sim: float


class InferenceEngine:
    """Autoregressive prefill + decode engine supporting Naive, Ring, PagedKV, and RadixPrefixCache."""

    def __init__(
        self,
        model: MicroLlama,
        kv_backend: str = "paged",
        use_radix_cache: bool = False,
        block_size: int = 16,
        max_blocks: int = 256,
        max_seq_len: int = 512,
        device: str | torch.device = "cpu",
    ) -> None:
        if kv_backend not in ("naive", "ring", "paged"):
            raise ValueError(f"Unsupported kv_backend: {kv_backend!r}")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.config = model.config
        self.kv_backend = kv_backend
        self.use_radix_cache = bool(use_radix_cache)
        self.block_size = int(block_size)
        self.max_blocks = int(max_blocks)
        self.max_seq_len = int(max_seq_len)

        self.kv_cache: KVCacheProtocol
        self.radix_cache: RadixPrefixCache | None = None

        if kv_backend == "naive":
            self.kv_cache = NaiveKVCache(
                num_layers=self.config.num_hidden_layers,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                max_seq_len=self.max_seq_len,
            )
        elif kv_backend == "ring":
            self.kv_cache = StaticRingKVCache(
                num_layers=self.config.num_hidden_layers,
                batch_size=1,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                max_seq_len=self.max_seq_len,
                device=self.device,
            )
        else:
            paged = PagedKVCache(
                num_layers=self.config.num_hidden_layers,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                block_size=self.block_size,
                max_blocks=self.max_blocks,
                device=self.device,
            )
            self.kv_cache = paged
            if self.use_radix_cache:
                self.radix_cache = RadixPrefixCache(
                    block_manager=paged.block_manager,
                    block_size=self.block_size,
                )

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 16,
        seq_id: int = 0,
        retain_kv_for_prefix_cache: bool = True,
    ) -> dict[str, object]:
        """Run prefill + greedy autoregressive decoding and return generated IDs and timings."""
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty.")

        t0 = time.perf_counter()
        matched_len = 0
        matched_node = None

        if isinstance(self.kv_cache, PagedKVCache) and self.radix_cache is not None:
            # Always keep at least the last prompt token for forward prefill to produce next-token logits
            searchable = prompt_ids[:-1] if len(prompt_ids) > 1 else []
            match = self.radix_cache.match_prefix(searchable)
            matched_len = match.num_matched_tokens
            matched_node = match.last_node
            if matched_len > 0:
                self.radix_cache.inc_lock_ref(matched_node)
                self.kv_cache.mount_prefix_blocks(
                    seq_id=seq_id,
                    prefix_blocks=match.matched_blocks,
                    prefix_len=matched_len,
                )

        # Prefill only the suffix not already mounted from RadixPrefixCache
        suffix_ids = prompt_ids[matched_len:]
        inp_tensor = torch.tensor([suffix_ids], dtype=torch.long, device=self.device)
        prefill_logits = self.model(
            inp_tensor,
            kv_cache=self.kv_cache,
            position_offset=matched_len,
            seq_ids=[seq_id],
        )
        ttft_ms = (time.perf_counter() - t0) * 1e3

        # Cache full block-aligned prompt prefix in RadixPrefixCache for subsequent requests
        if (
            isinstance(self.kv_cache, PagedKVCache)
            and self.radix_cache is not None
            and retain_kv_for_prefix_cache
        ):
            self.radix_cache.insert(
                prompt_ids,
                self.kv_cache.get_block_table(seq_id),
            )

        next_token = int(torch.argmax(prefill_logits[0, -1, :]).item())
        generated: list[int] = [next_token]

        t_decode_start = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            step_inp = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            step_logits = self.model(
                step_inp,
                kv_cache=self.kv_cache,
                seq_ids=[seq_id],
            )
            next_token = int(torch.argmax(step_logits[0, -1, :]).item())
            generated.append(next_token)
        decode_elapsed = max(time.perf_counter() - t_decode_start, 1e-9)

        if matched_node is not None and self.radix_cache is not None:
            self.radix_cache.dec_lock_ref(matched_node)

        return {
            "prompt_ids": prompt_ids,
            "generated_ids": generated,
            "matched_prefix_tokens": matched_len,
            "ttft_ms": ttft_ms,
            "decode_tok_per_sec": float(max(len(generated) - 1, 1)) / decode_elapsed,
            "prefill_last_logits": prefill_logits[0, -1, :].detach().clone(),
        }


@torch.no_grad()
def run_comparative_benchmark(
    num_requests: int = 6,
    system_prompt_len: int = 64,
    user_query_len: int = 16,
    decode_tokens: int = 16,
    seed: int = 42,
) -> list[BenchmarkMetrics]:
    """Run an end-to-end comparative benchmark across all 6 KV-cache & quantization strategies."""
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=688,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    base_model = MicroLlama(config)
    base_model.eval()

    # Construct a realistic multi-request workload sharing a 64-token system prompt (80% of prompt)
    system_prefix = torch.randint(4, config.vocab_size, (system_prompt_len,)).tolist()
    prompts: list[list[int]] = []
    for req_idx in range(num_requests):
        torch.manual_seed(seed + 100 + req_idx)
        user_suffix = torch.randint(4, config.vocab_size, (user_query_len,)).tolist()
        prompts.append(system_prefix + user_suffix)

    fp32_weight_mb = float(model_weight_bytes(base_model)) / (1024.0 * 1024.0)
    int8_model, int8_rep = quantize_model(base_model, mode="int8")
    q4_model, q4_rep = quantize_model(base_model, mode="q4_0", group_size=32)

    configs_to_run = [
        ("1. Baseline (Naive torch.cat + FP32)", base_model, "naive", False, fp32_weight_mb),
        ("2. Static Ring Buffer KV (FP32)", base_model, "ring", False, fp32_weight_mb),
        ("3. vLLM PagedKV (block=16, FP32)", base_model, "paged", False, fp32_weight_mb),
        ("4. vLLM PagedKV + SGLang RadixCache", base_model, "paged", True, fp32_weight_mb),
        ("5. PagedKV + Radix + INT8 Weights", int8_model, "paged", True, int8_rep.quantized_mb),
        ("6. PagedKV + Radix + Q4_0 Weights", q4_model, "paged", True, q4_rep.quantized_mb),
    ]

    reference_logits: list[torch.Tensor] = []
    results: list[BenchmarkMetrics] = []

    for label, active_model, backend, use_radix, weight_mb in configs_to_run:
        engine = InferenceEngine(
            model=active_model,
            kv_backend=backend,
            use_radix_cache=use_radix,
            block_size=16,
            max_blocks=256,
            max_seq_len=512,
        )

        # Warmup pass (1 request) then reset
        _ = engine.generate(
            prompts[0][:32], max_new_tokens=4, seq_id=99, retain_kv_for_prefix_cache=False
        )
        engine.kv_cache.reset()
        if engine.radix_cache is not None:
            engine.radix_cache = RadixPrefixCache(
                block_manager=engine.kv_cache.block_manager,  # type: ignore[attr-defined]
                block_size=16,
            )

        ttfts: list[float] = []
        decode_rates: list[float] = []
        run_logits: list[torch.Tensor] = []
        peak_kv_bytes = 0
        frag_ratios: list[float] = []

        for req_idx, prompt in enumerate(prompts):
            if not isinstance(engine.kv_cache, PagedKVCache):
                engine.kv_cache.reset()

            out = engine.generate(
                prompt,
                max_new_tokens=decode_tokens,
                seq_id=req_idx,
                retain_kv_for_prefix_cache=use_radix,
            )
            ttfts.append(float(out["ttft_ms"]))  # type: ignore[arg-type]
            decode_rates.append(float(out["decode_tok_per_sec"]))  # type: ignore[arg-type]
            run_logits.append(out["prefill_last_logits"])  # type: ignore[arg-type]

            peak_kv_bytes = max(peak_kv_bytes, engine.kv_cache.allocated_bytes())
            frag_ratios.append(engine.kv_cache.fragmentation_ratio())

            if isinstance(engine.kv_cache, PagedKVCache):
                # Release request-specific physical blocks upon completion (while RadixPrefixCache
                # retains shared prefix blocks when use_radix=True)
                engine.kv_cache.free_sequence(req_idx)

        if not reference_logits:
            reference_logits = run_logits
            cos_sim = 1.0
        else:
            sims = [
                compute_logit_fidelity(ref_l.unsqueeze(0), cur_l.unsqueeze(0))["cosine_similarity"]
                for ref_l, cur_l in zip(reference_logits, run_logits, strict=True)
            ]
            cos_sim = sum(sims) / len(sims)

        hit_rate_pct = (
            engine.radix_cache.hit_rate * 100.0 if engine.radix_cache is not None else 0.0
        )
        results.append(
            BenchmarkMetrics(
                strategy_name=label,
                weight_mb=weight_mb,
                peak_kv_kb=float(peak_kv_bytes) / 1024.0,
                fragmentation_pct=(sum(frag_ratios) / len(frag_ratios)) * 100.0,
                ttft_ms=sum(ttfts) / len(ttfts),
                decode_tok_per_sec=sum(decode_rates) / len(decode_rates),
                prefix_hit_rate_pct=hit_rate_pct,
                logit_cosine_sim=cos_sim,
            )
        )

    return results
