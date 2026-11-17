"""End-to-end tests for HFTokenizerAdapter, MicroLlama, InferenceEngine, and CLI."""

from __future__ import annotations

import torch

from kv_infer_opt import (
    HFTokenizerAdapter,
    InferenceEngine,
    LlamaConfig,
    MicroLlama,
)
from kv_infer_opt.cli import main as cli_main


class TestEndToEndPipelineAndCLI:
    """Verify Hugging Face tokenizer bridge, Radix + PagedKV generation parity, and CLI commands."""

    def test_hf_tokenizer_adapter_roundtrip_and_config_bridge(self) -> None:
        tok = HFTokenizerAdapter.build_local_bpe(vocab_size=512)
        text = "PagedAttention and RadixAttention accelerate Transformer decoding."
        ids = tok.encode(text, add_bos=True)
        assert ids[0] == tok.bos_token_id
        assert len(ids) > 5
        decoded = tok.decode(ids)
        assert "Attention" in decoded

        cfg = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=128)
        hf_cfg = HFTokenizerAdapter.to_hf_llama_config(cfg)
        roundtrip_cfg = LlamaConfig.from_hf_config(hf_cfg)
        assert roundtrip_cfg == cfg

    def test_inference_engine_radix_prefix_reuse_exact_generation_parity(self) -> None:
        torch.manual_seed(42)
        cfg = LlamaConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        model = MicroLlama(cfg)

        engine_no_radix = InferenceEngine(
            model=model, kv_backend="paged", use_radix_cache=False, block_size=8
        )
        engine_radix = InferenceEngine(
            model=model, kv_backend="paged", use_radix_cache=True, block_size=8
        )

        shared_prefix = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        prompt_1 = shared_prefix + [30, 31, 32, 33]
        prompt_2 = shared_prefix + [40, 41, 42, 43]

        # Warm up RadixPrefixCache with prompt_1
        _ = engine_radix.generate(prompt_1, max_new_tokens=6, seq_id=0)
        engine_radix.kv_cache.free_sequence(0)  # type: ignore[attr-defined]

        # Generate for prompt_2 with and without RadixPrefixCache
        out_plain = engine_no_radix.generate(prompt_2, max_new_tokens=6, seq_id=1)
        out_cached = engine_radix.generate(prompt_2, max_new_tokens=6, seq_id=1)

        assert out_cached["matched_prefix_tokens"] == 16
        assert out_plain["generated_ids"] == out_cached["generated_ids"]
        torch.testing.assert_close(
            out_plain["prefill_last_logits"],
            out_cached["prefill_last_logits"],
            atol=1e-5,
            rtol=1e-5,
        )

    def test_cli_subcommands_execute_cleanly(self) -> None:
        rc_radix = cli_main(["inspect-radix", "--seed", "42"])
        assert rc_radix == 0

        rc_bench = cli_main(
            [
                "benchmark",
                "--requests",
                "3",
                "--system-len",
                "32",
                "--query-len",
                "8",
                "--decode-tokens",
                "4",
            ]
        )
        assert rc_bench == 0
