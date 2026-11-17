"""Tests for per-channel INT8 and llama.cpp-style Q4_0 group-wise 4-bit block quantization."""

from __future__ import annotations

import torch
from torch import nn

from kv_infer_opt import (
    LlamaConfig,
    MicroLlama,
    QuantizedLinearInt4Block,
    QuantizedLinearInt8,
    compute_logit_fidelity,
    quantize_model,
)


class TestWeightQuantization:
    """Verify INT8 and Q4_0 byte packing, memory compression, and logit cosine similarity."""

    def test_quantized_linear_int8_reconstruction_and_compression(self) -> None:
        torch.manual_seed(42)
        fc = nn.Linear(128, 64, bias=True)
        q_fc = QuantizedLinearInt8.from_linear(fc)

        fp32_bytes = (fc.weight.numel() + fc.bias.numel()) * 4
        int8_bytes = q_fc.storage_bytes()
        assert int8_bytes < fp32_bytes * 0.28

        x = torch.randn(8, 128)
        with torch.no_grad():
            y_ref = fc(x)
            y_quant = q_fc(x)
        cos_sim = torch.nn.functional.cosine_similarity(y_ref, y_quant, dim=-1).mean().item()
        assert cos_sim > 0.999

    def test_quantized_linear_int4_block_packing_and_compression(self) -> None:
        torch.manual_seed(42)
        fc = nn.Linear(256, 128, bias=False)
        q4_fc = QuantizedLinearInt4Block.from_linear(fc, group_size=32)

        fp32_bytes = fc.weight.numel() * 4
        q4_bytes = q4_fc.storage_bytes()
        # 4-bit packed weights + 16-bit scale per 32 weights = 4.5 bits/weight (~7.1x compression)
        compression = float(fp32_bytes) / float(q4_bytes)
        assert compression > 6.5

        x = torch.randn(8, 256)
        with torch.no_grad():
            y_ref = fc(x)
            y_q4 = q4_fc(x)
        cos_sim = torch.nn.functional.cosine_similarity(y_ref, y_q4, dim=-1).mean().item()
        assert cos_sim > 0.985

    def test_full_model_quantization_and_logit_fidelity(self) -> None:
        torch.manual_seed(123)
        cfg = LlamaConfig(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=320,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        base_model = MicroLlama(cfg)
        base_model.eval()

        int8_model, rep_int8 = quantize_model(base_model, mode="int8")
        q4_model, rep_q4 = quantize_model(base_model, mode="q4_0", group_size=32)

        assert rep_int8.compression_ratio > 2.5
        assert rep_q4.compression_ratio > 3.5

        inp = torch.randint(4, 250, (2, 16), dtype=torch.long)
        with torch.no_grad():
            ref_logits = base_model(inp)
            int8_logits = int8_model(inp)
            q4_logits = q4_model(inp)

        fid_int8 = compute_logit_fidelity(ref_logits, int8_logits)
        fid_q4 = compute_logit_fidelity(ref_logits, q4_logits)

        assert fid_int8["cosine_similarity"] > 0.998
        assert fid_q4["cosine_similarity"] > 0.96

    def test_zero_weight_tensor_and_float16_input_safety(self) -> None:
        # 1. All-zero linear weight matrix must not trigger ZeroDivisionError or NaN
        fc_zero = nn.Linear(64, 32, bias=True)
        with torch.no_grad():
            fc_zero.weight.zero_()
            fc_zero.bias.zero_()

        q8_zero = QuantizedLinearInt8.from_linear(fc_zero)
        q4_zero = QuantizedLinearInt4Block.from_linear(fc_zero, group_size=32)

        x_f32 = torch.randn(4, 64, dtype=torch.float32)
        x_f16 = x_f32.to(torch.float16)

        with torch.no_grad():
            out8_f32 = q8_zero(x_f32)
            out4_f32 = q4_zero(x_f32)
            out8_f16 = q8_zero(x_f16)
            out4_f16 = q4_zero(x_f16)

        assert torch.all(out8_f32 == 0.0) and out8_f32.dtype == torch.float32
        assert torch.all(out4_f32 == 0.0) and out4_f32.dtype == torch.float32
        assert torch.all(out8_f16 == 0.0) and out8_f16.dtype == torch.float16
        assert torch.all(out4_f16 == 0.0) and out4_f16.dtype == torch.float16

        # 2. Verify in-place quantize_model preserves device and replaces all Linear layers
        small_net = nn.Sequential(
            nn.Linear(32, 32, bias=False), nn.ReLU(), nn.Linear(32, 16, bias=True)
        )
        _, rep = quantize_model(small_net, mode="q4_0", group_size=16, inplace=True)
        assert rep.layers_replaced == 2
        assert isinstance(small_net[0], QuantizedLinearInt4Block)
        assert isinstance(small_net[2], QuantizedLinearInt4Block)
