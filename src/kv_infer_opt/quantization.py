"""llama.cpp / GGUF-style Q4_0 block quantization and symmetric per-channel INT8 quantization."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class QuantizationReport:
    """Compression and numerical fidelity summary for a quantized model."""

    mode: str
    fp32_bytes: int
    quantized_bytes: int
    compression_ratio: float
    layers_replaced: int

    @property
    def fp32_mb(self) -> float:
        return float(self.fp32_bytes) / (1024.0 * 1024.0)

    @property
    def quantized_mb(self) -> float:
        return float(self.quantized_bytes) / (1024.0 * 1024.0)


class QuantizedLinearInt8(nn.Module):
    """Symmetric per-output-channel 8-bit weight-only quantized linear layer (`W8A32`).

    Stores:
    - `qweight`: `int8` tensor of shape `(out_features, in_features)`
    - `scales`: `float32` tensor of shape `(out_features, 1)` where `scale = max(|W_row|) / 127.0`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.register_buffer(
            "qweight",
            torch.zeros((self.out_features, self.in_features), dtype=torch.int8),
        )
        self.register_buffer(
            "scales",
            torch.ones((self.out_features, 1), dtype=torch.float32),
        )
        if bias:
            self.register_buffer("bias", torch.zeros(self.out_features, dtype=torch.float32))
        else:
            self.bias = None

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear) -> QuantizedLinearInt8:
        device = linear.weight.device
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
        ).to(device)
        w = linear.weight.detach().to(torch.float32)
        max_abs = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scales = max_abs / 127.0
        qweight = torch.clamp(torch.round(w / scales), -127, 127).to(torch.int8)

        layer.qweight.copy_(qweight)
        layer.scales.copy_(scales)
        if linear.bias is not None and layer.bias is not None:
            layer.bias.copy_(linear.bias.detach().to(torch.float32))
        return layer

    def dequantize_weight(self) -> torch.Tensor:
        """Reconstruct `float32` weight matrix `(out_features, in_features)`."""
        return self.qweight.to(torch.float32) * self.scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_deq = self.dequantize_weight().to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_deq, b)

    def storage_bytes(self) -> int:
        total = self.qweight.numel() * self.qweight.element_size()
        total += self.scales.numel() * self.scales.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total


class QuantizedLinearInt4Block(nn.Module):
    """llama.cpp `Q4_0`-inspired group-wise 4-bit block quantized linear layer.

    Partitions the flattened weight tensor into fixed blocks of `group_size` (default `32`),
    computes a `float16` scale per block (`scale = max(|block|) / 7.0`), maps signed 4-bit
    values in `[-8, 7]` to unsigned nibbles `[0, 15]` via `+8` offset, and packs two 4-bit
    nibbles into each `uint8` byte (`low_nibble | (high_nibble << 4)`).

    Achieves ~6.4x compression on the linear weight tensor (4 bits + 0.5 bits scale per weight).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 32,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if group_size <= 0 or group_size % 2 != 0:
            raise ValueError(f"group_size must be a positive even integer, got {group_size}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)

        total_elems = self.out_features * self.in_features
        self.padded_elems = (
            (total_elems + self.group_size - 1) // self.group_size
        ) * self.group_size
        self.num_groups = self.padded_elems // self.group_size

        self.register_buffer(
            "packed_weight",
            torch.zeros((self.num_groups, self.group_size // 2), dtype=torch.uint8),
        )
        self.register_buffer(
            "scales",
            torch.ones((self.num_groups, 1), dtype=torch.float16),
        )
        if bias:
            self.register_buffer("bias", torch.zeros(self.out_features, dtype=torch.float32))
        else:
            self.bias = None

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear, group_size: int = 32) -> QuantizedLinearInt4Block:
        device = linear.weight.device
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            group_size=group_size,
            bias=linear.bias is not None,
        ).to(device)
        w_flat = linear.weight.detach().to(torch.float32).reshape(-1)
        if w_flat.numel() < layer.padded_elems:
            w_flat = F.pad(w_flat, (0, layer.padded_elems - w_flat.numel()))

        groups = w_flat.view(layer.num_groups, layer.group_size)
        max_abs = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scales_f16 = (max_abs / 7.0).to(torch.float16)
        scales_f32 = scales_f16.to(torch.float32).clamp(min=1e-8)

        # Quantize to signed [-8, 7], then shift by +8 into unsigned nibble [0, 15]
        q_signed = torch.clamp(torch.round(groups / scales_f32), -8, 7).to(torch.int16)
        q_unsigned = (q_signed + 8).to(torch.uint8)

        low_nibbles = q_unsigned[:, 0::2]
        high_nibbles = q_unsigned[:, 1::2]
        packed = low_nibbles | (high_nibbles << 4)

        layer.packed_weight.copy_(packed)
        layer.scales.copy_(scales_f16)
        if linear.bias is not None and layer.bias is not None:
            layer.bias.copy_(linear.bias.detach().to(torch.float32))
        return layer

    def dequantize_weight(self) -> torch.Tensor:
        """Unpack 4-bit nibbles and dequantize back to `(out_features, in_features)` in `float32`."""
        low = (self.packed_weight & 0x0F).to(torch.int16) - 8
        high = ((self.packed_weight >> 4) & 0x0F).to(torch.int16) - 8

        unpacked = torch.empty(
            (self.num_groups, self.group_size),
            dtype=torch.float32,
            device=self.packed_weight.device,
        )
        unpacked[:, 0::2] = low.to(torch.float32)
        unpacked[:, 1::2] = high.to(torch.float32)

        deq_flat = (unpacked * self.scales.to(torch.float32)).reshape(-1)
        total_elems = self.out_features * self.in_features
        return deq_flat[:total_elems].view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_deq = self.dequantize_weight().to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_deq, b)

    def storage_bytes(self) -> int:
        total = self.packed_weight.numel() * self.packed_weight.element_size()
        total += self.scales.numel() * self.scales.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total


def model_weight_bytes(model: nn.Module, include_embeddings: bool = True) -> int:
    """Compute total memory footprint in bytes of all parameters and quantization buffers."""
    total = 0
    for name, param in model.named_parameters():
        if not include_embeddings and ("embed_tokens" in name or "lm_head" in name):
            continue
        total += param.numel() * param.element_size()
    for name, buf in model.named_buffers():
        if name in ("rope_cos", "rope_sin"):
            continue
        if not include_embeddings and ("embed_tokens" in name or "lm_head" in name):
            continue
        total += buf.numel() * buf.element_size()
    return total


def quantize_model(
    model: nn.Module,
    mode: str = "int8",
    group_size: int = 32,
    quantize_lm_head: bool = True,
    inplace: bool = False,
) -> tuple[nn.Module, QuantizationReport]:
    """Replace `nn.Linear` layers in `model` with `QuantizedLinearInt8` or `QuantizedLinearInt4Block`.

    Args:
        model: Target PyTorch module (`MicroLlama` or arbitrary `nn.Module`).
        mode: Either `"int8"` (per-channel W8A32) or `"q4_0"` (group-wise 4-bit block quantization).
        group_size: Block size for `"q4_0"` mode.
        quantize_lm_head: Whether to also quantize the output `lm_head` projection.
        inplace: If `False` (default), clones `model` before replacing layers.

    Returns:
        `(quantized_model, QuantizationReport)`
    """
    if mode not in ("int8", "q4_0"):
        raise ValueError(f"Unsupported quantization mode: {mode!r}. Expected 'int8' or 'q4_0'.")

    target = model if inplace else copy.deepcopy(model)
    fp32_bytes = model_weight_bytes(target, include_embeddings=True)
    replaced = 0

    def _replace_recursive(parent: nn.Module, prefix: str = "") -> None:
        nonlocal replaced
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                if not quantize_lm_head and child_name == "lm_head":
                    continue
                if mode == "int8":
                    q_layer: nn.Module = QuantizedLinearInt8.from_linear(child)
                else:
                    q_layer = QuantizedLinearInt4Block.from_linear(child, group_size=group_size)
                setattr(parent, child_name, q_layer)
                replaced += 1
            else:
                _replace_recursive(child, full_name)

    _replace_recursive(target)
    quant_bytes = model_weight_bytes(target, include_embeddings=True)
    ratio = float(fp32_bytes) / float(max(quant_bytes, 1))
    report = QuantizationReport(
        mode=mode,
        fp32_bytes=fp32_bytes,
        quantized_bytes=quant_bytes,
        compression_ratio=ratio,
        layers_replaced=replaced,
    )
    return target, report


@torch.no_grad()
def compute_logit_fidelity(
    ref_logits: torch.Tensor,
    quant_logits: torch.Tensor,
) -> dict[str, float]:
    """Compare quantized logits against FP32 reference logits (cosine similarity, max drift, KL)."""
    ref_flat = ref_logits.detach().to(torch.float32).reshape(-1, ref_logits.shape[-1])
    q_flat = quant_logits.detach().to(torch.float32).reshape(-1, quant_logits.shape[-1])

    cos_sim = float(F.cosine_similarity(ref_flat, q_flat, dim=-1).mean().item())
    max_abs_diff = float((ref_flat - q_flat).abs().max().item())

    p_ref = F.softmax(ref_flat, dim=-1).clamp(min=1e-12)
    log_q = F.log_softmax(q_flat, dim=-1)
    kl_div = float(F.kl_div(log_q, p_ref, reduction="batchmean").item())

    return {
        "cosine_similarity": cos_sim,
        "max_abs_diff": max_abs_diff,
        "kl_divergence": max(0.0, kl_div),
    }
