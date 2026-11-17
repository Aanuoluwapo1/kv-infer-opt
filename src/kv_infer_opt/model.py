"""Llama-3 style Decoder-Only Transformer with RoPE, Grouped-Query Attention (GQA), and SwiGLU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from kv_infer_opt.paged_kv import KVCacheProtocol


@dataclass
class LlamaConfig:
    """Configuration for `MicroLlama` with Hugging Face `PretrainedConfig` interoperability."""

    vocab_size: int = 1024
    hidden_size: int = 256
    intermediate_size: int = 688
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> LlamaConfig:
        """Construct a `LlamaConfig` from a `transformers.LlamaConfig` or compatible object."""
        return cls(
            vocab_size=int(getattr(hf_config, "vocab_size", 1024)),
            hidden_size=int(getattr(hf_config, "hidden_size", 256)),
            intermediate_size=int(getattr(hf_config, "intermediate_size", 688)),
            num_hidden_layers=int(getattr(hf_config, "num_hidden_layers", 4)),
            num_attention_heads=int(getattr(hf_config, "num_attention_heads", 8)),
            num_key_value_heads=int(
                getattr(
                    hf_config,
                    "num_key_value_heads",
                    getattr(hf_config, "num_attention_heads", 8),
                )
            ),
            max_position_embeddings=int(getattr(hf_config, "max_position_embeddings", 512)),
            rms_norm_eps=float(getattr(hf_config, "rms_norm_eps", 1e-5)),
            rope_theta=float(getattr(hf_config, "rope_theta", 10000.0)),
        )


class LlamaRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (`RMSNorm`)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm.to(orig_dtype)


def precompute_rope_cos_sin(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute Rotary Position Embedding (`RoPE`) cosine and sine frequency tables."""
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
    half_dim = head_dim // 2
    exponents = torch.arange(0, half_dim, dtype=torch.float32, device=device) / float(half_dim)
    inv_freq = 1.0 / (theta**exponents)
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(positions, inv_freq)  # (max_seq_len, half_dim)
    emb = torch.cat([angles, angles], dim=-1)  # (max_seq_len, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embeddings to `q` `(B, H_q, T, D)` and `k` `(B, H_kv, T, D)`."""
    seq_len = q.shape[2]
    cos_slice = cos[position_offset : position_offset + seq_len].unsqueeze(0).unsqueeze(0)
    sin_slice = sin[position_offset : position_offset + seq_len].unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos_slice) + (_rotate_half(q) * sin_slice)
    k_rot = (k * cos_slice) + (_rotate_half(k) * sin_slice)
    return q_rot, k_rot


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) supporting pluggable KV-Cache backends."""

    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_kv_heads})"
            )
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCacheProtocol | None = None,
        position_offset: int = 0,
        seq_ids: list[int] | None = None,
    ) -> torch.Tensor:
        bsz, t_new, _ = x.shape
        q = self.q_proj(x).view(bsz, t_new, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, t_new, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, t_new, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin, position_offset=position_offset)

        if kv_cache is not None:
            k_full, v_full = kv_cache.append(self.layer_idx, k, v, seq_ids=seq_ids)
        else:
            k_full, v_full = k, v

        # Expand KV heads to match Query heads when using Grouped-Query Attention
        if self.num_queries_per_kv > 1:
            k_attn = k_full.repeat_interleave(self.num_queries_per_kv, dim=1)
            v_attn = v_full.repeat_interleave(self.num_queries_per_kv, dim=1)
        else:
            k_attn, v_attn = k_full, v_full

        t_total = k_attn.shape[2]
        if t_new > 1:
            # Causal mask aligned with the bottom-right corner of (t_new, t_total)
            row_idx = torch.arange(t_new, device=x.device).unsqueeze(1) + (t_total - t_new)
            col_idx = torch.arange(t_total, device=x.device).unsqueeze(0)
            causal_mask = col_idx <= row_idx
            attn_out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=causal_mask)
        else:
            # Single-token decode step attends to all cached prefix positions
            attn_out = F.scaled_dot_product_attention(q, k_attn, v_attn, is_causal=False)

        merged = attn_out.transpose(1, 2).contiguous().view(bsz, t_new, self.hidden_size)
        return self.o_proj(merged)


class SwiGLUMLP(nn.Module):
    """Llama-style SwiGLU Feed-Forward Network (`down_proj(silu(gate_proj(x)) * up_proj(x))`)."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    """Single Pre-Norm Decoder Layer combining GQA and SwiGLU MLP."""

    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCacheProtocol | None = None,
        position_offset: int = 0,
        seq_ids: list[int] | None = None,
    ) -> torch.Tensor:
        h = x + self.self_attn(
            self.input_layernorm(x),
            cos=cos,
            sin=sin,
            kv_cache=kv_cache,
            position_offset=position_offset,
            seq_ids=seq_ids,
        )
        return h + self.mlp(self.post_attention_layernorm(h))


class MicroLlama(nn.Module):
    """Llama-3 style Decoder-Only Transformer supporting pluggable KV-Cache and Quantization."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        cos, sin = precompute_rope_cos_sin(
            head_dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCacheProtocol | None = None,
        position_offset: int | None = None,
        seq_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """Compute vocabulary logits `(B, T, vocab_size)` for `input_ids` `(B, T)`."""
        if position_offset is None:
            if kv_cache is not None:
                first_sid = seq_ids[0] if seq_ids else 0
                pos_off = kv_cache.get_seq_len(first_sid)
            else:
                pos_off = 0
        else:
            pos_off = int(position_offset)

        h = self.embed_tokens(input_ids)
        cos = self.rope_cos.to(h.device)
        sin = self.rope_sin.to(h.device)

        for layer in self.layers:
            h = layer(
                h,
                cos=cos,
                sin=sin,
                kv_cache=kv_cache,
                position_offset=pos_off,
                seq_ids=seq_ids,
            )

        h = self.norm(h)
        return self.lm_head(h)
