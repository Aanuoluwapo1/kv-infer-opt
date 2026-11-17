"""kv-infer-opt: High-Throughput Transformer Inference & KV-Cache Optimization Engine."""

from kv_infer_opt.engine import BenchmarkMetrics, InferenceEngine, run_comparative_benchmark
from kv_infer_opt.model import LlamaConfig, MicroLlama
from kv_infer_opt.paged_kv import (
    BlockManager,
    KVCacheProtocol,
    NaiveKVCache,
    PagedKVCache,
    StaticRingKVCache,
)
from kv_infer_opt.quantization import (
    QuantizationReport,
    QuantizedLinearInt4Block,
    QuantizedLinearInt8,
    compute_logit_fidelity,
    model_weight_bytes,
    quantize_model,
)
from kv_infer_opt.radix_cache import PrefixMatchResult, RadixNode, RadixPrefixCache
from kv_infer_opt.tokenizer_bridge import HFTokenizerAdapter

__version__ = "0.1.0"

__all__ = [
    "BenchmarkMetrics",
    "BlockManager",
    "HFTokenizerAdapter",
    "InferenceEngine",
    "KVCacheProtocol",
    "LlamaConfig",
    "MicroLlama",
    "NaiveKVCache",
    "PagedKVCache",
    "PrefixMatchResult",
    "QuantizationReport",
    "QuantizedLinearInt4Block",
    "QuantizedLinearInt8",
    "RadixNode",
    "RadixPrefixCache",
    "StaticRingKVCache",
    "compute_logit_fidelity",
    "model_weight_bytes",
    "quantize_model",
    "run_comparative_benchmark",
]
