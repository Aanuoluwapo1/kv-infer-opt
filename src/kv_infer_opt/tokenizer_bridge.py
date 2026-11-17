"""Hugging Face `tokenizers` and `transformers` bridge with offline-safe BPE initialization."""

from __future__ import annotations

from typing import Any

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import AutoTokenizer
from transformers import LlamaConfig as HFLlamaConfig

from kv_infer_opt.model import LlamaConfig


class HFTokenizerAdapter:
    """Fast Rust-backed BPE tokenizer powered by Hugging Face `tokenizers` / `transformers`.

    Provides a deterministic offline BPE vocabulary constructor (`build_local_bpe`) so CI and
    benchmark suites run with zero external network calls, while also supporting `from_pretrained`
    via `transformers.AutoTokenizer`.
    """

    def __init__(self, backend: Tokenizer | Any, vocab_size: int) -> None:
        self._backend = backend
        self.vocab_size = int(vocab_size)
        self.bos_token_id: int = 1
        self.eos_token_id: int = 2
        self.pad_token_id: int = 0

    @classmethod
    def build_local_bpe(cls, vocab_size: int = 1024) -> HFTokenizerAdapter:
        """Build a deterministic in-memory Hugging Face `tokenizers.Tokenizer(BPE)` instance."""
        if vocab_size < 320:
            raise ValueError(f"vocab_size must be >= 320, got {vocab_size}")

        vocab: dict[str, int] = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
        }
        next_id = 4

        # Base printable ASCII + ByteLevel alphabet characters
        alphabet = ByteLevel.alphabet()
        for ch in alphabet:
            if ch not in vocab and next_id < vocab_size:
                vocab[ch] = next_id
                next_id += 1

        # Common English subword & domain pairs for deterministic multi-character merges
        common_pairs = [
            ("Ġ", "t"),
            ("Ġt", "he"),
            ("h", "e"),
            ("i", "n"),
            ("e", "r"),
            ("a", "n"),
            ("r", "e"),
            ("o", "n"),
            ("a", "t"),
            ("e", "n"),
            ("Ġ", "a"),
            ("Ġ", "s"),
            ("Ġ", "p"),
            ("Ġ", "c"),
            ("s", "y"),
            ("t", "e"),
            ("m", "p"),
            ("r", "o"),
            ("k", "v"),
            ("c", "a"),
        ]
        merges: list[tuple[str, str]] = []
        for left, right in common_pairs:
            merged = left + right
            if left in vocab and right in vocab and merged not in vocab and next_id < vocab_size:
                vocab[merged] = next_id
                next_id += 1
                merges.append((left, right))

        # Fill remaining vocabulary slots up to vocab_size
        idx = 0
        while next_id < vocab_size:
            tok = f"<tok_{idx}>"
            if tok not in vocab:
                vocab[tok] = next_id
                next_id += 1
            idx += 1

        tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        return cls(backend=tokenizer, vocab_size=vocab_size)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> HFTokenizerAdapter:
        """Load a Hugging Face `AutoTokenizer` by repository ID or local directory."""
        hf_tok = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(backend=hf_tok, vocab_size=len(hf_tok))

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        """Encode string `text` into a list of integer token IDs."""
        if isinstance(self._backend, Tokenizer):
            ids = list(self._backend.encode(text).ids)
        else:
            ids = list(self._backend.encode(text, add_special_tokens=False))
        if add_bos:
            return [self.bos_token_id, *ids]
        return ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode integer `token_ids` back into text."""
        filtered = [
            int(t)
            for t in token_ids
            if int(t) not in (self.pad_token_id, self.bos_token_id, self.eos_token_id)
        ]
        if isinstance(self._backend, Tokenizer):
            return str(self._backend.decode(filtered))
        return str(self._backend.decode(filtered, skip_special_tokens=True))

    @staticmethod
    def to_hf_llama_config(config: LlamaConfig) -> HFLlamaConfig:
        """Convert our `LlamaConfig` into an official `transformers.LlamaConfig` instance."""
        return HFLlamaConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
        )
