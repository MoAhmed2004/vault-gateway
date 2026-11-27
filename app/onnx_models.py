"""Local ONNX model wrappers: MiniLM embedder (L2, X3) and DeBERTa classifier (L3) (IP-2, IP-7, §2.7, §5.3).

Both models run on CPU through onnxruntime with `intra_op_num_threads=1` and
`inter_op_num_threads=1` [Project Decision, extends IP-7: keeps per-call CPU
use predictable inside the bounded layer pool], sequential execution, and a
Hugging Face `tokenizers` fast tokenizer loaded from `tokenizer.json`.

Never truncate (IP-2): truncation and padding are switched off on both
tokenizers, `count_tokens` counts the special tokens the model adds, and a
text above the model's token limit raises `ValueError` instead of being cut.
The InspectionBundle splits windows to fit before any layer sees them, so a
layer never hits that error on bundle windows.

MiniLM long words (Step 5B.1 M-3, IP-2, §3.6): the `all-MiniLM-L6-v2`
WordPiece tokenizer ships with `max_input_chars_per_word = 100`, so any
spaceless run longer than 100 characters became one `[UNK]` token and was
opaque to L2. `MiniLMEmbedder` raises the limit to
`MINILM_MAX_INPUT_CHARS_PER_WORD` (600, one full inspection window) through
the `tokenizers` WordPiece model attribute and reads it back, so the setting
is verified to take effect; such runs are now split into word pieces.
`unk_ratio(text)` reports the share of `[UNK]` among the non-special tokens,
which the L2 hardening path uses to refuse degenerate windows.

Versions: each wrapper's `version` is a SHA-256 over the wrapper revision, the
SHA-256 of `model.onnx` and `tokenizer.json`, the token limit, the tokenizer
overrides (MiniLM `max_input_chars_per_word`) and the scoring rule, so
swapping any artifact (for example the Colab fine-tuned L3 export) or a
tokenizer setting changes the layer version and with it the IP-5
layer-verdict key, and a ChromaDB collection built with the old embedder is
refused until it is rebuilt. The artifact digests are also what
`docs/artifacts.md` records (§5.3).

Artifacts are fetched by `scripts/fetch_models.py` into `models/` (git-ignored).
Loading fails fast with the missing path named.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from app.inspection import DEBERTA_TOKEN_LIMIT, MINILM_TOKEN_LIMIT, TokenizerLimit
from contracts.hashing import canonical_hash

EMBEDDER_REVISION = 2
CLASSIFIER_REVISION = 1
MINILM_TOKENIZER_NAME = "minilm"
DEBERTA_TOKENIZER_NAME = "deberta"
MINILM_DIMENSION = 384
MINILM_MAX_INPUT_CHARS_PER_WORD = 600
DEFAULT_MALICIOUS_LABEL = "INJECTION"
_CHUNK_BYTES = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"model artifact missing: {path} (run scripts/fetch_models.py)")
    return path


def _session(model_path: Path):  # noqa: ANN202  onnxruntime has no type stubs
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])


def load_tokenizer(path: Path | str, *, max_input_chars_per_word: int | None = None):  # noqa: ANN201
    """A `tokenizers` fast tokenizer with truncation and padding off (IP-2).

    With `max_input_chars_per_word`, the WordPiece model's per-word character
    limit is set and read back; a model without that attribute (not WordPiece)
    or a read-back mismatch raises, so the override can never silently fail.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    if max_input_chars_per_word is not None:
        if max_input_chars_per_word < 1:
            raise ValueError("max_input_chars_per_word must be >= 1")
        model = tokenizer.model
        if not hasattr(model, "max_input_chars_per_word"):
            raise ValueError(f"{type(model).__name__} tokenizer has no max_input_chars_per_word setting")
        model.max_input_chars_per_word = max_input_chars_per_word
        if tokenizer.model.max_input_chars_per_word != max_input_chars_per_word:
            raise RuntimeError("tokenizers did not apply max_input_chars_per_word")
    return tokenizer


@runtime_checkable
class Embedder(Protocol):
    """Sentence embedder used by L2 and the X3 similarity leak check."""

    name: str
    version: str
    token_limit: int

    def count_tokens(self, text: str) -> int: ...

    def embed(self, text: str) -> np.ndarray: ...


@runtime_checkable
class Classifier(Protocol):
    """Binary prompt-injection classifier used by L3."""

    name: str
    version: str
    token_limit: int

    def count_tokens(self, text: str) -> int: ...

    def malicious_probability(self, text: str) -> float: ...


def tokenizer_limit_for(model: Embedder | Classifier) -> TokenizerLimit:
    """The IP-2 sub-window splitting limit this model registers with the InspectionBundle."""
    return TokenizerLimit(name=model.name, token_limit=model.token_limit, count_tokens=model.count_tokens)


class MiniLMEmbedder:
    """`all-MiniLM-L6-v2` ONNX: mean pooling over the attention mask, then L2 normalization.

    This is the sentence-transformers pipeline for this model (Transformer,
    Pooling mean, Normalize), so cosine similarity is a dot product.
    """

    name = MINILM_TOKENIZER_NAME

    def __init__(
        self,
        model_dir: Path | str,
        *,
        token_limit: int = MINILM_TOKEN_LIMIT,
        max_input_chars_per_word: int = MINILM_MAX_INPUT_CHARS_PER_WORD,
    ) -> None:
        directory = Path(model_dir)
        model_path = _require(directory / "model.onnx")
        tokenizer_path = _require(directory / "tokenizer.json")
        self.token_limit = token_limit
        self.model_sha256 = sha256_file(model_path)
        self.tokenizer_sha256 = sha256_file(tokenizer_path)
        self._tokenizer = load_tokenizer(tokenizer_path, max_input_chars_per_word=max_input_chars_per_word)
        self.max_input_chars_per_word = int(self._tokenizer.model.max_input_chars_per_word)
        unk_token = str(getattr(self._tokenizer.model, "unk_token", "[UNK]"))
        unk_id = self._tokenizer.token_to_id(unk_token)
        if unk_id is None:
            raise ValueError(f"MiniLM tokenizer has no {unk_token!r} token")
        self.unk_id = int(unk_id)
        self._session = _session(model_path)
        self._inputs = {i.name for i in self._session.get_inputs()}
        self.version = canonical_hash(
            {
                "component": "MiniLMEmbedder",
                "revision": EMBEDDER_REVISION,
                "model_sha256": self.model_sha256,
                "tokenizer_sha256": self.tokenizer_sha256,
                "token_limit": token_limit,
                "max_input_chars_per_word": self.max_input_chars_per_word,
                "pooling": "mean",
                "normalize": True,
            }
        )
        self.dimension = int(self.embed("warm-up").shape[0])

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text).ids)

    def unk_ratio(self, text: str) -> float:
        """Share of `[UNK]` among the non-special tokens of `text` (0.0 when there are none)."""
        encoding = self._tokenizer.encode(text)
        content = [i for i, special in zip(encoding.ids, encoding.special_tokens_mask, strict=True) if not special]
        if not content:
            return 0.0
        return sum(1 for i in content if i == self.unk_id) / len(content)

    def embed(self, text: str) -> np.ndarray:
        encoding = self._tokenizer.encode(text)
        if len(encoding.ids) > self.token_limit:
            raise ValueError(f"{len(encoding.ids)} tokens above the {self.token_limit}-token MiniLM limit (IP-2: split first)")
        ids = np.asarray([encoding.ids], dtype=np.int64)
        mask = np.asarray([encoding.attention_mask], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.asarray([encoding.type_ids], dtype=np.int64)
        hidden = self._session.run(None, feeds)[0][0].astype(np.float64)
        weights = mask[0].astype(np.float64)[:, None]
        pooled = (hidden * weights).sum(axis=0) / max(weights.sum(), 1e-9)
        norm = float(np.linalg.norm(pooled))
        return (pooled / norm if norm > 0 else pooled).astype(np.float32)

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.embed(text) for text in texts]) if texts else np.zeros((0, self.dimension), np.float32)


class DebertaClassifier:
    """DeBERTa-v3 sequence classifier ONNX export; softmax probability of the malicious label.

    The label index is read from the export's `config.json` (`id2label`), so a
    fine-tuned export with a different label order still scores the right class.
    """

    name = DEBERTA_TOKENIZER_NAME

    def __init__(
        self,
        model_dir: Path | str,
        *,
        token_limit: int = DEBERTA_TOKEN_LIMIT,
        malicious_label: str = DEFAULT_MALICIOUS_LABEL,
    ) -> None:
        directory = Path(model_dir)
        model_path = _require(directory / "model.onnx")
        tokenizer_path = _require(directory / "tokenizer.json")
        config_path = _require(directory / "config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        labels = {str(v): int(k) for k, v in config.get("id2label", {}).items()}
        if malicious_label not in labels:
            raise ValueError(f"config.json id2label has no {malicious_label!r} label: {sorted(labels)}")
        self.label_index = labels[malicious_label]
        self.num_labels = len(labels)
        self.token_limit = token_limit
        self.model_sha256 = sha256_file(model_path)
        self.tokenizer_sha256 = sha256_file(tokenizer_path)
        self.config_sha256 = sha256_file(config_path)
        self._tokenizer = load_tokenizer(tokenizer_path)
        self._session = _session(model_path)
        self._inputs = {i.name for i in self._session.get_inputs()}
        self.version = canonical_hash(
            {
                "component": "DebertaClassifier",
                "revision": CLASSIFIER_REVISION,
                "model_sha256": self.model_sha256,
                "tokenizer_sha256": self.tokenizer_sha256,
                "config_sha256": self.config_sha256,
                "token_limit": token_limit,
                "malicious_label": malicious_label,
                "score": "softmax",
            }
        )
        self.malicious_probability("warm-up")

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text).ids)

    def malicious_probability(self, text: str) -> float:
        encoding = self._tokenizer.encode(text)
        if len(encoding.ids) > self.token_limit:
            raise ValueError(f"{len(encoding.ids)} tokens above the {self.token_limit}-token DeBERTa limit (IP-2: split first)")
        feeds = {
            "input_ids": np.asarray([encoding.ids], dtype=np.int64),
            "attention_mask": np.asarray([encoding.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.asarray([encoding.type_ids], dtype=np.int64)
        logits = self._session.run(None, feeds)[0][0].astype(np.float64)
        if logits.shape[0] != self.num_labels:
            raise ValueError(f"model returned {logits.shape[0]} logits, config.json names {self.num_labels} labels")
        shifted = np.exp(logits - logits.max())
        return float(shifted[self.label_index] / shifted.sum())
