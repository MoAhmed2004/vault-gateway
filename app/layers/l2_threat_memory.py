"""L2 threat memory: MiniLM embeddings against a versioned ChromaDB collection (IP-1, IP-2, IP-3, IP-4, IP-5, IP-7, IP-9, TM-R2, TM-R3, §1.9, §3.6).

Scoring (IP-1, IP-2): L2 reads the `minilm` window plan of the shared
InspectionBundle, where every 600-character window (stride 300) of both views
was already split to fit MiniLM's 256-token limit, so nothing is truncated
(`truncation_count` stays 0). Each window is embedded (mean pooling, L2
normalized) and scored against the pinned collection version: its score is
the exact maximum cosine similarity to every stored threat window, computed
with numpy over the version's in-memory embedding matrix (Step 5B.1 LM-3,
§1.9: no approximate HNSW search, so the same input gives the same score on
every machine and at every threshold). A view's score is the maximum over its
windows, and the view blocks when that score reaches the threshold. A block on
either view blocks the request. Identical window texts (for example when the
decoded view equals the raw view) are embedded and scored once per call.

Scan mode (IP-3, IP-4, Step 5B.1 M-2): with `early_stop=True` (production and
`eval`) the scan stops at the first window that reaches the threshold and
returns `block` at once, so a block is never lost to a later window hitting
the deadline; the result holds every completely scanned view plus the
blocking view. With `early_stop=False` (both shadow tiers) every window of
both views is scored. The scan mode is part of the layer version, so the IP-5
cache never replays an early-stopped result into a shadow run.

Versions (IP-5, IP-7): `pin()` reads the store's active-version pointer once
and returns a `PinnedL2` bound to that version; the pipeline pins at request
start and uses the pinned `chroma_version` for the layer-verdict key and every
score, so a concurrent hardening swap never mixes versions inside a request.
`version` is the L2 layer version (embedder, threshold, scoring rule, scan
mode, revision); the collection version is keyed separately as
`chroma_version`.

Budget (TM-R2): with a `budget_ms` the layer checks its deadline after every
window (after the block check) and stops with `LayerBudgetExceeded` (a
`timeout`) once its budget is spent. The shadow tiers build L2 with
`budget_ms=None` (IP-4, Step 5B.1 H2).

P-degrade (TM-R3): `degraded=True` lowers the threshold by `degrade_delta`
(default 0.05); `tighten()` applies the same lowered threshold to completed
results from their scores, without re-running the model.

Hardening (§3.6 step 2): `harden_threat_memory()` embeds every window of both
views of each text and inserts them as a new collection version through
`ThreatMemoryStore.harden()`, which swaps the active pointer atomically and
thereby invalidates cached L2 verdicts (the key's `chroma_version` changes).
Two guards keep the memory clean:

- Windows whose MiniLM tokens are mostly `[UNK]` (`unk_ratio >=
  max_unk_ratio`, default 0.50) are skipped, so a degenerate embedding can
  never poison L2 (Step 5B.1 M-3). The check needs an embedder with
  `unk_ratio(text)`; `MiniLMEmbedder` has one.
- `forbidden_embeddings` (optional) drops every candidate whose cosine
  similarity to any reference row reaches `max_similarity` (default 0.95), the
  §3.6 near-duplicate rule against the test split. The test-split reference
  matrix is wired by `harness/` in W6 and W14.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from app.inspection import InspectionConfig, build_inspection_bundle
from app.layers.base import Deadline, view_results
from app.onnx_models import Embedder, tokenizer_limit_for
from app.vector_store import CollectionVersion, ThreatMemoryStore, ThreatRecord, normalize_rows
from contracts.hashing import canonical_hash
from contracts.inspection import CapExceeded, InspectionBundle, TextWindow
from contracts.layer_result import LayerResult

logger = logging.getLogger("vault.layers.l2")

L2_REVISION = 2
DEFAULT_L2_THRESHOLD = 0.80
DEFAULT_DEGRADE_DELTA = 0.05
DEFAULT_MAX_UNK_RATIO = 0.50
DEFAULT_MAX_TEST_SIMILARITY = 0.95


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


class L2ThreatMemoryLayer:
    """Embedding-similarity layer over a versioned threat memory (L2)."""

    name = "L2"
    enabled = True

    def __init__(
        self,
        embedder: Embedder,
        store: ThreatMemoryStore,
        *,
        threshold: float = DEFAULT_L2_THRESHOLD,
        degrade_delta: float = DEFAULT_DEGRADE_DELTA,
        budget_ms: float | None = None,
        early_stop: bool = True,
    ) -> None:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("L2 threshold must be a cosine similarity in [-1, 1]")
        if degrade_delta < 0:
            raise ValueError("degrade_delta must be >= 0")
        if store.embedder_version != embedder.version:
            raise ValueError("the threat memory store was opened for a different embedder version")
        self.embedder = embedder
        self.store = store
        self.threshold = threshold
        self.degrade_delta = degrade_delta
        self.budget_ms = budget_ms
        self.early_stop = early_stop
        self.tokenizer_limit = tokenizer_limit_for(embedder)
        self.version = canonical_hash(
            {
                "layer": "L2",
                "revision": L2_REVISION,
                "embedder_version": embedder.version,
                "tokenizer": embedder.name,
                "token_limit": embedder.token_limit,
                "threshold": threshold,
                "score": "max-exact-cosine",
                "scan": "early-stop" if early_stop else "all-windows",
            }
        )

    # Pinning (IP-7)

    def _require_active(self) -> CollectionVersion:
        active = self.store.active()
        if active is None:
            raise RuntimeError("L2 has no active ChromaDB collection version (run scripts/build_l2_collection.py)")
        return active

    @property
    def chroma_version(self) -> str | None:
        return self._require_active().version

    def pin(self) -> PinnedL2:
        """Read the active-version pointer once; the returned layer uses only that version."""
        return PinnedL2(self, self._require_active())

    def inspect(self, bundle: InspectionBundle, *, degraded: bool = False) -> tuple[LayerResult, ...]:
        return self.pin().inspect(bundle, degraded=degraded)

    # Scoring

    def threshold_for(self, degraded: bool) -> float:
        return self.threshold - self.degrade_delta if degraded else self.threshold

    def score(self, bundle: InspectionBundle, collection: CollectionVersion, *, degraded: bool) -> tuple[LayerResult, ...]:
        deadline = Deadline(self.budget_ms)
        plan = bundle.plan_for(self.embedder.name)
        views: tuple[tuple[str, tuple[TextWindow, ...]], ...] = (
            ("raw_view", plan.raw_windows),
            ("decoded_view", plan.decoded_windows),
        )
        threshold = self.threshold_for(degraded)
        scores: dict[str, float] = {}
        stopped = False
        for _, windows in views:
            for window in windows:
                if window.text in scores:
                    continue
                vector = self.embedder.embed(window.text)
                scores[window.text] = _clamp(collection.max_similarity(vector[None, :])[0])
                if self.early_stop and scores[window.text] >= threshold:
                    stopped = True  # IP-3: the block is decided; later windows are not needed
                    break
                deadline.check()
            if stopped:
                break
        return view_results(self.name, views, scores, threshold, deadline.elapsed_ms())

    def tighten(self, results: tuple[LayerResult, ...]) -> tuple[LayerResult, ...]:
        """Re-decide completed L2 results with the P-degrade threshold (TM-R3), from their scores."""
        threshold = self.threshold_for(True)
        return tuple(
            r.model_copy(update={"verdict": "block" if r.overall_score >= threshold else "allow"})
            if r.layer == self.name and r.verdict in ("allow", "block")
            else r
            for r in results
        )


@dataclass(frozen=True)
class PinnedL2:
    """L2 bound to one collection version for the lifetime of one request (IP-7)."""

    layer: L2ThreatMemoryLayer
    collection: CollectionVersion
    name: str = "L2"
    enabled: bool = True

    @property
    def version(self) -> str:
        return self.layer.version

    @property
    def chroma_version(self) -> str | None:
        return self.collection.version

    def inspect(self, bundle: InspectionBundle, *, degraded: bool = False) -> tuple[LayerResult, ...]:
        return self.layer.score(bundle, self.collection, degraded=degraded)

    def tighten(self, results: tuple[LayerResult, ...]) -> tuple[LayerResult, ...]:
        return self.layer.tighten(results)


def _window_record_id(label: str, text: str) -> str:
    return f"{label}:{hashlib.sha256(text.encode('utf-8', 'surrogatepass')).hexdigest()[:32]}"


def threat_records(
    texts: Iterable[str],
    embedder: Embedder,
    *,
    label: str,
    source: str,
    inspection_config: InspectionConfig | None = None,
    max_unk_ratio: float = DEFAULT_MAX_UNK_RATIO,
) -> list[ThreatRecord]:
    """Embed every window of both views of each text (§3.6: "both views and all windows").

    Windows come from the same InspectionBundle builder and MiniLM split plan
    the gateway uses, so stored and queried windows match. Identical windows
    are stored once. A text the bundle refuses (decoded or window cap) raises.
    A window whose tokens are mostly `[UNK]` (`unk_ratio >= max_unk_ratio`) is
    skipped and counted in the log (Step 5B.1 M-3), when the embedder reports
    `unk_ratio`.
    """
    if not 0.0 < max_unk_ratio <= 1.0:
        raise ValueError("max_unk_ratio must be in (0, 1]")
    config = inspection_config or InspectionConfig(tokenizers=(tokenizer_limit_for(embedder),))
    unk_ratio: Callable[[str], float] | None = getattr(embedder, "unk_ratio", None)
    seen: dict[str, ThreatRecord] = {}
    skipped: set[str] = set()
    for text in texts:
        bundle = build_inspection_bundle(text, config)
        if isinstance(bundle, CapExceeded):
            raise ValueError(f"threat text refused by the bundle caps ({bundle.blocked_by}); split it before inserting")
        plan = bundle.plan_for(embedder.name)
        for view, windows in (("raw_view", plan.raw_windows), ("decoded_view", plan.decoded_windows)):
            for window in windows:
                if not window.text.strip():
                    continue
                record_id = _window_record_id(label, window.text)
                if record_id in seen or record_id in skipped:
                    continue
                if callable(unk_ratio) and unk_ratio(window.text) >= max_unk_ratio:
                    skipped.add(record_id)
                    continue
                seen[record_id] = ThreatRecord(
                    record_id=record_id, text=window.text, embedding=embedder.embed(window.text), source=source, view=view
                )
    if skipped:
        logger.warning("L2 refused %d window(s) that are mostly [UNK] tokens (unk_ratio >= %.2f)", len(skipped), max_unk_ratio)
    return [seen[key] for key in sorted(seen)]


def near_duplicates(records: Iterable[ThreatRecord], forbidden_embeddings: np.ndarray, max_similarity: float) -> set[str]:
    """Record ids whose cosine similarity to any forbidden reference row is >= `max_similarity` (§3.6 step 2)."""
    reference = normalize_rows(np.asarray(forbidden_embeddings, dtype=np.float64))
    if reference.ndim != 2 or reference.shape[0] == 0:
        return set()
    flagged = set()
    for record in records:
        vector = normalize_rows(np.asarray(record.embedding, dtype=np.float64)[None, :])
        if vector.shape[1] != reference.shape[1]:
            raise ValueError("forbidden_embeddings and the records have different dimensions")
        if float(np.max(vector @ reference.T)) >= max_similarity:
            flagged.add(record.record_id)
    return flagged


def harden_threat_memory(
    store: ThreatMemoryStore,
    embedder: Embedder,
    texts: Iterable[str],
    *,
    label: str,
    source: str = "hardening",
    inspection_config: InspectionConfig | None = None,
    forbidden_embeddings: np.ndarray | None = None,
    max_similarity: float = DEFAULT_MAX_TEST_SIMILARITY,
    max_unk_ratio: float = DEFAULT_MAX_UNK_RATIO,
) -> CollectionVersion:
    """Insert `texts` as a new collection version and swap the active pointer atomically (§3.6, IP-5, IP-7).

    Existing records keep their ids; a window already stored under the same
    label is skipped. Mostly-`[UNK]` windows are skipped (`max_unk_ratio`).
    With `forbidden_embeddings` (the test-split reference matrix, one row per
    reference embedding), every candidate window with cosine similarity
    `>= max_similarity` to any row is dropped (§3.6 step 2 near-duplicate
    guard). Returns the new active version. Cached L2 verdicts keyed by the
    previous version no longer match.
    """
    if not -1.0 <= max_similarity <= 1.0:
        raise ValueError("max_similarity must be a cosine similarity in [-1, 1]")
    started = time.perf_counter()
    active = store.active()
    existing = {r.record_id for r in active.records()} if active is not None else set()
    records = [
        r for r in threat_records(
            texts, embedder, label=label, source=source, inspection_config=inspection_config, max_unk_ratio=max_unk_ratio
        )
        if r.record_id not in existing
    ]
    if forbidden_embeddings is not None:
        dropped = near_duplicates(records, forbidden_embeddings, max_similarity)
        if dropped:
            logger.warning(
                "L2 hardening %s dropped %d window(s) with cosine >= %.2f to a test-split reference (§3.6)",
                label, len(dropped), max_similarity,
            )
        records = [r for r in records if r.record_id not in dropped]
    if not records:
        raise ValueError("nothing new to insert: every window is already in the active version or was filtered out")
    version = store.harden(records, label=label)
    logger.info(
        "L2 hardening %s inserted %d windows in %.0f ms; active version %s",
        label, len(records), (time.perf_counter() - started) * 1000.0, version.version[:12],
    )
    return version
