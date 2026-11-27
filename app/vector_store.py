"""L2 threat memory store: immutable ChromaDB collection versions and an atomic pointer (IP-2, IP-5, IP-7, IP-8, §1.9, §3.6).

Storage (IP-8): one ChromaDB `PersistentClient` directory on local disk
(`CHROMA_DB_PATH`, checked by `app.config`; never a synced folder), with
ChromaDB's anonymized telemetry switched off so the store never makes a
network call. ChromaDB is the persistent, versioned storage only.

Scoring (Step 5B.1 LM-3, IP-2, §1.9): `max_similarity` never uses ChromaDB's
approximate HNSW search. The first score against a version loads that
version's embeddings once (rows ordered by record id, each row L2 normalized)
into a read-only in-memory matrix, cached per immutable version id, and every
score is the exact `max(query @ matrix.T)` in float64. The same query gives
the same maximum on every machine and at every threshold. The active version's
matrix is loaded when it is opened or activated, so the first request is not
charged for it.

Versions (IP-7): a collection version is immutable. Its id is a SHA-256 over
the store revision, the embedder version, the label, the parent version and
every record (id, text digest, float32 embedding digest), so the same content
always yields the same id and a different content can never reuse one. The
collection is named after the id and sealed with it in its metadata; nothing
ever adds to, updates or deletes from a version after it is built.

Active-version pointer (IP-7): `active_version.json` in the store directory
names the active version. Activation writes a temporary file, flushes and
fsyncs it, then `os.replace`s it over the pointer (atomic on one volume), and
only then swaps the in-memory reference, which is a single attribute
assignment. A request reads `active()` once at its start and keeps the
returned `CollectionVersion` for all its L2 queries and its IP-5 cache key,
so a swap during the request cannot mix two versions.

Hardening (§3.6 step 2): `harden()` builds a new version holding every record
of the active version plus the new records, then swaps the pointer. Because
the ChromaDB version is part of the L2 layer-verdict key (IP-5), the swap
invalidates every cached L2 verdict without touching the cache. Older
versions stay on disk, so `activate()` on an older version is a rollback.

Development seed (§2.4): a version labelled `bootstrap` (built from
`data/l2_seed/bootstrap.jsonl`) is development and demo data only.
`lineage()` returns a version and every parent, so startup can refuse a
deterministic mode whose active version is, or descends from, `bootstrap`.

Thread safety: queries run concurrently on layer-pool threads; version builds
and activations are serialized by one lock, and matrix loads by another.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from contracts.hashing import canonical_hash

logger = logging.getLogger("vault.vector_store")

STORE_REVISION = 1
POINTER_FILE = "active_version.json"
COLLECTION_PREFIX = "vault_l2_"
BOOTSTRAP_LABEL = "bootstrap"
_NAME_HEX = 40
_SHA256_HEX = frozenset("0123456789abcdef")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _embedding_digest(vector: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(vector, dtype="<f4").tobytes()).hexdigest()


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Each row divided by its L2 norm (float64); an all-zero row stays zero."""
    rows = np.asarray(matrix, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError("expected a 2-D matrix")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)


class _MatrixCache:
    """One read-only, L2-normalized embedding matrix per immutable collection version (LM-3)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_version: dict[str, np.ndarray] = {}

    def get(self, version: str, collection: Any, count: int) -> np.ndarray:
        with self._lock:
            matrix = self._by_version.get(version)
            if matrix is None:
                got = collection.get(include=["embeddings"])
                ordered = sorted(zip(got["ids"], got["embeddings"], strict=True), key=lambda item: item[0])
                matrix = normalize_rows(np.asarray([embedding for _, embedding in ordered], dtype=np.float64))
                if matrix.shape[0] != count:
                    raise RuntimeError(f"collection version {version[:12]} returned {matrix.shape[0]} rows, sealed with {count}")
                matrix.flags.writeable = False
                self._by_version[version] = matrix
            return matrix

    def clear(self) -> None:
        with self._lock:
            self._by_version.clear()


@dataclass(frozen=True)
class ThreatRecord:
    """One stored threat window: its text, embedding and provenance."""

    record_id: str
    text: str
    embedding: np.ndarray = field(repr=False, compare=False)
    source: str = "seed"
    view: str = "raw_view"

    def __post_init__(self) -> None:
        if not self.record_id or not isinstance(self.record_id, str):
            raise ValueError("record_id must be a non-empty string")
        vector = np.asarray(self.embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError("embedding must be a non-empty finite 1-D vector")
        object.__setattr__(self, "embedding", vector)

    def digest(self) -> list[str]:
        return [self.record_id, _sha256_text(self.text), _embedding_digest(self.embedding), self.source, self.view]


@dataclass(frozen=True)
class CollectionVersion:
    """One immutable, sealed collection version. Holds the live ChromaDB handle for queries."""

    version: str
    label: str
    collection_name: str
    count: int
    parent: str | None
    embedder_version: str
    _collection: Any = field(repr=False, compare=False)
    _matrices: _MatrixCache = field(default_factory=_MatrixCache, repr=False, compare=False)

    def matrix(self) -> np.ndarray:
        """The version's L2-normalized embedding matrix (rows by record id), loaded once per version."""
        return self._matrices.get(self.version, self._collection, self.count)

    def max_similarity(self, embeddings: np.ndarray) -> list[float]:
        """Exact maximum cosine similarity of each embedding to every record of this version (LM-3)."""
        queries = np.asarray(embeddings, dtype=np.float64)
        if queries.ndim != 2 or queries.shape[0] == 0:
            return []
        matrix = self.matrix()
        if queries.shape[1] != matrix.shape[1]:
            raise ValueError(f"query dimension {queries.shape[1]} differs from the stored dimension {matrix.shape[1]}")
        return [float(value) for value in np.max(normalize_rows(queries) @ matrix.T, axis=1)]

    def records(self) -> list[ThreatRecord]:
        """Every record of this version (used to build the next version)."""
        got = self._collection.get(include=["embeddings", "documents", "metadatas"])
        records = []
        for record_id, embedding, text, meta in zip(
            got["ids"], got["embeddings"], got["documents"], got["metadatas"], strict=True
        ):
            meta = meta or {}
            records.append(
                ThreatRecord(
                    record_id=record_id,
                    text=text or "",
                    embedding=np.asarray(embedding, dtype=np.float32),
                    source=str(meta.get("source", "seed")),
                    view=str(meta.get("view", "raw_view")),
                )
            )
        return sorted(records, key=lambda r: r.record_id)


def version_id(*, embedder_version: str, label: str, parent: str | None, records: Sequence[ThreatRecord]) -> str:
    """Deterministic content id of a collection version (IP-5 `chroma_version`)."""
    return canonical_hash(
        {
            "component": "L2ThreatMemory",
            "revision": STORE_REVISION,
            "embedder_version": embedder_version,
            "label": label,
            "parent": parent,
            "records": sorted(record.digest() for record in records),
        }
    )


class ThreatMemoryStore:
    """Versioned ChromaDB threat memory for L2."""

    def __init__(self, path: Path | str, *, embedder_version: str) -> None:
        if not embedder_version:
            raise ValueError("embedder_version must be non-empty")
        self.path = Path(path)
        self.embedder_version = embedder_version
        self._client: Any = None
        self._active: CollectionVersion | None = None
        self._write_lock = threading.RLock()
        self._matrices = _MatrixCache()

    # Lifecycle

    def open(self) -> None:
        """Open the persistent client and load the active version named by the pointer file (fail fast)."""
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.path), settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False)
        )
        pointer = self.path / POINTER_FILE
        if pointer.exists():
            data = json.loads(pointer.read_text(encoding="utf-8"))
            active = self._load(str(data["version"]))
            active.matrix()  # load the exact-scoring matrix before the first request (LM-3)
            self._active = active
            logger.info("L2 active collection version %s (%s, %d records)", active.version[:12], active.label, active.count)

    def close(self) -> None:
        self._active = None
        self._client = None
        self._matrices.clear()

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("threat memory store is not open")
        return self._client

    # Reads

    def active(self) -> CollectionVersion | None:
        """The active version. One attribute read: call once per request and keep the result (IP-7)."""
        return self._active

    def _load(self, version: str) -> CollectionVersion:
        if len(version) != 64 or not set(version) <= _SHA256_HEX:
            raise ValueError("collection version must be a SHA-256 hex digest")
        collection = self._require_client().get_collection(COLLECTION_PREFIX + version[:_NAME_HEX], embedding_function=None)
        meta = collection.metadata or {}
        if meta.get("vault_version") != version:
            raise ValueError(f"collection {collection.name} is not sealed with version {version[:12]}")
        if meta.get("vault_embedder_version") != self.embedder_version:
            raise ValueError(
                f"collection version {version[:12]} was built with a different embedder; rebuild it for this L2 model"
            )
        count = collection.count()
        if count != int(meta.get("vault_count", -1)):
            raise ValueError(f"collection version {version[:12]} holds {count} records, sealed with {meta.get('vault_count')}")
        return CollectionVersion(
            version=version,
            label=str(meta.get("vault_label", "")),
            collection_name=collection.name,
            count=count,
            parent=str(meta.get("vault_parent") or "") or None,
            embedder_version=self.embedder_version,
            _collection=collection,
            _matrices=self._matrices,
        )

    def lineage(self, version: CollectionVersion) -> list[CollectionVersion]:
        """`version` followed by every ancestor up to its root (each parent must still be on disk)."""
        chain = [version]
        seen = {version.version}
        while chain[-1].parent is not None:
            parent = self._load(chain[-1].parent)
            if parent.version in seen:
                raise ValueError(f"collection version {parent.version[:12]} has a cyclic parent chain")
            seen.add(parent.version)
            chain.append(parent)
        return chain

    def versions(self) -> list[CollectionVersion]:
        found = []
        for item in self._require_client().list_collections():
            name = item if isinstance(item, str) else item.name
            if not name.startswith(COLLECTION_PREFIX):
                continue
            meta = self._require_client().get_collection(name, embedding_function=None).metadata or {}
            version = meta.get("vault_version")
            if isinstance(version, str) and meta.get("vault_embedder_version") == self.embedder_version:
                found.append(self._load(version))
        return sorted(found, key=lambda v: v.version)

    # Writes (serialized)

    def create_version(
        self, records: Iterable[ThreatRecord], *, label: str, parent: CollectionVersion | None = None
    ) -> CollectionVersion:
        """Build (or reuse, if the same content exists) a sealed version: parent's records plus `records`."""
        if not label or not label.replace("_", "").replace("-", "").isalnum():
            raise ValueError("label must be alphanumeric with '_' or '-'")
        with self._write_lock:
            combined: dict[str, ThreatRecord] = {}
            for record in (parent.records() if parent is not None else []):
                combined[record.record_id] = record
            for record in records:
                if record.record_id in combined:
                    raise ValueError(f"duplicate record id {record.record_id!r}")
                combined[record.record_id] = record
            if not combined:
                raise ValueError("a collection version needs at least one record")
            ordered = [combined[key] for key in sorted(combined)]
            dims = {r.embedding.shape[0] for r in ordered}
            if len(dims) != 1:
                raise ValueError(f"records mix embedding dimensions {sorted(dims)}")
            parent_id = parent.version if parent is not None else None
            version = version_id(embedder_version=self.embedder_version, label=label, parent=parent_id, records=ordered)
            client = self._require_client()
            name = COLLECTION_PREFIX + version[:_NAME_HEX]
            existing = {item if isinstance(item, str) else item.name for item in client.list_collections()}
            if name in existing:
                return self._load(version)  # same content: the sealed version already exists
            collection = client.create_collection(
                name,
                embedding_function=None,
                metadata={
                    "hnsw:space": "cosine",
                    "vault_version": version,
                    "vault_label": label,
                    "vault_parent": parent_id or "",
                    "vault_embedder_version": self.embedder_version,
                    "vault_count": len(ordered),
                    "vault_store_revision": STORE_REVISION,
                },
            )
            batch = max(1, int(client.get_max_batch_size()))
            for start in range(0, len(ordered), batch):
                chunk = ordered[start:start + batch]
                collection.add(
                    ids=[r.record_id for r in chunk],
                    embeddings=[r.embedding.tolist() for r in chunk],
                    documents=[r.text for r in chunk],
                    metadatas=[{"source": r.source, "view": r.view} for r in chunk],
                )
            built = self._load(version)
            logger.info("L2 built collection version %s (%s, %d records)", version[:12], label, built.count)
            return built

    def activate(self, version: CollectionVersion) -> None:
        """Atomically make `version` active: pointer file first (os.replace), then the in-memory reference."""
        if version.embedder_version != self.embedder_version:
            raise ValueError("cannot activate a version built with a different embedder")
        with self._write_lock:
            loaded = self._load(version.version)
            loaded.matrix()  # load the exact-scoring matrix before the pointer moves (LM-3)
            pointer = self.path / POINTER_FILE
            temporary = pointer.with_name(POINTER_FILE + ".tmp")
            payload = {
                "version": loaded.version,
                "collection": loaded.collection_name,
                "label": loaded.label,
                "activated_utc": datetime.now(UTC).isoformat(),
            }
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, pointer)
            self._active = loaded
        logger.info("L2 active collection version is now %s (%s)", loaded.version[:12], loaded.label)

    def harden(self, records: Iterable[ThreatRecord], *, label: str) -> CollectionVersion:
        """New version = active version plus `records`, then an atomic pointer swap (§3.6, IP-5, IP-7)."""
        with self._write_lock:
            built = self.create_version(records, label=label, parent=self._active)
            self.activate(built)
        return built
