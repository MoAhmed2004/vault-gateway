"""Build or extend the L2 threat memory and activate the new collection version (operator utility, IP-5, IP-7, §3.6).

Usage:
    python scripts/build_l2_collection.py --input FILE.jsonl --label LABEL [--fresh]
                                         [--chroma DIR] [--model-dir DIR]

`FILE.jsonl` holds one JSON object per line with a `text` field (and an
optional `source`). Every window of both views of each text (the same
InspectionBundle windows and MiniLM sub-window split the gateway uses) is
embedded and stored.

- Without `--fresh`, the records are added on top of the active version as a
  new immutable version (the hardening path, §3.6 step 2).
- With `--fresh` (or when no version is active yet), a new root version holds
  only these records.

Either way the active-version pointer is swapped atomically, which bumps the
ChromaDB version in the IP-5 L2 cache key and so invalidates cached L2
verdicts. Measured runs swap only between runs (IP-7).

Population rule (§2.4): measured runs populate L2 from S1-train only.
`data/l2_seed/bootstrap.jsonl` is a small hand-written bootstrap set for
local development and the demo, not experiment data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import DEFAULT_CHROMA_DB_PATH, DEFAULT_MINILM_MODEL_DIR  # noqa: E402
from app.inspection import InspectionConfig  # noqa: E402
from app.layers.l2_threat_memory import harden_threat_memory, threat_records  # noqa: E402
from app.onnx_models import MiniLMEmbedder, tokenizer_limit_for  # noqa: E402
from app.vector_store import ThreatMemoryStore  # noqa: E402


def read_texts(path: Path) -> tuple[list[str], str]:
    texts: list[str] = []
    sources: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError(f"{path}:{number}: each line needs a non-empty 'text' string")
        texts.append(item["text"])
        sources.add(str(item.get("source", path.stem)))
    if not texts:
        raise ValueError(f"{path} holds no texts")
    return texts, ",".join(sorted(sources))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--fresh", action="store_true", help="start a new root version instead of extending the active one")
    parser.add_argument("--chroma", type=Path, default=DEFAULT_CHROMA_DB_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MINILM_MODEL_DIR)
    args = parser.parse_args(argv)

    texts, source = read_texts(args.input)
    embedder = MiniLMEmbedder(args.model_dir)
    store = ThreatMemoryStore(args.chroma, embedder_version=embedder.version)
    store.open()
    config = InspectionConfig(tokenizers=(tokenizer_limit_for(embedder),))
    if args.fresh or store.active() is None:
        records = threat_records(texts, embedder, label=args.label, source=source, inspection_config=config)
        version = store.create_version(records, label=args.label, parent=None)
        store.activate(version)
    else:
        version = harden_threat_memory(store, embedder, texts, label=args.label, source=source, inspection_config=config)
    print(f"active L2 collection version {version.version} label={version.label} records={version.count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
