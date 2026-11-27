"""Download the pinned L2 and L3 model artifacts into `models/` (operator utility, §2.7, §5.3).

Usage:
    python scripts/fetch_models.py [--dest DIR] [--only minilm|deberta]

Each artifact is fetched from a fixed Hugging Face commit (never a branch
name), written to a temporary file, checked against the SHA-256 pinned below,
and only then moved into place. A file already present with the pinned hash is
kept. A hash mismatch deletes the temporary file and exits 1. The same hashes
are recorded in `docs/artifacts.md`.

`models/` is git-ignored (model weights are never committed, §5.3). The
directory must be on local disk, like every other gateway store (IP-8).

Artifacts:
- `minilm/`: `sentence-transformers/all-MiniLM-L6-v2`, the ONNX export shipped
  in the model repository (`onnx/model.onnx`) and its WordPiece tokenizer
  (L2 threat memory and the X3 similarity leak check).
- `deberta/`: `protectai/deberta-v3-base-prompt-injection-v2`, an
  off-the-shelf DeBERTa-v3 prompt-injection classifier with an ONNX export in
  the repository (`onnx/model.onnx`), its tokenizer and label config (L3).
  The L3 path is a setting, so a Colab fine-tuned export replaces it later.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT / "models"
CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class Artifact:
    repo: str
    revision: str
    remote_path: str
    local_name: str
    sha256: str


MODELS: dict[str, tuple[Artifact, ...]] = {
    "minilm": (
        Artifact(
            "sentence-transformers/all-MiniLM-L6-v2",
            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "onnx/model.onnx",
            "model.onnx",
            "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
        ),
        Artifact(
            "sentence-transformers/all-MiniLM-L6-v2",
            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "tokenizer.json",
            "tokenizer.json",
            "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
        ),
    ),
    "deberta": (
        Artifact(
            "protectai/deberta-v3-base-prompt-injection-v2",
            "90c9989b1a342275dd0d1a95aad283c04e075671",
            "onnx/model.onnx",
            "model.onnx",
            "f0ea7f239f765aedbde7c9e163a7cb38a79c5b8853d3f76db5152172047b228c",
        ),
        Artifact(
            "protectai/deberta-v3-base-prompt-injection-v2",
            "90c9989b1a342275dd0d1a95aad283c04e075671",
            "onnx/tokenizer.json",
            "tokenizer.json",
            "752fe5f0d5678ad563e1bd2ecc1ddf7a3ba7e2024d0ac1dba1a72975e26dff2f",
        ),
        Artifact(
            "protectai/deberta-v3-base-prompt-injection-v2",
            "90c9989b1a342275dd0d1a95aad283c04e075671",
            "onnx/config.json",
            "config.json",
            "3093743035223c46b1497a72e939e56fa0a50afbd7bafbf7eb8aad060b8d23f8",
        ),
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(artifact: Artifact, dest_dir: Path, client: httpx.Client) -> bool:
    """Download one artifact; True when the file in place matches its pinned hash."""
    target = dest_dir / artifact.local_name
    if target.exists() and sha256_file(target) == artifact.sha256:
        print(f"ok (present)  {target}")
        return True
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{artifact.repo}/resolve/{artifact.revision}/{artifact.remote_path}"
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    with client.stream("GET", url) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for block in response.iter_bytes(CHUNK_BYTES):
                handle.write(block)
                digest.update(block)
    actual = digest.hexdigest()
    if actual != artifact.sha256:
        partial.unlink(missing_ok=True)
        print(f"FAIL hash     {artifact.repo}/{artifact.remote_path}: got {actual}, pinned {artifact.sha256}")
        return False
    partial.replace(target)
    print(f"ok (fetched)  {target}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--only", choices=sorted(MODELS), default=None)
    args = parser.parse_args(argv)
    names = [args.only] if args.only else sorted(MODELS)
    ok = True
    timeout = httpx.Timeout(60.0, read=300.0)
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        for name in names:
            for artifact in MODELS[name]:
                ok = fetch(artifact, args.dest / name, client) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
