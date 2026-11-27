"""Artifact registry consistency (§5.3, §2.7): pinned model hashes in the fetch script equal docs/artifacts.md.

When the artifacts are present locally, their on-disk SHA-256 must match the
pins too, and model weights must stay git-ignored.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fetch_module():
    spec = importlib.util.spec_from_file_location("vault_fetch_models", REPO_ROOT / "scripts" / "fetch_models.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


def test_every_pinned_artifact_is_recorded_in_docs_artifacts():
    registry = (REPO_ROOT / "docs" / "artifacts.md").read_text(encoding="utf-8")
    for artifacts in _fetch_module().MODELS.values():
        for artifact in artifacts:
            assert artifact.sha256 in registry, f"{artifact.repo}/{artifact.remote_path} hash missing from docs/artifacts.md"
            assert artifact.revision in registry
            assert len(artifact.revision) == 40  # a commit, never a branch name


def test_local_artifacts_match_their_pins_when_present():
    module = _fetch_module()
    checked = 0
    for name, artifacts in module.MODELS.items():
        for artifact in artifacts:
            path = REPO_ROOT / "models" / name / artifact.local_name
            if path.exists():
                assert module.sha256_file(path) == artifact.sha256, f"{path} differs from its pinned hash"
                checked += 1
    if not checked:
        pytest.skip("no local artifacts (run scripts/fetch_models.py)")


@pytest.mark.parametrize("path", ["models/deberta/model.onnx", "models/minilm/tokenizer.json", "data/chroma_db/active_version.json"])
def test_model_weights_and_the_vector_store_are_git_ignored(path):
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO_ROOT, capture_output=True)
    assert result.returncode == 0, f"{path} is not git-ignored"
