"""Attacker isolation guard (PS-2, §5.2, AGENTS.md Rule 9).

`attacker/` may talk to the gateway only through its public HTTP endpoint. An
AST scan of every `.py` file under `attacker/` fails on:

- imports of `app`, `telemetry`, `harness`, and the storage and model stacks
  (`sqlite3`, `aiosqlite`, `chromadb`, `onnxruntime`, `presidio_analyzer`,
  `presidio_anonymizer`, `sentence_transformers`);
- any reference to `HMAC_SECRET` or the storage path settings `CHROMA_DB_PATH`,
  `CACHE_DB_PATH`, `LOG_DB_PATH` (and the model directory settings),
  case-insensitive (names, attributes, strings such as
  `os.environ.get("hmac_secret")`);
- any path to `data/manifests`, `data/cache`, `data/logs`, `data/l2_seed`, or the
  L2 store `data/chroma_db` (its contents are secret from the attacker, §1.3),
  written as a string or constructed with `Path(...) / ...`, `Path(a, b)`,
  `.joinpath`, or `os.path.join`, with a path boundary required before `data`.
  Paths are normalized first (backslashes, repeated separators such as
  `data//chroma_db`, `.` segments), a `chroma_db` segment is forbidden anywhere
  (bare `chroma_db/` included), and a relative path whose first segment is the
  repo-root `models/` directory (model weights and tokenizers) is forbidden
  (Step 5B.1, PS-2, CA-4);
- dynamic `importlib.import_module(x)` and `__import__(x)` unless `x` is a
  constant safe string literal naming an allowed module;
- relative imports that climb out of `attacker/`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ATTACKER_DIR = REPO_ROOT / "attacker"

FORBIDDEN_MODULES = frozenset(
    {
        "app",
        "telemetry",
        "harness",
        "sqlite3",
        "aiosqlite",
        "chromadb",
        "onnxruntime",
        "presidio_analyzer",
        "presidio_anonymizer",
        "sentence_transformers",
    }
)
FORBIDDEN_SECRET_NAMES = (
    "hmac_secret",
    "chroma_db_path",
    "cache_db_path",
    "log_db_path",
    "minilm_model_dir",
    "l3_model_dir",
)
# chroma_db: L2 DB contents (§1.3); l2_seed: the L2 seed texts the threat memory is built from.
FORBIDDEN_DATA_DIRS = frozenset({"manifests", "cache", "logs", "chroma_db", "l2_seed"})
FORBIDDEN_ANYWHERE = frozenset({"chroma_db"})
FORBIDDEN_ROOT_DIRS = frozenset({"models"})
DYNAMIC_IMPORTERS = frozenset({"import_module", "__import__"})
PATH_BUILDERS = frozenset({"join", "joinpath", "Path", "PurePath", "PosixPath", "WindowsPath", "PurePosixPath", "PureWindowsPath"})
SAFE_MODULE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _module_is_forbidden(module: str | None) -> bool:
    return bool(module) and module.split(".")[0] in FORBIDDEN_MODULES


def _is_secret_name(value: str) -> bool:
    lowered = value.lower()
    return any(name in lowered for name in FORBIDDEN_SECRET_NAMES)


def _segments(value: str) -> list[str]:
    """Lowercase path segments of a string: backslashes and repeated separators collapsed, `.` dropped."""
    return [p.lower() for p in re.split(r"[/\\]+", value) if p and p != "."]


def _string_path_violation(value: str) -> bool:
    """True when a string names a forbidden path (any separator style, `data//x`, `./data/./x`, bare `chroma_db/`)."""
    if "://" in value:  # a URL is not a repository path (the attacker's own gateway URL)
        return False
    parts = _segments(value)
    if not parts:
        return False
    if _has_forbidden_data_path(parts) or any(p in FORBIDDEN_ANYWHERE for p in parts):
        return True
    while parts and parts[0] == "..":
        parts = parts[1:]
    has_separator = bool(re.search(r"[/\\]", value))  # a bare word "models" is not a path
    return has_separator and bool(parts) and parts[0] in FORBIDDEN_ROOT_DIRS


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _path_parts(node: ast.AST) -> list[str | None]:
    """Flatten a path expression into lowercase segments; None marks an unknown segment."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return list(_segments(node.value))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _path_parts(node.left) + _path_parts(node.right)
    if isinstance(node, ast.Call) and _call_name(node) in PATH_BUILDERS:
        parts: list[str | None] = []
        if isinstance(node.func, ast.Attribute) and node.func.attr == "joinpath":
            parts.extend(_path_parts(node.func.value))
        for arg in node.args:
            parts.extend(_path_parts(arg))
        return parts
    return [None]


def _has_forbidden_data_path(parts: list[str | None]) -> bool:
    for left, right in zip(parts, parts[1:]):
        if left == "data" and right in FORBIDDEN_DATA_DIRS:
            return True
    return False


def _constructed_path_violation(parts: list[str | None]) -> bool:
    """A constructed path into a forbidden data dir, through any `chroma_db`, or into the repo-root `models/`."""
    if _has_forbidden_data_path(parts) or any(p in FORBIDDEN_ANYWHERE for p in parts):
        return True
    known = [p for p in parts if p != ".."]
    # `models` first, or right after an unknown base such as `Path(__file__).parent`, is the repo-root dir.
    return any(p in FORBIDDEN_ROOT_DIRS and (i == 0 or known[i - 1] is None) for i, p in enumerate(known))


def find_violations(source: str, filename: str = "<memory>", depth: int = 0) -> list[str]:
    """Return human-readable isolation violations found in `source`.

    `depth` is the number of directories between `attacker/` and the file. A
    relative import of `level` stays inside `attacker/` when `level - 1 <= depth`.
    """
    tree = ast.parse(source, filename=filename)
    violations: list[str] = []

    def report(node: ast.AST, message: str) -> None:
        violations.append(f"{filename}:{getattr(node, 'lineno', 0)}: {message}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _module_is_forbidden(alias.name):
                    report(node, f"import {alias.name}")
                if _is_secret_name(alias.name):
                    report(node, f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                if node.level - 1 > depth:
                    report(node, f"relative import escapes attacker/: level {node.level}")
            elif _module_is_forbidden(node.module):
                report(node, f"from {node.module} import ...")
            for alias in node.names:
                if _is_secret_name(alias.name):
                    report(node, f"from {node.module} import {alias.name}")
        elif isinstance(node, ast.Name) and _is_secret_name(node.id):
            report(node, f"name {node.id}")
        elif isinstance(node, ast.Attribute) and _is_secret_name(node.attr):
            report(node, f"attribute {node.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _is_secret_name(node.value):
                report(node, f"string reference {node.value!r}")
            if _string_path_violation(node.value):
                report(node, f"path reference {node.value!r}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if _constructed_path_violation(_path_parts(node)):
                report(node, "constructed path to a forbidden data, store, or models directory")
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name in PATH_BUILDERS and _constructed_path_violation(_path_parts(node)):
                report(node, "constructed path to a forbidden data, store, or models directory")
            if name in DYNAMIC_IMPORTERS:
                target: ast.AST | None = node.args[0] if node.args else None
                for kw in node.keywords:
                    if kw.arg == "name":
                        target = kw.value
                if not (
                    isinstance(target, ast.Constant)
                    and isinstance(target.value, str)
                    and SAFE_MODULE_NAME.match(target.value)
                ):
                    report(node, f"dynamic import with non-constant or unsafe module name ({name})")
                elif _module_is_forbidden(target.value):
                    report(node, f"dynamic import {target.value!r}")
    return violations


def scan_directory(directory: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(directory.rglob("*.py")):
        depth = len(path.relative_to(directory).parts) - 1
        violations.extend(find_violations(path.read_text(encoding="utf-8"), str(path), depth=depth))
    return violations


# Real package scan


def test_attacker_package_exists_and_has_python_files():
    assert ATTACKER_DIR.is_dir()
    assert list(ATTACKER_DIR.rglob("*.py")), "attacker/ has no Python files to scan"


def test_attacker_package_is_isolated():
    violations = scan_directory(ATTACKER_DIR)
    assert violations == [], "\n".join(violations)


# Negative tests: the scanner must catch each forbidden pattern


@pytest.mark.parametrize(
    "source",
    [
        "import app.config",
        "import app",
        "import telemetry.writer as w",
        "import harness",
        "from telemetry import db",
        "from app import settings",
        "from app.layers.l4_egress import scan",
        "from harness.oracles import judge",
        "import os, app.config",
        "def f():\n    import app.config\n",
    ],
)
def test_scanner_catches_forbidden_imports(source):
    assert find_violations(source), f"scanner missed: {source!r}"


@pytest.mark.parametrize(
    "source",
    [
        "import sqlite3",
        "import aiosqlite",
        "import chromadb",
        "import onnxruntime as ort",
        "import presidio_analyzer",
        "import presidio_anonymizer",
        "import sentence_transformers",
        "from sqlite3 import connect",
        "from aiosqlite import connect",
        "from chromadb.config import Settings",
        "from onnxruntime import InferenceSession",
        "from presidio_analyzer import AnalyzerEngine",
        "from presidio_anonymizer import AnonymizerEngine",
        "from sentence_transformers import SentenceTransformer",
        "m = __import__('sqlite3')",
        "import importlib\nm = importlib.import_module('chromadb')",
    ],
)
def test_scanner_catches_storage_and_model_stack_imports(source):
    assert find_violations(source), f"scanner missed: {source!r}"


@pytest.mark.parametrize(
    "source",
    [
        "from somewhere import HMAC_SECRET",
        "from somewhere import hmac_secret",
        "x = HMAC_SECRET",
        "x = hmac_secret",
        "x = settings.HMAC_SECRET",
        "x = settings.Hmac_Secret",
        "import os\nx = os.environ['HMAC_SECRET']",
        "import os\nx = os.environ['hmac_secret']",
        "import os\nx = os.environ.get('hmac_secret')",
        "import os\nx = os.getenv('hmac_secret')",
        "import os\nx = os.getenv('Hmac_Secret', 'd')",
    ],
)
def test_scanner_catches_hmac_secret(source):
    assert find_violations(source), f"scanner missed: {source!r}"


@pytest.mark.parametrize(
    "source",
    [
        "p = 'data/manifests/objectives.json'",
        "p = 'data\\\\manifests\\\\objectives.json'",
        "from pathlib import Path\np = Path('data/cache/responses.sqlite')",
        "p = '../DATA/Manifests/x.json'",
        "p = 'data/logs/requests.db'",
        "p = './data/cache'",
        "from pathlib import Path\np = Path('data') / 'manifests'",
        "from pathlib import Path\np = Path('data') / 'cache' / 'x.json'",
        "from pathlib import Path\np = Path('data') / 'logs'",
        "from pathlib import Path\np = Path('root') / 'data' / 'manifests'",
        "from pathlib import Path\np = Path('data', 'cache')",
        "from pathlib import Path\np = Path('data').joinpath('manifests')",
        "from pathlib import Path\np = Path('.').joinpath('data', 'logs')",
        "import os\np = os.path.join('data', 'cache')",
        "import os\np = os.path.join('data', 'manifests', 'x.json')",
        "import os\np = os.path.join('base', 'data', 'logs')",
        "import os\np = os.path.join('DATA', 'Cache')",
        "p = 'data/chroma_db/active_version.json'",
        "from pathlib import Path\np = Path('data') / 'chroma_db'",
        "import os\np = os.path.join('data', 'chroma_db')",
    ],
)
def test_scanner_catches_manifest_cache_and_log_paths(source):
    assert find_violations(source), f"scanner missed: {source!r}"


@pytest.mark.parametrize(
    "source",
    [
        "import importlib\nm = importlib.import_module('app.config')",
        "m = __import__('telemetry')",
        "from .. import app",
        "import importlib\nm = importlib.import_module(name)",
        "m = __import__(name)",
        "import importlib\nm = importlib.import_module('a' + 'pp')",
        "import importlib\nm = importlib.import_module(f'{pkg}.x')",
        "import importlib\nm = importlib.import_module(name='x' + y)",
        "import importlib\nm = importlib.import_module('.sibling')",
        "import importlib\nm = importlib.import_module(*args)",
    ],
)
def test_scanner_catches_dynamic_and_escaping_imports(source):
    assert find_violations(source), f"scanner missed: {source!r}"


@pytest.mark.parametrize(
    "source",
    [
        # data/l2_seed: the L2 seed texts
        "p = 'data/l2_seed/bootstrap.jsonl'",
        "from pathlib import Path\np = Path('data') / 'l2_seed'",
        "import os\np = os.path.join('data', 'l2_seed', 'bootstrap.jsonl')",
        # the repo-root models/ directory
        "p = 'models/minilm/tokenizer.json'",
        "p = './models/deberta/model.onnx'",
        "p = '../models/minilm'",
        "p = 'models\\\\deberta\\\\config.json'",
        "from pathlib import Path\np = Path('models') / 'minilm'",
        "from pathlib import Path\np = Path(__file__).resolve().parent.parent / 'models'",
        "import os\np = os.path.join(ROOT, 'models', 'deberta')",
        # storage path settings
        "x = settings.CHROMA_DB_PATH",
        "x = CACHE_DB_PATH",
        "import os\nx = os.environ['LOG_DB_PATH']",
        "import os\nx = os.getenv('chroma_db_path')",
        "from somewhere import Log_Db_Path",
        "x = settings.MINILM_MODEL_DIR",
        # normalized path tricks
        "p = 'data//chroma_db/x'",
        "p = 'data\\\\\\\\cache'",
        "p = './data/./manifests/o.json'",
        "p = 'chroma_db/'",
        "p = 'chroma_db'",
        "p = 'store/chroma_db/chroma.sqlite3'",
        "from pathlib import Path\np = Path('chroma_db')",
        "from pathlib import Path\np = Path('data') / '.' / 'logs'",
        "from pathlib import Path\np = Path('data//cache')",
    ],
)
def test_scanner_catches_seed_models_settings_and_normalized_paths(source):
    """Step 5B.1 (PS-2, CA-4): seed data, model weights, storage settings and separator tricks."""
    assert find_violations(source), f"scanner missed: {source!r}"


@pytest.mark.parametrize(
    "source",
    [
        "url = 'http://127.0.0.1:8000/v1/models'",  # a URL is not a repository path
        "x = {'models': 3}",
        "p = 'harness/models/notes.txt'",
        "p = 'app/models/x'",
        "from pathlib import Path\np = Path('reports') / 'models.txt'",
        "p = 'data/modelsx/y'",
        "p = 'my_chroma_db_notes.txt'",
    ],
)
def test_scanner_new_rules_have_no_false_positives(source):
    assert find_violations(source) == []


def test_scanner_allows_constant_safe_dynamic_import():
    assert find_violations("import importlib\nm = importlib.import_module('json')") == []
    assert find_violations("m = __import__('hashlib')") == []


# Relative-import depth handling


def test_relative_import_two_levels_inside_nested_package_is_allowed():
    assert find_violations("from .. import util", "attacker/sub/mod.py", depth=1) == []
    assert find_violations("from ..util import helper", "attacker/sub/mod.py", depth=1) == []


def test_relative_import_escaping_from_nested_package_is_flagged():
    assert find_violations("from ... import app", "attacker/sub/mod.py", depth=1)
    assert find_violations("from ... import util", "attacker/sub/mod.py", depth=1)


def test_relative_import_escaping_from_top_level_file_is_flagged():
    assert find_violations("from .. import util", "attacker/mod.py", depth=0)
    assert find_violations("from ..x import y", "attacker/mod.py", depth=0)


def test_scan_directory_computes_depth_for_nested_files(tmp_path):
    pkg = tmp_path / "attacker"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "util.py").write_text("", encoding="utf-8")
    (pkg / "sub" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "sub" / "mod.py").write_text("from .. import util\n", encoding="utf-8")
    assert scan_directory(pkg) == []
    (pkg / "sub" / "bad.py").write_text("from ... import app\n", encoding="utf-8")
    violations = scan_directory(pkg)
    assert len(violations) == 1 and "bad.py" in violations[0]
    (pkg / "top.py").write_text("from .. import util\n", encoding="utf-8")
    assert any("top.py" in v for v in scan_directory(pkg))


def test_scanner_catches_violation_in_file_tree(tmp_path):
    pkg = tmp_path / "attacker"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "client.py").write_text("import httpx\nfrom app import settings\n", encoding="utf-8")
    (pkg / "loader.py").write_text("P = 'data/manifests/objectives.json'\n", encoding="utf-8")
    violations = scan_directory(pkg)
    assert any("from app import" in v for v in violations)
    assert any("data/manifests" in v for v in violations)


# False-positive guards


def test_path_boundary_required_before_data():
    assert find_violations("p = 'metadata/cache_notes.txt'") == []
    assert find_violations("p = 'metadata/cache/x'") == []
    assert find_violations("p = 'mydata/manifests/x'") == []
    assert find_violations("p = 'data/cache_notes.txt'") == []
    assert find_violations("import os\np = os.path.join('metadata', 'cache')") == []
    assert find_violations("from pathlib import Path\np = Path('metadata') / 'cache_notes.txt'") == []


@pytest.mark.parametrize(
    "source",
    [
        "import httpx",
        "import json, os, hashlib",
        "from contracts import Outcome, canonical_hash",
        "from . import client",
        "import application_helpers",
        "x = 'apply the payload'",
        "client = httpx.Client(base_url='http://127.0.0.1:8000', timeout=10.0)",
        "p = 'data/raw/queries.jsonl'",
        "import os\np = os.path.join('data', 'raw')",
        "import sqlite_utils_not_banned",
    ],
)
def test_scanner_allows_public_http_client_code(source):
    assert find_violations(source) == []
