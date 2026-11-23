"""Attacker isolation guard (PS-2, §5.2, AGENTS.md Rule 9).

`attacker/` may talk to the gateway only through its public HTTP endpoint. An
AST scan of every `.py` file under `attacker/` fails on:

- imports of `app`, `telemetry`, `harness`, and the storage and model stacks
  (`sqlite3`, `aiosqlite`, `chromadb`, `onnxruntime`, `presidio_analyzer`,
  `presidio_anonymizer`, `sentence_transformers`);
- any reference to `HMAC_SECRET`, case-insensitive (names, attributes, strings
  such as `os.environ.get("hmac_secret")`);
- any path to `data/manifests`, `data/cache`, or `data/logs`, written as a
  string or constructed with `Path(...) / ...`, `Path(a, b)`, `.joinpath`, or
  `os.path.join`, with a path boundary required before `data`;
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
FORBIDDEN_SECRET_NAME = "hmac_secret"
FORBIDDEN_DATA_DIRS = frozenset({"manifests", "cache", "logs"})
DYNAMIC_IMPORTERS = frozenset({"import_module", "__import__"})
PATH_BUILDERS = frozenset({"join", "joinpath", "Path", "PurePath", "PosixPath", "WindowsPath", "PurePosixPath", "PureWindowsPath"})
SAFE_MODULE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
PATH_STRING = re.compile(r"(?:^|[/\\])data[/\\](manifests|cache|logs)(?:$|[/\\])", re.IGNORECASE)


def _module_is_forbidden(module: str | None) -> bool:
    return bool(module) and module.split(".")[0] in FORBIDDEN_MODULES


def _is_secret_name(value: str) -> bool:
    return FORBIDDEN_SECRET_NAME in value.lower()


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
        return [p.lower() for p in re.split(r"[/\\]+", node.value) if p]
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
            if PATH_STRING.search(node.value.replace("\\", "/") + "/"):
                report(node, f"path reference {node.value!r}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if _has_forbidden_data_path(_path_parts(node)):
                report(node, "constructed path to data/manifests, data/cache, or data/logs")
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name in PATH_BUILDERS and _has_forbidden_data_path(_path_parts(node)):
                report(node, "constructed path to data/manifests, data/cache, or data/logs")
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
