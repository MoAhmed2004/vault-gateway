"""Repo-wide package boundary guard (AGENTS.md Package Boundaries, §5.2, §5.3).

AST scan of the import table:

- `contracts/`: no import of `app`, `telemetry`, `attacker`, `harness`.
- `telemetry/`: may import `contracts`; no `app`, `attacker`, `harness`.
- `app/` (all files except `app/main.py` and `app/wiring.py`, including
  `app/layers/`): no `telemetry`, `attacker`, `harness`.
- `app/main.py` and `app/wiring.py`: no `attacker`, `harness`; `telemetry` is
  allowed for startup composition.

Relative imports that climb out of the package directory are resolved as
top-level imports so `from .. import telemetry` cannot hide a violation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

PROJECT_PACKAGES = frozenset({"app", "telemetry", "attacker", "harness", "contracts"})
WIRING_FILES = frozenset({"main.py", "wiring.py"})


def forbidden_for(package: str, relative: Path) -> frozenset[str]:
    """Return the project packages that the file at `relative` (inside `package/`) must not import."""
    if package == "contracts":
        return frozenset({"app", "telemetry", "attacker", "harness"})
    if package == "telemetry":
        return frozenset({"app", "attacker", "harness"})
    if package == "app":
        if relative.parent == Path(".") and relative.name in WIRING_FILES:
            return frozenset({"attacker", "harness"})
        return frozenset({"telemetry", "attacker", "harness"})
    raise ValueError(f"no boundary rule for package {package!r}")


def imported_top_levels(source: str, depth: int = 0) -> set[str]:
    """Top-level module names imported by `source`; `depth` is the subdirectory depth inside the package."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    names.add(node.module.split(".")[0])
            elif node.level - 1 > depth:
                # Escapes the package directory, so the target is a top-level module.
                if node.module:
                    names.add(node.module.split(".")[0])
                else:
                    names.update(alias.name.split(".")[0] for alias in node.names)
    return names


def check_tree(root: Path) -> list[str]:
    """Return boundary violations for the packages found under `root`."""
    violations: list[str] = []
    for package in ("contracts", "telemetry", "app"):
        package_dir = root / package
        if not package_dir.is_dir():
            continue
        for path in sorted(package_dir.rglob("*.py")):
            relative = path.relative_to(package_dir)
            depth = len(relative.parts) - 1
            bad = imported_top_levels(path.read_text(encoding="utf-8"), depth) & forbidden_for(package, relative)
            for name in sorted(bad):
                violations.append(f"{package}/{relative.as_posix()} imports {name}")
    return violations


# Real repository


def test_real_repository_respects_package_boundaries():
    violations = check_tree(REPO_ROOT)
    assert violations == [], "\n".join(violations)


def test_real_contracts_package_is_scanned():
    assert list((REPO_ROOT / "contracts").rglob("*.py"))


# Rule table


@pytest.mark.parametrize(
    ("package", "relative", "expected"),
    [
        ("contracts", "layer_result.py", {"app", "telemetry", "attacker", "harness"}),
        ("telemetry", "writer.py", {"app", "attacker", "harness"}),
        ("app", "config.py", {"telemetry", "attacker", "harness"}),
        ("app", "layers/l1_regex.py", {"telemetry", "attacker", "harness"}),
        ("app", "main.py", {"attacker", "harness"}),
        ("app", "wiring.py", {"attacker", "harness"}),
        ("app", "sub/main.py", {"telemetry", "attacker", "harness"}),
    ],
)
def test_rule_table(package, relative, expected):
    assert forbidden_for(package, Path(relative)) == expected


def test_telemetry_may_import_contracts_and_wiring_may_import_telemetry():
    assert "contracts" not in forbidden_for("telemetry", Path("writer.py"))
    assert "telemetry" not in forbidden_for("app", Path("wiring.py"))
    assert "telemetry" not in forbidden_for("app", Path("main.py"))


# Negative and positive tests on synthetic trees


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_layer_importing_telemetry_fails(tmp_path):
    _write(tmp_path, "app/layers/l1.py", "import telemetry\n")
    assert check_tree(tmp_path) == ["app/layers/l1.py imports telemetry"]


def test_layer_from_import_telemetry_fails(tmp_path):
    _write(tmp_path, "app/layers/l1.py", "from telemetry.writer import Writer\n")
    assert check_tree(tmp_path)


def test_wiring_importing_telemetry_passes(tmp_path):
    _write(tmp_path, "app/wiring.py", "from telemetry.writer import Writer\nimport contracts\n")
    _write(tmp_path, "app/main.py", "import telemetry\n")
    assert check_tree(tmp_path) == []


def test_wiring_importing_attacker_or_harness_fails(tmp_path):
    _write(tmp_path, "app/wiring.py", "import attacker\n")
    _write(tmp_path, "app/main.py", "from harness import runner\n")
    assert sorted(check_tree(tmp_path)) == ["app/main.py imports harness", "app/wiring.py imports attacker"]


def test_nested_main_py_is_not_exempt(tmp_path):
    _write(tmp_path, "app/layers/main.py", "import telemetry\n")
    assert check_tree(tmp_path)


def test_app_module_importing_attacker_or_harness_fails(tmp_path):
    _write(tmp_path, "app/pipeline.py", "import harness\nimport attacker.client\n")
    assert sorted(check_tree(tmp_path)) == ["app/pipeline.py imports attacker", "app/pipeline.py imports harness"]


def test_app_may_import_contracts_and_itself(tmp_path):
    _write(tmp_path, "app/pipeline.py", "from contracts import Outcome\nfrom app.config import X\nfrom . import config\n")
    assert check_tree(tmp_path) == []


def test_contracts_importing_any_project_package_fails(tmp_path):
    for name in ("app", "telemetry", "attacker", "harness"):
        _write(tmp_path, f"contracts/{name}_user.py", f"import {name}\n")
    assert len(check_tree(tmp_path)) == 4


def test_telemetry_may_import_contracts_but_not_app(tmp_path):
    _write(tmp_path, "telemetry/writer.py", "from contracts import RequestLogRecord\n")
    assert check_tree(tmp_path) == []
    _write(tmp_path, "telemetry/bad.py", "from app.config import Settings\n")
    _write(tmp_path, "telemetry/bad2.py", "import harness\nimport attacker\n")
    assert sorted(check_tree(tmp_path)) == [
        "telemetry/bad.py imports app",
        "telemetry/bad2.py imports attacker",
        "telemetry/bad2.py imports harness",
    ]


def test_relative_import_escaping_package_is_resolved_as_top_level(tmp_path):
    _write(tmp_path, "app/layers/l1.py", "from ...telemetry import writer\n")
    assert check_tree(tmp_path) == ["app/layers/l1.py imports telemetry"]
    _write(tmp_path, "app/layers/l2.py", "from ... import harness\n")
    assert "app/layers/l2.py imports harness" in check_tree(tmp_path)


def test_relative_import_inside_package_is_not_flagged(tmp_path):
    _write(tmp_path, "app/layers/l1.py", "from .. import config\nfrom ..config import X\n")
    assert check_tree(tmp_path) == []
