"""Unit tests for the permanent PS-5 linter (PS-5, AGENTS.md Rule 11).

Banned characters are built from escapes so this file passes the linter.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ps5_lint  # noqa: E402

EM_DASH = "\u2014"
EN_DASH = "\u2013"
EMOJI_ROCKET = "\U0001F680"
EMOJI_CHECK = "\u2705"
EMOJI_SPARKLES = "\u2728"


@pytest.mark.parametrize("char", [EM_DASH, EN_DASH, EMOJI_ROCKET, EMOJI_CHECK, EMOJI_SPARKLES, "\ufe0f"])
def test_banned_characters_detected(char):
    assert ps5_lint.is_banned(char)


@pytest.mark.parametrize("char", ["-", "a", "\u00a7", "\u00e9", "\u4e2d", "|", ">", "\u2192"])
def test_allowed_characters_pass(char):
    assert not ps5_lint.is_banned(char)


def test_scan_text_reports_line_and_column():
    hits = ps5_lint.scan_text(f"ok line\nbad {EM_DASH} here", Path("x.md"))
    assert len(hits) == 1
    assert (hits[0].line, hits[0].column, hits[0].char) == (2, 5, EM_DASH)
    assert "U+2014" in hits[0].format()


def test_main_exit_1_on_dirty_tree(tmp_path):
    (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "dirty.md").write_text(f"# Title {EMOJI_ROCKET}\nrange 1{EN_DASH}2\n", encoding="utf-8")
    assert ps5_lint.main(["--root", str(tmp_path)]) == 1


def test_main_exit_0_on_clean_tree(tmp_path):
    (tmp_path / "clean.py").write_text("x = 1  # plain hyphen - only\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("Section \u00a75.5\n", encoding="utf-8")
    assert ps5_lint.main(["--root", str(tmp_path)]) == 0


def test_skipped_directories_and_binaries_ignored(tmp_path):
    for skipped in (".venv", ".git", "results"):
        d = tmp_path / skipped
        d.mkdir()
        (d / "file.txt").write_text(EM_DASH, encoding="utf-8")
    (tmp_path / "spec.pdf").write_bytes(EM_DASH.encode("utf-8"))
    (tmp_path / "blob.dat").write_bytes(b"\x00\x01" + EM_DASH.encode("utf-8"))
    assert ps5_lint.lint([tmp_path]) == []


@pytest.mark.parametrize("rel", ["data/raw", "data/cache", "data/translations", "results"])
def test_dataset_and_cache_dirs_skipped_with_jsonl(tmp_path, rel):
    d = tmp_path / rel
    d.mkdir(parents=True)
    line = '{"text": "rocket ' + EMOJI_ROCKET + " dash " + EM_DASH + '"}\n'
    (d / "rows.jsonl").write_text(line, encoding="utf-8")
    assert ps5_lint.lint([tmp_path]) == []
    assert ps5_lint.main(["--root", str(tmp_path)]) == 0


def test_similar_named_dir_is_not_skipped(tmp_path):
    d = tmp_path / "metadata" / "raw"
    d.mkdir(parents=True)
    (d / "x.md").write_text(EM_DASH, encoding="utf-8")
    assert ps5_lint.main(["--root", str(tmp_path)]) == 1


def test_files_over_5_mib_are_skipped(tmp_path):
    big = tmp_path / "big.txt"
    big.write_bytes(b"a" * (5 * 1024 * 1024) + EM_DASH.encode("utf-8"))
    assert ps5_lint.lint([tmp_path]) == []
    small = tmp_path / "small.txt"
    small.write_text(EM_DASH, encoding="utf-8")
    assert len(ps5_lint.lint([tmp_path])) == 1


@pytest.mark.parametrize("char", ["\u2713", "\u2717", "\u2715", "\u25cf", "\u25a0", "\u2502", "\u2514", "|", "+"])
def test_plain_status_glyphs_allowed(char):
    assert not ps5_lint.is_banned(char)


def test_status_glyphs_pass_in_tree_but_emoji_fails(tmp_path):
    (tmp_path / "console.md").write_text("\u2713 ok \u2717 fail \u25cf up\n", encoding="utf-8")
    assert ps5_lint.main(["--root", str(tmp_path)]) == 0
    (tmp_path / "bad.md").write_text(f"status {EMOJI_CHECK}\n", encoding="utf-8")
    assert ps5_lint.main(["--root", str(tmp_path)]) == 1


@pytest.mark.parametrize("char", ["\u2764", "\u26a0", "\u274c", "\u2b50", "\U0001F525"])
def test_true_pictographs_still_banned(char):
    assert ps5_lint.is_banned(char)


def test_repository_is_ps5_clean():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "ps5_lint.py")],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
