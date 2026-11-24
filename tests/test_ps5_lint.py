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


@pytest.mark.parametrize("char", ["\x00", "\x07", "\x08", "\t", "\x0b", "\x0c", "\x1f", "\x7f"])
def test_ascii_control_characters_banned(char):
    assert ps5_lint.is_banned(char)
    hits = ps5_lint.scan_text(f"abc{char}def", Path("doc.md"))
    assert len(hits) == 1
    assert (hits[0].line, hits[0].column, hits[0].char) == (1, 4, char)
    assert f"U+{ord(char):04X}" in hits[0].format()


@pytest.mark.parametrize("char", ["\n", "\r"])
def test_newline_and_carriage_return_allowed(char):
    assert not ps5_lint.is_banned(char)


def test_crlf_file_passes_cleanly(tmp_path):
    (tmp_path / "win.txt").write_bytes(b"line1\r\nline2\r\n")
    assert ps5_lint.main(["--root", str(tmp_path)]) == 0


@pytest.mark.parametrize("char", ["\x07", "\x08", "\x00", "\t", "\x7f"])
def test_main_fails_on_control_characters_in_text_file(tmp_path, char):
    (tmp_path / "dirty.md").write_bytes(f"ok{char}bad\n".encode("utf-8"))
    assert ps5_lint.main(["--root", str(tmp_path)]) == 1


# Step 3.6: Unicode Cf format characters (BOM, zero-width, bidi) and non-UTF-8 files


@pytest.mark.parametrize(
    "char",
    [
        "\ufeff",  # UTF-8 BOM / zero width no-break space
        "\u200b",  # zero width space
        "\u200c",  # zero width non-joiner
        "\u200d",  # zero width joiner
        "\u2060",  # word joiner
        "\u00ad",  # soft hyphen
        "\u200e",  # left-to-right mark
        "\u200f",  # right-to-left mark
        "\u202a",  # left-to-right embedding
        "\u202e",  # right-to-left override
        "\u2066",  # left-to-right isolate
        "\u2067",  # right-to-left isolate
        "\u2069",  # pop directional isolate
    ],
)
def test_unicode_format_cf_characters_banned(char):
    assert ps5_lint.is_banned(char)
    hits = ps5_lint.scan_text(f"a{char}b", Path("file.md"))
    assert len(hits) == 1
    assert (hits[0].line, hits[0].column, hits[0].char) == (1, 2, char)
    assert f"U+{ord(char):04X}" in hits[0].format()


def test_utf8_bom_at_start_of_file_fails_lint(tmp_path):
    (tmp_path / "bom.md").write_bytes("\ufeff# Heading\n".encode("utf-8"))
    hits = ps5_lint.lint([tmp_path])
    assert len(hits) == 1
    assert (hits[0].line, hits[0].column, hits[0].char) == (1, 1, "\ufeff")
    assert ps5_lint.main(["--root", str(tmp_path)]) == 1


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-16-be"])
def test_utf16_markdown_file_with_em_dash_fails_lint(tmp_path, encoding):
    md = tmp_path / f"doc_{encoding}.md"
    md.write_bytes(f"# Notes\nbad {EM_DASH} dash\n".encode(encoding))
    hits = ps5_lint.scan_file(md)
    assert len(hits) >= 1
    assert ps5_lint.main(["--root", str(tmp_path)]) == 1


def test_non_utf8_text_file_reports_violation_while_binary_allowlist_passes(tmp_path):
    txt = tmp_path / "latin1.txt"
    txt.write_bytes(b"caf\xe9 invalid utf8\n")
    hits = ps5_lint.lint([tmp_path])
    assert len(hits) == 1
    assert "non-UTF-8" in hits[0].format()
    assert ps5_lint.main(["--root", str(tmp_path)]) == 1

    # Explicit binary suffixes are still skipped cleanly.
    txt.unlink()
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
    (tmp_path / "model.onnx").write_bytes(b"\xff\xfe\x80\x00")
    assert ps5_lint.lint([tmp_path]) == []
    assert ps5_lint.main(["--root", str(tmp_path)]) == 0
