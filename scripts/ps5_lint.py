"""PS-5 linter: fail on emojis, em dashes, and en dashes in project text files.

Usage:
    python scripts/ps5_lint.py [--root DIR] [PATH ...]

Exit code 0 when clean, 1 when any banned character is found. The banned
characters are written as escapes in this file so the linter passes on itself.

Banned: emojis and pictographs (PS-5 pattern 3), em dashes (PS-5 pattern 16),
and en dashes ([Project Decision, extends PS-5]: spec pattern 16 names em dashes
only; this project bans en dashes as well). Plain utilitarian status glyphs
(check mark, ballot cross, black circle, black square, box-drawing, arrows) are
allowed for the W15 SOC console.

Skipped: dataset and cache directories, virtualenv, VCS, results, ChromaDB,
`data/translations/` (hand-curated Arabic, French, Spanish text), binary and
PDF files, and any file larger than 5 MiB.

TODO before W15: add text-level UI pattern checks for `linear-gradient`,
`backdrop-filter`, `Inter`, `Space Grotesk`, `Instrument Serif`, and `lucide`.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

EM_DASH = "\u2014"
EN_DASH = "\u2013"  # [Project Decision, extends PS-5]

# Inclusive code point ranges treated as emoji (PS-5 pattern 3).
# Dingbats U+2700-U+27BF are banned except the plain check/cross family
# U+2713-U+2718, which the W15 console may use as status glyphs. U+25A0 and
# U+25CF are not in any banned range.
EMOJI_RANGES: tuple[tuple[int, int], ...] = (
    (0x1F000, 0x1FAFF),  # mahjong, cards, enclosed, pictographs, emoticons, transport, symbols ext
    (0x2600, 0x26FF),  # misc symbols (weather, zodiac, warning sign, etc.)
    (0x2700, 0x2712),  # dingbats before the check/cross family
    (0x2719, 0x27BF),  # dingbats after the check/cross family
    (0x25AA, 0x25AB),  # small squares (emoji)
    (0x25B6, 0x25B6),  # play button
    (0x25C0, 0x25C0),  # reverse button
    (0x25FB, 0x25FE),  # medium squares (emoji)
    (0x2B05, 0x2B07),  # emoji arrows
    (0x2B1B, 0x2B1C),  # large squares
    (0x2B50, 0x2B50),  # star
    (0x2B55, 0x2B55),  # heavy large circle
    (0x231A, 0x231B),  # watch, hourglass
    (0x23E9, 0x23F3),  # media controls, alarm clock
    (0x23F8, 0x23FA),
    (0xFE0F, 0xFE0F),  # emoji presentation selector
    (0xE0020, 0xE007F),  # tag characters used in flag sequences
)

# Directories skipped by name at any depth.
SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "results",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".ipynb_checkpoints",
        "node_modules",
        "chroma_db",
    }
)

# Directories skipped by path relative to the scan root (posix separators).
# data/translations holds hand-curated multilingual text and is excluded.
SKIP_REL_DIRS = frozenset(
    {
        "data/raw",
        "data/cache",
        "data/translations",
        "results",
    }
)

MAX_FILE_BYTES = 5 * 1024 * 1024

BINARY_SUFFIXES = frozenset(
    {
        ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".bmp",
        ".onnx", ".pt", ".safetensors", ".ckpt", ".bin", ".npy", ".npz",
        ".db", ".sqlite", ".sqlite3", ".zip", ".gz", ".tar", ".whl",
        ".pyc", ".pyd", ".so", ".dll", ".exe", ".woff", ".woff2", ".ttf",
    }
)


@dataclass(frozen=True)
class Hit:
    path: Path
    line: int
    column: int
    char: str

    def format(self) -> str:
        name = unicodedata.name(self.char, "UNKNOWN")
        return f"{self.path}:{self.line}:{self.column}: U+{ord(self.char):04X} {name}"


def is_banned(char: str) -> bool:
    if char in (EM_DASH, EN_DASH):
        return True
    code = ord(char)
    return any(low <= code <= high for low, high in EMOJI_RANGES)


def scan_text(text: str, path: Path) -> list[Hit]:
    hits: list[Hit] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for col_no, char in enumerate(line, start=1):
            if is_banned(char):
                hits.append(Hit(path, line_no, col_no, char))
    return hits


def scan_file(path: Path) -> list[Hit]:
    if path.suffix.lower() in BINARY_SUFFIXES:
        return []
    if path.stat().st_size > MAX_FILE_BYTES:
        return []
    data = path.read_bytes()
    if b"\x00" in data:
        return []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return []
    return scan_text(text, path)


def iter_files(root: Path, base: Path | None = None) -> Iterator[Path]:
    base = root if base is None else base
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            if entry.name in SKIP_DIRS:
                continue
            if entry.relative_to(base).as_posix() in SKIP_REL_DIRS:
                continue
            yield from iter_files(entry, base)
        elif entry.is_file():
            yield entry


def lint(paths: Iterable[Path]) -> list[Hit]:
    hits: list[Hit] = []
    for path in paths:
        if path.is_dir():
            for file in iter_files(path):
                hits.extend(scan_file(file))
        elif path.is_file():
            hits.extend(scan_file(path))
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PS-5 banned character linter")
    parser.add_argument("paths", nargs="*", type=Path, help="files or directories (default: --root)")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)
    targets = args.paths or [args.root]
    hits = lint(targets)
    for hit in hits:
        print(hit.format())
    if hits:
        print(f"PS-5 lint: FAIL ({len(hits)} banned character(s))")
        return 1
    print("PS-5 lint: OK (0 banned characters)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
