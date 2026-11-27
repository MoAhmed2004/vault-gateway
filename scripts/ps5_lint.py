"""PS-5 linter: fail on emojis, em/en dashes, control/format chars, and non-UTF-8 text files.

Usage:
    python scripts/ps5_lint.py [--root DIR] [PATH ...]

Exit code 0 when clean, 1 when any banned character or non-UTF-8 text file is
found. The banned characters are written as escapes in this file so the linter
passes on itself.

Banned: emojis and pictographs (PS-5 pattern 3), em dashes (PS-5 pattern 16),
en dashes ([Project Decision, extends PS-5]: spec pattern 16 names em dashes
only; this project bans en dashes as well), ASCII control characters
(`ord(ch) < 0x20` except `\n` and `\r`, plus `0x7F` DEL), Unicode format
characters (`unicodedata.category(ch) == "Cf"`, including BOM `U+FEFF`,
zero-width spaces `U+200B`, and bidirectional controls `U+202E`, `U+2066`,
etc.), and any file outside `BINARY_SUFFIXES` that is not valid UTF-8. Plain
utilitarian status glyphs (check mark, ballot cross, black circle, black
square, box-drawing, arrows) are allowed for the W15 SOC console.

Curated multilingual data (`CURATED_MULTILINGUAL_DIRS`): `data/translations/`
(the pre-translated Arabic, French, Spanish table, §2.7, §3.4) and
`data/benign/` (the hand-curated D-BEN-2 set, which includes Arabic
translation requests, §2.2). These files are dataset text, not authored code
or prose, so the PS-5 typography checks (emoji, em/en dash) do not apply to
them. They are still scanned for integrity: they must be valid UTF-8, contain
no ASCII control characters, and contain no `Cf` format character except the
script-required marks in `CURATED_ALLOWED_FORMAT_CHARS` (ZWNJ `U+200C`, LRM
`U+200E`, RLM `U+200F`, Arabic letter mark `U+061C`). Every other `Cf`
character (BOM, `U+200B`, `U+200D`, bidi embeddings, overrides and isolates
`U+202A` to `U+202E` and `U+2066` to `U+2069`, the tag block) still fails
there, and the allowlist applies nowhere else.

Skipped: uncurated dataset and cache directories (`data/raw/`, `data/cache/`),
downloaded model artifacts (`models/`, whose third-party tokenizer vocabularies
contain emoji and dashes as data), virtualenv, VCS, results, ChromaDB, binary
and PDF files (`BINARY_SUFFIXES`), and any file larger than 5 MiB.

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
# These hold uncurated or generated data that is never tracked.
SKIP_REL_DIRS = frozenset(
    {
        "data/raw",
        "data/cache",
        "results",
        # Downloaded third-party model artifacts (scripts/fetch_models.py); git-ignored, pinned by SHA-256.
        "models",
    }
)

# Tracked, hand-curated multilingual dataset directories (relative to the scan root).
CURATED_MULTILINGUAL_DIRS: tuple[str, ...] = (
    "data/translations",
    "data/benign",
)

# Format (Cf) characters that Arabic and Persian text legitimately needs.
# Kept identical to `contracts.datasets.CURATED_ALLOWED_FORMAT_CHARS` (tested).
CURATED_ALLOWED_FORMAT_CHARS = frozenset(
    {
        "\u200c",  # ZERO WIDTH NON-JOINER (Persian, Arabic morphology)
        "\u200e",  # LEFT-TO-RIGHT MARK (English marker text inside Arabic)
        "\u200f",  # RIGHT-TO-LEFT MARK
        "\u061c",  # ARABIC LETTER MARK
    }
)

MAX_FILE_BYTES = 5 * 1024 * 1024

BINARY_SUFFIXES = frozenset(
    {
        # documents and images
        ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".bmp",
        ".docx", ".xlsx", ".pptx", ".mp4",
        # model weights and serialized arrays or objects
        ".onnx", ".pt", ".safetensors", ".ckpt", ".bin", ".dat", ".npy", ".npz",
        ".pkl", ".pickle", ".joblib", ".parquet", ".feather", ".h5", ".gguf", ".tflite",
        # databases
        ".db", ".sqlite", ".sqlite3",
        # archives
        ".zip", ".gz", ".tar", ".tgz", ".whl", ".7z", ".rar", ".xz", ".bz2",
        # compiled code
        ".pyc", ".pyd", ".so", ".dll", ".exe",
        # fonts
        ".woff", ".woff2", ".ttf", ".otf",
    }
)


@dataclass(frozen=True)
class Hit:
    path: Path
    line: int
    column: int
    char: str
    reason: str | None = None

    def format(self) -> str:
        if self.reason is not None:
            return f"{self.path}:{self.line}:{self.column}: {self.reason}"
        code = ord(self.char)
        if code < 0x20 or code == 0x7F:
            default_name = "CONTROL"
        elif unicodedata.category(self.char) == "Cf":
            default_name = "FORMAT"
        else:
            default_name = "UNKNOWN"
        name = unicodedata.name(self.char, default_name)
        return f"{self.path}:{self.line}:{self.column}: U+{code:04X} {name}"


def _is_control(char: str) -> bool:
    code = ord(char)
    return (code < 0x20 and char not in ("\n", "\r")) or code == 0x7F


def is_banned(char: str) -> bool:
    """Strict rule applied to every scanned file outside the curated multilingual directories."""
    if char in (EM_DASH, EN_DASH):
        return True
    if _is_control(char):
        return True
    if unicodedata.category(char) == "Cf":
        return True
    code = ord(char)
    return any(low <= code <= high for low, high in EMOJI_RANGES)


def is_banned_in_curated_data(char: str) -> bool:
    """Integrity rule for curated multilingual data: controls and every non-allowlisted Cf character."""
    if _is_control(char):
        return True
    return unicodedata.category(char) == "Cf" and char not in CURATED_ALLOWED_FORMAT_CHARS


def scan_text(text: str, path: Path, *, curated: bool = False) -> list[Hit]:
    banned = is_banned_in_curated_data if curated else is_banned
    hits: list[Hit] = []
    line_no = 1
    col_no = 0
    for char in text:
        if char == "\n":
            line_no += 1
            col_no = 0
            continue
        if char == "\r":
            continue
        col_no += 1
        if banned(char):
            hits.append(Hit(path, line_no, col_no, char))
    return hits


def scan_file(path: Path, *, curated: bool = False) -> list[Hit]:
    if path.suffix.lower() in BINARY_SUFFIXES:
        return []
    if path.stat().st_size > MAX_FILE_BYTES:
        return []
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [
            Hit(
                path,
                1,
                exc.start + 1,
                "",
                reason=f"non-UTF-8 file ({exc.reason} at byte {exc.start})",
            )
        ]
    return scan_text(text, path, curated=curated)


def is_curated_multilingual(path: Path, base: Path) -> bool:
    """True when `path` lies inside one of `CURATED_MULTILINGUAL_DIRS` under `base`."""
    try:
        rel = path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return False
    return any(rel.startswith(f"{prefix}/") for prefix in CURATED_MULTILINGUAL_DIRS)


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


def lint(paths: Iterable[Path], *, base: Path | None = None) -> list[Hit]:
    """Lint files and directories. Curated-directory detection is relative to `base`.

    A directory argument is its own base unless `base` is given. A file argument
    is curated only when it lies inside a curated directory under `base`, so
    the strict rule is the default.
    """
    hits: list[Hit] = []
    for path in paths:
        if path.is_dir():
            scan_base = path
            if base is not None and path.resolve().is_relative_to(base.resolve()):
                scan_base = base
            for file in iter_files(path.resolve(), scan_base.resolve()):
                hits.extend(scan_file(file, curated=is_curated_multilingual(file, scan_base)))
        elif path.is_file():
            curated = base is not None and is_curated_multilingual(path, base)
            hits.extend(scan_file(path, curated=curated))
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PS-5 banned character linter")
    parser.add_argument("paths", nargs="*", type=Path, help="files or directories (default: --root)")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)
    targets = args.paths or [args.root]
    hits = lint(targets, base=args.root)
    for hit in hits:
        print(hit.format())
    if hits:
        print(f"PS-5 lint: FAIL ({len(hits)} violation(s))")
        return 1
    print("PS-5 lint: OK (0 violations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
