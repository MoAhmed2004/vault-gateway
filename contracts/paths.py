"""Synced-folder detection for local-disk-only storage (IP-8). Pure stdlib.

Sync clients corrupt or lock SQLite WAL files, so log databases must never sit
inside a synced folder. Matching is by path component: long vendor names match
as substrings, while the short names `box` and `box sync` must match a whole
component so ordinary names such as `sandbox` do not trigger.
"""

from __future__ import annotations

from pathlib import Path

SYNCED_SUBSTRING_MARKERS = (
    "onedrive",
    "dropbox",
    "google drive",
    "googledrive",
    "iclouddrive",
    "mobile documents",
    "com~apple~clouddocs",
)
SYNCED_EXACT_MARKERS = ("box", "box sync")
SYNCED_FOLDER_MARKERS = SYNCED_SUBSTRING_MARKERS + SYNCED_EXACT_MARKERS


def synced_folder_component(path: Path) -> str | None:
    """Return the first path component that names a cloud-synced folder, else None."""
    for part in Path(path).parts:
        lowered = part.lower()
        if lowered in SYNCED_EXACT_MARKERS:
            return part
        if any(marker in lowered for marker in SYNCED_SUBSTRING_MARKERS):
            return part
    return None
