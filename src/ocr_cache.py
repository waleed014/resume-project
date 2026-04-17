"""Shared OCR cache helpers.

Each image gets a deterministic on-disk text sidecar so OCR runs are resumable
and the heavy (GPU) OCR is done exactly once per file.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

from .config import DATA_DIR

OCR_CACHE_DIR = DATA_DIR / "ocr_cache"
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_path_for(image_path: str | Path) -> Path:
    """Deterministic cache file for an absolute image path."""
    p = str(Path(image_path).resolve()).lower()
    h = hashlib.sha1(p.encode("utf-8")).hexdigest()
    # Use first two chars as a shard to avoid one mega-folder.
    sub = OCR_CACHE_DIR / h[:2]
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{h}.txt"


def read_cached(image_path: str | Path) -> Optional[str]:
    """Return cached OCR text (possibly empty) if it exists, else None."""
    cp = cache_path_for(image_path)
    if not cp.exists():
        return None
    try:
        return cp.read_text(encoding="utf-8")
    except OSError:
        return None


def write_cached(image_path: str | Path, text: str) -> Path:
    cp = cache_path_for(image_path)
    cp.write_text(text or "", encoding="utf-8")
    return cp


def has_cache(image_path: str | Path) -> bool:
    return cache_path_for(image_path).exists()
