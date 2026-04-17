"""Walk the configured raw resume sources and yield (path, category) pairs."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Tuple

from .config import SETTINGS, canonical_category
from .extraction import is_supported

logger = logging.getLogger(__name__)


def iter_resume_files(
    sources: List[Path] | None = None,
    *,
    per_category_limit: int | None = None,
) -> Iterator[Tuple[Path, str]]:
    """Yield (file_path, canonical_category) tuples from the dataset.

    ``per_category_limit`` caps the number of files per (source, category) bucket
    — useful for quick smoke tests so we don't process all 21,000 files.
    """
    sources = sources or SETTINGS.raw_sources
    counts: dict[tuple[str, str], int] = {}
    for src in sources:
        if not src.exists():
            logger.warning("Source not found, skipping: %s", src)
            continue
        # Each immediate subfolder of `src` is a category folder.
        for cat_dir in sorted(p for p in src.iterdir() if p.is_dir()):
            category = canonical_category(cat_dir.name)
            for file in sorted(cat_dir.iterdir()):
                if not file.is_file() or not is_supported(file):
                    continue
                key = (str(src), category)
                if per_category_limit is not None and counts.get(key, 0) >= per_category_limit:
                    break
                counts[key] = counts.get(key, 0) + 1
                yield file, category


def summarize(sources: List[Path] | None = None) -> dict[str, int]:
    """Return a {category: count} summary across all sources."""
    summary: dict[str, int] = {}
    for _, category in iter_resume_files(sources):
        summary[category] = summary.get(category, 0) + 1
    return summary
