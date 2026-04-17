"""Inspect the OCR cache: counts, samples, and per-source breakdown.

Usage::

    python -m scripts.ocr_inspect                       # summary
    python -m scripts.ocr_inspect --samples 5           # show 5 random text samples
    python -m scripts.ocr_inspect --grep python         # show files whose OCR text matches
    python -m scripts.ocr_inspect --file path\to\image.png   # show OCR for a specific image
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SETTINGS  # noqa: E402
from src.dataset import iter_resume_files  # noqa: E402
from src.extraction import SUPPORTED_IMAGE_EXTS  # noqa: E402
from src.ocr_cache import OCR_CACHE_DIR, cache_path_for, read_cached  # noqa: E402


def gather():
    items = []
    for p, cat in iter_resume_files():
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            items.append((p, cat))
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=0, help="Show N random cached samples.")
    ap.add_argument("--grep", type=str, default=None, help="Substring to search for inside cached text.")
    ap.add_argument("--file", type=str, default=None, help="Show cached OCR for a specific image path.")
    ap.add_argument("--max-chars", type=int, default=600, help="Snippet length per sample.")
    ap.add_argument("--list-empty", action="store_true", help="List images whose OCR returned empty text.")
    args = ap.parse_args()

    if args.file:
        p = Path(args.file)
        cp = cache_path_for(p)
        print(f"image : {p}")
        print(f"cache : {cp}")
        print(f"exists: {cp.exists()}")
        if cp.exists():
            text = read_cached(p) or ""
            print(f"chars : {len(text)}")
            print("-" * 60)
            print(text[: args.max_chars])
        return 0

    items = gather()
    total = len(items)
    cached = []
    empty = []
    for p, cat in items:
        cp = cache_path_for(p)
        if cp.exists():
            text = (read_cached(p) or "").strip()
            if text:
                cached.append((p, cat, len(text)))
            else:
                empty.append((p, cat))

    print(f"Total images discovered : {total}")
    print(f"Cached (any)            : {len(cached) + len(empty)}")
    print(f"Cached non-empty        : {len(cached)}")
    print(f"Cached empty            : {len(empty)}")
    print(f"Pending                 : {total - len(cached) - len(empty)}")
    print(f"Cache dir               : {OCR_CACHE_DIR}")

    # Per source folder breakdown.
    print("\nPer-source progress:")
    for src in SETTINGS.raw_sources:
        s_total = sum(1 for p, _ in items if str(p).startswith(str(src)))
        s_done = sum(1 for p, _, _ in cached if str(p).startswith(str(src)))
        s_done += sum(1 for p, _ in empty if str(p).startswith(str(src)))
        print(f"  {src.name:30s}  {s_done:>5d} / {s_total:<5d}")

    # Per category top-10 by cached count.
    by_cat: dict[str, int] = {}
    for _, cat, _ in cached:
        by_cat[cat] = by_cat.get(cat, 0) + 1
    if by_cat:
        print("\nTop categories by cached non-empty:")
        for cat, n in sorted(by_cat.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cat:30s}  {n}")

    if args.list_empty and empty:
        print(f"\nFirst 20 empty-OCR files (of {len(empty)}):")
        for p, cat in empty[:20]:
            print(f"  [{cat}] {p}")

    if args.grep:
        needle = args.grep.lower()
        hits = []
        for p, cat, _ in cached:
            text = (read_cached(p) or "").lower()
            if needle in text:
                hits.append((p, cat))
        print(f"\nMatches for '{args.grep}': {len(hits)}")
        for p, cat in hits[:10]:
            print(f"  [{cat}] {p}")

    if args.samples > 0 and cached:
        picks = random.sample(cached, min(args.samples, len(cached)))
        print(f"\n--- {len(picks)} random samples ---")
        for p, cat, n in picks:
            text = (read_cached(p) or "").strip().replace("\n", " ⏎ ")
            snippet = text[: args.max_chars]
            print(f"\n[{cat}]  {p.name}  ({n} chars)")
            print(f"  {snippet}{' ...' if len(text) > args.max_chars else ''}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
