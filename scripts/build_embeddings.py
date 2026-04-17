"""Build sentence-transformer embeddings for every resume in the dataset.

Usage (from repo root):
    python -m scripts.build_embeddings --limit 25     # quick smoke test
    python -m scripts.build_embeddings                # full build
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

# Ensure project root on sys.path when run as `python scripts/build_embeddings.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SETTINGS  # noqa: E402
from src.dataset import iter_resume_files  # noqa: E402
from src.embeddings import encode_texts  # noqa: E402
from src.extraction import extract_text  # noqa: E402
from src.preprocessing import clean_text, truncate_words  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_embeddings")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build resume embeddings")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max files per (source, category) bucket. Useful for smoke tests.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=1024,
        help="Truncate resumes to this many words before encoding.",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip image (OCR) files entirely — much faster.",
    )
    args = parser.parse_args()

    if args.skip_ocr:
        SETTINGS.enable_ocr = False
    else:
        # Auto-disable OCR when tesseract binary is missing, so we don't silently
        # turn every image into a short/empty row.
        from src.extraction import ocr_available  # noqa: WPS433
        ok, reason = ocr_available()
        if not ok:
            logger.warning(
                "OCR disabled automatically: %s. Image resumes will be skipped. "
                "Pass --skip-ocr to silence, or install tesseract to enable.",
                reason,
            )
            SETTINGS.enable_ocr = False
            args.skip_ocr = True

    filenames: List[str] = []
    categories: List[str] = []
    texts: List[str] = []

    logger.info("Scanning sources: %s", [str(s) for s in SETTINGS.raw_sources])
    files = list(iter_resume_files(per_category_limit=args.limit))
    logger.info("Found %d candidate files", len(files))

    skipped_short = 0
    skipped_failed = 0
    skipped_ocr = 0
    per_ext_kept: dict[str, int] = {}
    per_ext_short: dict[str, int] = {}

    for file_path, category in tqdm(files, desc="Extract"):
        ext = file_path.suffix.lower()
        if args.skip_ocr and ext in {
            ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff",
        }:
            skipped_ocr += 1
            continue
        try:
            raw = extract_text(file_path) or ""
        except Exception as e:  # noqa: BLE001
            logger.debug("Extraction failed for %s: %s", file_path, e)
            skipped_failed += 1
            continue
        cleaned = clean_text(raw)
        if len(cleaned) < SETTINGS.min_text_length:
            skipped_short += 1
            per_ext_short[ext] = per_ext_short.get(ext, 0) + 1
            continue
        cleaned = truncate_words(cleaned, args.max_words)
        filenames.append(str(file_path))
        categories.append(category)
        texts.append(cleaned)
        per_ext_kept[ext] = per_ext_kept.get(ext, 0) + 1

    logger.info(
        "Kept %d resumes (skipped: %d short, %d failed, %d image-ocr-skipped)",
        len(texts), skipped_short, skipped_failed, skipped_ocr,
    )
    logger.info("  kept by ext : %s", per_ext_kept)
    if per_ext_short:
        logger.info("  short by ext: %s", per_ext_short)

    if not texts:
        logger.error("No usable resumes found. Aborting.")
        return 1

    logger.info("Encoding %d resumes with %s ...", len(texts), SETTINGS.embedding_model)
    embeddings = encode_texts(texts, show_progress=True, normalize=True)

    SETTINGS.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(SETTINGS.embeddings_path, embeddings)
    with open(SETTINGS.metadata_path, "w", encoding="utf-8") as f:
        json.dump({"filenames": filenames, "categories": categories}, f)
    with open(SETTINGS.texts_path, "w", encoding="utf-8") as f:
        json.dump(texts, f)

    logger.info("Saved:")
    logger.info("  embeddings -> %s  (shape=%s)", SETTINGS.embeddings_path, embeddings.shape)
    logger.info("  metadata   -> %s", SETTINGS.metadata_path)
    logger.info("  texts      -> %s", SETTINGS.texts_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
