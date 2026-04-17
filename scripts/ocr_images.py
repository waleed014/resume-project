"""Run OCR on every image resume and cache the text on disk.

Resumable: each image's text is written to a deterministic sidecar file in
``data/ocr_cache/``. Re-running the script skips any image whose cache
file already exists, so you can stop at any time (Ctrl+C) and resume later.

GPU acceleration: uses EasyOCR with CUDA when available (RTX/GTX cards).
Falls back to Tesseract if EasyOCR is not installed and the Tesseract
binary is on PATH.

Usage::

    python -m scripts.ocr_images                         # OCR every image
    python -m scripts.ocr_images --limit 50              # 50 per category (smoke test)
    python -m scripts.ocr_images --engine tesseract      # force tesseract
    python -m scripts.ocr_images --no-gpu                # CPU only
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SETTINGS  # noqa: E402
from src.dataset import iter_resume_files  # noqa: E402
from src.extraction import SUPPORTED_IMAGE_EXTS  # noqa: E402
from src.ocr_cache import cache_path_for, write_cached  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ocr_images")


# ---- engines -----------------------------------------------------------------

def _make_easyocr(use_gpu: bool, max_dim: int, paragraph: bool):
    import easyocr  # type: ignore
    import numpy as np
    import torch
    from PIL import Image

    gpu = bool(use_gpu and torch.cuda.is_available())
    if use_gpu and not gpu:
        logger.warning("CUDA not available; falling back to CPU for EasyOCR.")
    logger.info(
        "Initializing EasyOCR (gpu=%s, max_dim=%d, paragraph=%s) ...",
        gpu, max_dim, paragraph,
    )
    reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def _load(path: Path):
        with Image.open(path) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            w, h = img.size
            scale = min(1.0, max_dim / max(w, h))
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            return np.asarray(img)

    def run(path: Path) -> str:
        try:
            arr = _load(path)
            lines = reader.readtext(
                arr,
                detail=0,
                paragraph=paragraph,
                canvas_size=max_dim,
            )
            return "\n".join(lines).strip()
        except Exception as e:  # noqa: BLE001
            logger.debug("easyocr failed on %s: %s", path, e)
            return ""

    return run, ("easyocr-gpu" if gpu else "easyocr-cpu")


def _make_tesseract():
    import os
    import pytesseract
    from PIL import Image

    tc = SETTINGS.tesseract_cmd
    if tc and os.path.exists(tc):
        pytesseract.pytesseract.tesseract_cmd = tc
    logger.info("Initializing Tesseract (cmd=%s) ...", pytesseract.pytesseract.tesseract_cmd)

    def run(path: Path) -> str:
        try:
            with Image.open(path) as img:
                return (pytesseract.image_to_string(img) or "").strip()
        except Exception as e:  # noqa: BLE001
            logger.debug("tesseract failed on %s: %s", path, e)
            return ""

    return run, "tesseract"


def _select_engine(name: str, use_gpu: bool, max_dim: int, paragraph: bool):
    if name == "easyocr":
        return _make_easyocr(use_gpu, max_dim, paragraph)
    if name == "tesseract":
        return _make_tesseract()
    # auto: prefer easyocr (GPU), fall back to tesseract
    try:
        return _make_easyocr(use_gpu, max_dim, paragraph)
    except ImportError:
        logger.info("easyocr not installed; trying tesseract.")
        return _make_tesseract()


# ---- main --------------------------------------------------------------------

def _gather_images(per_cat_limit: int | None) -> List[Tuple[Path, str]]:
    out: List[Tuple[Path, str]] = []
    for path, cat in iter_resume_files(per_category_limit=per_cat_limit):
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            out.append((path, cat))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable OCR of all image resumes.")
    parser.add_argument("--engine", choices=["auto", "easyocr", "tesseract"], default="auto")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU even if CUDA is present.")
    parser.add_argument("--limit", type=int, default=None, help="Per-category file cap (smoke test).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-OCR even if cache file already exists.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1600,
        help="Resize images so the largest side is <= this many pixels (default 1600).",
    )
    parser.add_argument(
        "--no-paragraph",
        action="store_true",
        help="Skip EasyOCR's paragraph reflow (much faster, slightly noisier).",
    )
    args = parser.parse_args()

    images = _gather_images(args.limit)
    logger.info("Discovered %d images across %d sources.", len(images), len(SETTINGS.raw_sources))

    if not args.force:
        pending = [(p, c) for p, c in images if not cache_path_for(p).exists()]
    else:
        pending = images
    skipped_cached = len(images) - len(pending)
    logger.info("%d already cached, %d to OCR.", skipped_cached, len(pending))

    if not pending:
        logger.info("Nothing to do. Cache is up to date.")
        return 0

    run, engine_name = _select_engine(
        args.engine,
        use_gpu=not args.no_gpu,
        max_dim=args.max_dim,
        paragraph=not args.no_paragraph,
    )

    start = time.time()
    done = 0
    empty = 0
    failed = 0
    try:
        for path, _cat in tqdm(pending, desc=f"OCR ({engine_name})"):
            try:
                text = run(path)
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning("OCR failed for %s: %s", path, e)
                failed += 1
                # Still write empty so we don't loop on it forever.
                write_cached(path, "")
                continue
            write_cached(path, text)
            done += 1
            if not text:
                empty += 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user — progress saved. Re-run to resume.")

    elapsed = time.time() - start
    rate = done / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Done. processed=%d (empty=%d, failed=%d) skipped_cached=%d  in %.1fs  (%.2f img/s)",
        done, empty, failed, skipped_cached, elapsed, rate,
    )
    logger.info("Cache directory: %s", cache_path_for(pending[0][0]).parent.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
