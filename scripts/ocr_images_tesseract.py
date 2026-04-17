"""Parallel CPU OCR using Tesseract — much faster than the GPU EasyOCR pipeline
on this dataset because Tesseract is light, parallelizable, and you have 20 cores.

Resumable: same on-disk cache as scripts/ocr_images.py
(``data/ocr_cache/<xx>/<sha1>.txt``). Re-running skips already-cached images.

Usage::

    python -m scripts.ocr_images_tesseract                 # all images, default workers
    python -m scripts.ocr_images_tesseract --workers 12    # explicit worker count
    python -m scripts.ocr_images_tesseract --limit 5       # smoke test (5 per category)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
logger = logging.getLogger("ocr_tesseract")

DEFAULT_TESSERACT = SETTINGS.tesseract_cmd or r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Module-level worker — each child process initialises Tesseract once.
_WORKER_TESS_PATH: str | None = None
_WORKER_MAX_DIM: int = 2000


def _init_worker(tesseract_path: str, max_dim: int) -> None:
    global _WORKER_TESS_PATH, _WORKER_MAX_DIM
    _WORKER_TESS_PATH = tesseract_path
    _WORKER_MAX_DIM = max_dim
    # Limit BLAS threads inside each worker so 12 workers don't fight for cores.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import pytesseract  # noqa: F401
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


def _ocr_one(path_str: str) -> Tuple[str, int, str]:
    """Run OCR on a single image. Returns (path, char_count, error_msg)."""
    from PIL import Image
    import pytesseract

    p = Path(path_str)
    try:
        with Image.open(p) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            w, h = img.size
            scale = min(1.0, _WORKER_MAX_DIM / max(w, h))
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            text = (pytesseract.image_to_string(img) or "").strip()
        write_cached(p, text)
        return path_str, len(text), ""
    except Exception as e:  # noqa: BLE001
        # Cache an empty result so we don't retry forever.
        write_cached(p, "")
        return path_str, 0, str(e)


def _gather_images(per_cat_limit: int | None) -> List[Path]:
    out: List[Path] = []
    for path, _ in iter_resume_files(per_category_limit=per_cat_limit):
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            out.append(path)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Resumable parallel Tesseract OCR.")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 4),
                    help="Number of parallel worker processes (default: cpu_count-4).")
    ap.add_argument("--max-dim", type=int, default=2000,
                    help="Resize images so the largest side is <= this many pixels (default 2000).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Per-category file cap (smoke test).")
    ap.add_argument("--force", action="store_true",
                    help="Re-OCR even if a cache file already exists.")
    ap.add_argument("--tesseract", type=str, default=DEFAULT_TESSERACT,
                    help=f"Path to tesseract.exe (default: {DEFAULT_TESSERACT}).")
    args = ap.parse_args()

    if not Path(args.tesseract).exists():
        logger.error("Tesseract binary not found at %s", args.tesseract)
        return 2

    images = _gather_images(args.limit)
    logger.info("Discovered %d images.", len(images))

    if not args.force:
        pending = [p for p in images if not cache_path_for(p).exists()]
    else:
        pending = images
    skipped_cached = len(images) - len(pending)
    logger.info("%d already cached, %d to OCR with %d workers.",
                skipped_cached, len(pending), args.workers)

    if not pending:
        logger.info("Nothing to do. Cache is up to date.")
        return 0

    start = time.time()
    done = 0
    empty = 0
    failed = 0

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.tesseract, args.max_dim),
    ) as pool:
        futures = {pool.submit(_ocr_one, str(p)): p for p in pending}
        try:
            with tqdm(total=len(futures), desc=f"OCR (tess x{args.workers})") as bar:
                for fut in as_completed(futures):
                    path_str, n_chars, err = fut.result()
                    done += 1
                    if err:
                        failed += 1
                    if n_chars == 0:
                        empty += 1
                    bar.update(1)
                    if done % 100 == 0:
                        elapsed = time.time() - start
                        rate = done / elapsed if elapsed else 0
                        bar.set_postfix({"img/s": f"{rate:.1f}", "empty": empty, "fail": failed})
        except KeyboardInterrupt:
            logger.warning("Interrupted — workers will finish in-flight items, progress saved.")
            for f in futures:
                f.cancel()

    elapsed = time.time() - start
    rate = done / elapsed if elapsed else 0
    logger.info(
        "Done. processed=%d (empty=%d, failed=%d) skipped_cached=%d  in %.1fs  (%.1f img/s)",
        done, empty, failed, skipped_cached, elapsed, rate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
