"""Build embeddings for the real-world job-description CSV.

Usage:
    python -m scripts.build_job_index
    python -m scripts.build_job_index --limit 500     # quick subset
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SETTINGS  # noqa: E402
from src.jobs import build_job_index, load_jd_dataframe  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build JD embeddings")
    parser.add_argument("--csv", type=str, default=None, help="Override path to JD CSV or folder")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only embed the first N rows"
    )
    parser.add_argument(
        "--filter-by-resume-categories",
        action="store_true",
        help=(
            "Keep only JD rows whose title (or source-CSV filename) matches a "
            "canonical resume category. Greatly reduces the JD index size and "
            "improves the relevance of recruiter / applicant matches."
        ),
    )
    parser.add_argument(
        "--per-category-limit",
        type=int,
        default=None,
        help="When filtering by resume category, cap rows per category.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Encoder batch size (default: SETTINGS.embedding_batch_size).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else SETTINGS.jd_csv_path

    if args.limit is not None:
        # Write a truncated CSV to a cache location and embed that.
        df = load_jd_dataframe(
            csv_path,
            filter_to_resume_categories=args.filter_by_resume_categories,
            per_category_limit=args.per_category_limit,
        ).head(args.limit)
        tmp_csv = SETTINGS.jd_embeddings_path.parent / "_limited_jd.csv"
        df.rename(columns={"title": "Job Title", "description": "Job Description"}).to_csv(
            tmp_csv, index=False
        )
        csv_path = tmp_csv
        # The temp CSV no longer carries the category column, so re-derive
        # categories during the actual build below.

    count = build_job_index(
        csv_path,
        filter_to_resume_categories=args.filter_by_resume_categories,
        per_category_limit=args.per_category_limit,
        batch_size=args.batch_size,
    )
    print(f"[OK] Indexed {count:,} job descriptions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
