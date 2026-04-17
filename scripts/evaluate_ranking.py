"""Evaluate the ranking pipeline using category labels as weak relevance signal.

For each held-out resume, we treat all other resumes in the same category as
"relevant". We query with its text, retrieve top-K from the index, and compute
Precision@K / Recall@K / NDCG@K / MAP.

Usage:
    python -m scripts.evaluate_ranking --queries 200 --ks 1,5,10,20
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.config import SETTINGS  # noqa: E402
from src.evaluation import summarize  # noqa: E402
from src.ranking import CandidateIndex  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evaluate_ranking")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate resume ranking")
    parser.add_argument("--queries", type=int, default=200, help="Number of sampled query resumes")
    parser.add_argument(
        "--ks", type=str, default="1,5,10,20", help="Comma-separated K values",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ks = tuple(int(k) for k in args.ks.split(","))
    rng = random.Random(args.seed)

    idx = CandidateIndex.load()
    if not idx.texts:
        log.error("resume_texts.json not found; rebuild embeddings first.")
        return 1

    n = len(idx)
    log.info("Index has %d resumes across %d categories", n, len(set(idx.categories)))

    sample_size = min(args.queries, n)
    sample_idx = rng.sample(range(n), sample_size)

    all_relevant, all_retrieved = [], []
    emb = idx.embeddings

    for qi in sample_idx:
        q_vec = emb[qi]
        scores = emb @ q_vec
        scores[qi] = -np.inf  # exclude the query itself
        order = np.argsort(-scores)[: max(ks)]
        retrieved = [idx.filenames[i] for i in order]
        relevant = [
            idx.filenames[j]
            for j in range(n)
            if j != qi and idx.categories[j] == idx.categories[qi]
        ]
        all_relevant.append(relevant)
        all_retrieved.append(retrieved)

    metrics = summarize(all_relevant, all_retrieved, ks=ks)

    print("\n=== Ranking metrics ===")
    for k, v in metrics.items():
        print(f"  {k:<10} {v:.4f}")

    out = SETTINGS.classifier_path.parent / "ranking_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved metrics to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
