"""Retrieval and classification evaluation metrics.

- Precision@K
- Recall@K
- NDCG@K
- MAP
- Per-class F1 (via scikit-learn)
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


def precision_at_k(relevant: Iterable, retrieved: Sequence, k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = list(retrieved)[:k]
    rel_set = set(relevant)
    return sum(1 for d in top_k if d in rel_set) / k


def recall_at_k(relevant: Iterable, retrieved: Sequence, k: int) -> float:
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    top_k = list(retrieved)[:k]
    return sum(1 for d in top_k if d in rel_set) / len(rel_set)


def ndcg_at_k(relevant: Iterable, retrieved: Sequence, k: int) -> float:
    rel_set = set(relevant)
    dcg = sum(
        1.0 / np.log2(i + 2) for i, doc in enumerate(list(retrieved)[:k]) if doc in rel_set
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel_set), k)))
    return float(dcg / max(idcg, 1e-10))


def average_precision(relevant: Iterable, retrieved: Sequence) -> float:
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    hits, ap = 0, 0.0
    for i, doc in enumerate(retrieved):
        if doc in rel_set:
            hits += 1
            ap += hits / (i + 1)
    return ap / len(rel_set)


def mean_average_precision(
    queries_relevant: Sequence[Iterable], queries_retrieved: Sequence[Sequence]
) -> float:
    aps = [
        average_precision(rel, ret)
        for rel, ret in zip(queries_relevant, queries_retrieved)
    ]
    return float(np.mean(aps)) if aps else 0.0


def summarize(
    queries_relevant: Sequence[Iterable],
    queries_retrieved: Sequence[Sequence],
    ks: Sequence[int] = (1, 5, 10, 20),
) -> dict:
    """Compute P@k, R@k, NDCG@k for each k plus MAP over all queries."""
    out: dict = {"MAP": mean_average_precision(queries_relevant, queries_retrieved)}
    for k in ks:
        p = [precision_at_k(r, ret, k) for r, ret in zip(queries_relevant, queries_retrieved)]
        r = [recall_at_k(rel, ret, k) for rel, ret in zip(queries_relevant, queries_retrieved)]
        n = [ndcg_at_k(rel, ret, k) for rel, ret in zip(queries_relevant, queries_retrieved)]
        out[f"P@{k}"] = float(np.mean(p))
        out[f"R@{k}"] = float(np.mean(r))
        out[f"NDCG@{k}"] = float(np.mean(n))
    return out
