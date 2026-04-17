"""Semantic candidate ranking against pre-computed resume embeddings."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import SETTINGS
from .embeddings import encode_one

logger = logging.getLogger(__name__)


@dataclass
class RankedCandidate:
    rank: int
    score: float
    file: str
    category: str
    snippet: str

    def to_dict(self) -> dict:
        return asdict(self)


class CandidateIndex:
    """In-memory index of resume embeddings + metadata."""

    def __init__(
        self,
        embeddings: np.ndarray,
        filenames: List[str],
        categories: List[str],
        texts: Optional[List[str]] = None,
    ) -> None:
        self.embeddings = embeddings.astype(np.float32, copy=False)
        # Embeddings produced by `encode_texts(normalize=True)` are unit length,
        # so cosine similarity reduces to a dot product.
        self.filenames = filenames
        self.categories = categories
        self.texts = texts or []

    @classmethod
    def load(cls) -> "CandidateIndex":
        emb_path: Path = SETTINGS.embeddings_path
        meta_path: Path = SETTINGS.metadata_path
        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Embeddings/metadata missing. Run scripts/build_embeddings.py first."
            )
        embeddings = np.load(emb_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        texts: List[str] = []
        if SETTINGS.texts_path.exists():
            with open(SETTINGS.texts_path, "r", encoding="utf-8") as f:
                texts = json.load(f)
        return cls(embeddings, meta["filenames"], meta["categories"], texts)

    def __len__(self) -> int:
        return len(self.filenames)

    def rank(
        self,
        query_text: str,
        top_k: int = 10,
        category_filter: Optional[str] = None,
    ) -> List[RankedCandidate]:
        """Rank resumes by cosine similarity to ``query_text``."""
        q = encode_one(query_text, normalize=True)
        scores = self.embeddings @ q  # cosine sim (vectors are normalized)

        idx_pool = np.arange(len(self))
        if category_filter:
            mask = np.array(
                [c.lower() == category_filter.lower() for c in self.categories]
            )
            idx_pool = idx_pool[mask]
            if idx_pool.size == 0:
                return []
            scores_pool = scores[idx_pool]
        else:
            scores_pool = scores

        k = min(top_k, len(scores_pool))
        top_local = np.argpartition(-scores_pool, kth=k - 1)[:k]
        top_local = top_local[np.argsort(-scores_pool[top_local])]
        top_global = idx_pool[top_local]

        results: List[RankedCandidate] = []
        for rank, gi in enumerate(top_global, 1):
            snippet = ""
            if self.texts:
                snippet = (self.texts[gi] or "")[:300]
            results.append(
                RankedCandidate(
                    rank=rank,
                    score=float(scores[gi]),
                    file=self.filenames[gi],
                    category=self.categories[gi],
                    snippet=snippet,
                )
            )
        return results
