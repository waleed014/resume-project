"""Load, embed and search the real-world job-description dataset.

The dataset is ``job_title_des.csv`` with columns::

    ,Job Title,Job Description

`build_job_index` embeds all JDs once and persists the vectors + metadata.
`JobIndex.match` then returns the top-k best-fitting jobs for any resume text.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import SETTINGS
from .embeddings import encode_one, encode_texts
from .preprocessing import clean_text, truncate_words

logger = logging.getLogger(__name__)


@dataclass
class JobMatch:
    rank: int
    score: float
    title: str
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_jd_dataframe(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Read the JD CSV and return a dataframe with ``title`` + ``description`` columns."""
    path = Path(csv_path) if csv_path else SETTINGS.jd_csv_path
    if not path.exists():
        raise FileNotFoundError(f"Job description CSV not found: {path}")
    df = pd.read_csv(path)
    # Normalise column names; the file has a leading unnamed index column.
    rename_map = {}
    for col in df.columns:
        low = col.lower().strip()
        if low == "job title":
            rename_map[col] = "title"
        elif low == "job description":
            rename_map[col] = "description"
    df = df.rename(columns=rename_map)
    if "title" not in df.columns or "description" not in df.columns:
        raise ValueError(
            f"Unexpected JD CSV columns: {list(df.columns)}. "
            "Expected 'Job Title' and 'Job Description'."
        )
    df = df[["title", "description"]].copy()
    df = df.dropna(subset=["description"])
    df["title"] = df["title"].fillna("Unknown").astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()
    df = df[df["description"].str.len() >= 30].reset_index(drop=True)
    return df


def build_job_index(
    csv_path: Path | str | None = None,
    *,
    max_words: int = 512,
    dedupe: bool = True,
) -> int:
    """Embed every row in the JD CSV and save to disk. Returns row count."""
    df = load_jd_dataframe(csv_path)
    if dedupe:
        df = df.drop_duplicates(subset=["title", "description"]).reset_index(drop=True)

    titles = df["title"].tolist()
    descriptions = df["description"].tolist()

    # Combine title + description so short descriptions still get useful signal.
    texts = [
        truncate_words(clean_text(f"{t}. {d}"), max_words)
        for t, d in zip(titles, descriptions)
    ]

    logger.info("Encoding %d job descriptions ...", len(texts))
    embeddings = encode_texts(texts, show_progress=True, normalize=True)

    SETTINGS.jd_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(SETTINGS.jd_embeddings_path, embeddings)
    with open(SETTINGS.jd_metadata_path, "w", encoding="utf-8") as f:
        json.dump({"titles": titles, "descriptions": descriptions}, f)

    logger.info(
        "Saved %d JD embeddings to %s (shape=%s)",
        len(texts), SETTINGS.jd_embeddings_path, embeddings.shape,
    )
    return len(texts)


class JobIndex:
    """Reverse lookup: resume text -> top-K matching job descriptions."""

    def __init__(
        self,
        embeddings: np.ndarray,
        titles: List[str],
        descriptions: List[str],
    ) -> None:
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.titles = titles
        self.descriptions = descriptions

    @classmethod
    def load(cls) -> "JobIndex":
        if not SETTINGS.jd_embeddings_path.exists() or not SETTINGS.jd_metadata_path.exists():
            raise FileNotFoundError(
                "Job embeddings not built. Run `python -m scripts.build_job_index`."
            )
        emb = np.load(SETTINGS.jd_embeddings_path)
        with open(SETTINGS.jd_metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(emb, meta["titles"], meta["descriptions"])

    def __len__(self) -> int:
        return len(self.titles)

    def match(
        self,
        query_text: str,
        top_k: int = 10,
        title_contains: Optional[str] = None,
    ) -> List[JobMatch]:
        q = encode_one(clean_text(query_text), normalize=True)
        scores = self.embeddings @ q

        idx_pool = np.arange(len(self))
        if title_contains:
            needle = title_contains.lower()
            mask = np.array([needle in t.lower() for t in self.titles])
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

        out: List[JobMatch] = []
        for rank, gi in enumerate(top_global, 1):
            out.append(
                JobMatch(
                    rank=rank,
                    score=float(scores[gi]),
                    title=self.titles[gi],
                    description=(self.descriptions[gi] or "")[:1200],
                )
            )
        return out
