"""Embedding model wrapper around sentence-transformers."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, List

import numpy as np

from .config import SETTINGS

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def get_encoder(model_name: str | None = None):
    """Lazy-load and cache the sentence-transformer model."""
    from sentence_transformers import SentenceTransformer

    name = model_name or SETTINGS.embedding_model
    logger.info("Loading embedding model: %s", name)
    model = SentenceTransformer(name)
    try:
        model.max_seq_length = SETTINGS.max_seq_length
    except Exception:  # noqa: BLE001
        pass
    return model


def encode_texts(
    texts: Iterable[str],
    *,
    batch_size: int | None = None,
    show_progress: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Encode a list of texts to a (N, D) float32 matrix."""
    model = get_encoder()
    bs = batch_size or SETTINGS.embedding_batch_size
    arr = model.encode(
        list(texts),
        batch_size=bs,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return arr.astype(np.float32, copy=False)


def encode_one(text: str, normalize: bool = True) -> np.ndarray:
    """Encode a single text and return a 1-D vector."""
    vec = encode_texts([text], show_progress=False, normalize=normalize)
    return vec[0]
