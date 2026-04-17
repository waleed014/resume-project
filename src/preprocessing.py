"""Text cleaning and normalization for resumes and job descriptions."""
from __future__ import annotations

import re
import unicodedata

# Common OCR/PDF artifacts.
_BULLETS = re.compile(r"[•●◦▪▫■□◇◆★☆➢➤▶►–—]+")
_MULTI_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_NON_PRINT = re.compile(r"[^\x09\x0a\x0d\x20-\x7e\u00a0-\uffff]")


def clean_text(text: str, *, drop_pii: bool = False) -> str:
    """Normalize whitespace and strip OCR artifacts.

    If ``drop_pii`` is True, emails/phones/URLs are removed.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _NON_PRINT.sub(" ", text)
    text = _BULLETS.sub(" ", text)
    if drop_pii:
        text = _URL.sub(" ", text)
        text = _EMAIL.sub(" ", text)
        text = _PHONE.sub(" ", text)
    text = _MULTI_WS.sub(" ", text)
    return text.strip()


def truncate_words(text: str, max_words: int = 1024) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])
