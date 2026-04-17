"""High-level pipeline that stitches all HireFormer modules together."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .extraction import extract_text, is_supported
from .preprocessing import clean_text, truncate_words
from .ranking import CandidateIndex, RankedCandidate
from .gap_analysis import gap_analysis, generate_feedback, GapResult
from .jobs import JobIndex, JobMatch

logger = logging.getLogger(__name__)


@dataclass
class MatchReport:
    job_description: str
    candidates: List[RankedCandidate] = field(default_factory=list)
    predicted_categories: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "job_description": self.job_description[:500],
            "candidates": [c.to_dict() for c in self.candidates],
            "predicted_categories": self.predicted_categories,
        }


@dataclass
class ApplicantReport:
    resume_text: str
    predicted_categories: List[dict] = field(default_factory=list)
    gap: Optional[GapResult] = None
    feedback: List[str] = field(default_factory=list)
    matching_jobs: List[JobMatch] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "resume_excerpt": self.resume_text[:500],
            "predicted_categories": self.predicted_categories,
            "gap": self.gap.to_dict() if self.gap else None,
            "feedback": self.feedback,
            "matching_jobs": [j.to_dict() for j in self.matching_jobs],
        }


def load_index() -> CandidateIndex:
    return CandidateIndex.load()


def load_job_index() -> JobIndex:
    return JobIndex.load()


def text_from_file(file_path: str | Path) -> str:
    """Extract + clean text from a single resume/JD file."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(p)
    if not is_supported(p):
        raise ValueError(f"Unsupported file type: {p.suffix}")
    raw = extract_text(p) or ""
    return clean_text(raw)


def rank_for_jd(
    jd_text: str,
    *,
    top_k: int = 10,
    category_filter: Optional[str] = None,
    index: Optional[CandidateIndex] = None,
    classify: bool = True,
) -> MatchReport:
    """Rank candidates from the index for a given job description."""
    idx = index or load_index()
    cleaned = truncate_words(clean_text(jd_text), 1024)
    candidates = idx.rank(cleaned, top_k=top_k, category_filter=category_filter)

    predicted: List[dict] = []
    if classify:
        try:
            from . import classifier  # local import to avoid hard dep
            predicted = classifier.predict(cleaned, top_k=3)
        except FileNotFoundError:
            predicted = []
    return MatchReport(job_description=cleaned, candidates=candidates, predicted_categories=predicted)


def analyze_applicant(
    resume_text: str,
    *,
    jd_text: Optional[str] = None,
    classify: bool = True,
    suggest_jobs: bool = False,
    jobs_top_k: int = 5,
    job_index: Optional[JobIndex] = None,
) -> ApplicantReport:
    """Produce a full applicant report (category prediction + gap analysis + job matches)."""
    cleaned_resume = truncate_words(clean_text(resume_text), 1024)

    predicted: List[dict] = []
    if classify:
        try:
            from . import classifier
            predicted = classifier.predict(cleaned_resume, top_k=3)
        except FileNotFoundError:
            predicted = []

    gap: Optional[GapResult] = None
    feedback: List[str] = []
    if jd_text:
        cleaned_jd = clean_text(jd_text)
        gap = gap_analysis(cleaned_resume, cleaned_jd)
        feedback = generate_feedback(gap)

    matches: List[JobMatch] = []
    if suggest_jobs:
        try:
            idx = job_index or load_job_index()
            matches = idx.match(cleaned_resume, top_k=jobs_top_k)
        except FileNotFoundError:
            matches = []

    return ApplicantReport(
        resume_text=cleaned_resume,
        predicted_categories=predicted,
        gap=gap,
        feedback=feedback,
        matching_jobs=matches,
    )
