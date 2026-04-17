"""Skill extraction, gap analysis and applicant feedback generation."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Set

# A pragmatic skill lexicon. Multi-word skills are matched as phrases first.
SKILL_LEXICON: dict[str, List[str]] = {
    "languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "kotlin", "swift", "ruby", "php", "scala", "r", "matlab", "sql", "bash",
        "shell", "perl",
    ],
    "web_frameworks": [
        "django", "flask", "fastapi", "spring", "spring boot", "express",
        "next.js", "nuxt", "react", "angular", "vue", "svelte", "laravel",
        "rails", "asp.net",
    ],
    "data_ml": [
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
        "xgboost", "lightgbm", "huggingface", "transformers", "spark", "hadoop",
        "airflow", "dbt", "snowflake", "databricks", "tableau", "power bi",
        "looker", "matplotlib", "seaborn", "plotly", "nlp", "computer vision",
        "deep learning", "machine learning", "data science",
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform",
        "ansible", "jenkins", "github actions", "gitlab ci", "circleci",
        "ci/cd", "linux", "bash scripting", "prometheus", "grafana",
    ],
    "databases": [
        "mysql", "postgresql", "postgres", "mongodb", "redis", "cassandra",
        "elasticsearch", "dynamodb", "oracle", "sqlite", "sql server", "mariadb",
        "neo4j",
    ],
    "qa_security": [
        "selenium", "cypress", "jest", "pytest", "junit", "owasp", "burp suite",
        "penetration testing", "siem", "splunk",
    ],
    "soft_skills": [
        "leadership", "communication", "teamwork", "problem solving",
        "project management", "stakeholder management", "presentation",
        "mentoring",
    ],
    "business": [
        "agile", "scrum", "kanban", "waterfall", "sap", "salesforce",
        "jira", "confluence", "tableau", "power bi", "excel", "vba",
        "financial modeling", "auditing", "tax", "ifrs", "gaap",
    ],
}

ALL_SKILLS: List[str] = sorted(
    {s.lower() for group in SKILL_LEXICON.values() for s in group},
    key=lambda s: -len(s),  # longest first to greedy-match phrases
)


def extract_skills(text: str) -> Set[str]:
    """Return the set of skills found in ``text`` (case-insensitive)."""
    if not text:
        return set()
    lower = text.lower()
    found: Set[str] = set()
    for skill in ALL_SKILLS:
        # Word-boundary match; skip pure punctuation skills.
        pattern = r"(?<![A-Za-z0-9+#])" + re.escape(skill) + r"(?![A-Za-z0-9])"
        if re.search(pattern, lower):
            found.add(skill)
    return found


@dataclass
class GapResult:
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    extra_skills: List[str] = field(default_factory=list)
    match_percentage: float = 0.0

    def to_dict(self) -> dict:
        return {
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "extra_skills": self.extra_skills,
            "match_percentage": self.match_percentage,
        }


def gap_analysis(resume_text: str, jd_text: str) -> GapResult:
    """Compare a resume against a job description; return matched/missing skills."""
    r = extract_skills(resume_text)
    j = extract_skills(jd_text)
    matched = r & j
    missing = j - r
    extra = r - j
    pct = (len(matched) / max(len(j), 1)) * 100.0
    return GapResult(
        matched_skills=sorted(matched),
        missing_skills=sorted(missing),
        extra_skills=sorted(extra),
        match_percentage=round(pct, 2),
    )


def generate_feedback(gap: GapResult) -> List[str]:
    """Return a list of human-readable improvement suggestions."""
    feedback: List[str] = []
    if gap.match_percentage >= 75:
        feedback.append(
            "Strong match — your skills align very well with this role's requirements."
        )
    elif gap.match_percentage >= 50:
        feedback.append(
            "Moderate match. With a few targeted additions you could be a strong fit."
        )
    else:
        feedback.append(
            "Low overlap with the job requirements. Consider tailoring your resume "
            "more closely to this role."
        )

    if gap.missing_skills:
        top_missing = ", ".join(gap.missing_skills[:8])
        feedback.append(
            f"Highlight or develop these missing skills: {top_missing}."
        )
    if gap.matched_skills:
        feedback.append(
            f"Lead with these matched strengths: {', '.join(gap.matched_skills[:5])}."
        )
    if gap.extra_skills and len(gap.extra_skills) > 5:
        feedback.append(
            "You list many skills that are not asked for in this JD — consider "
            "trimming the resume to keep it focused."
        )
    return feedback
