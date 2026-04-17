import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gap_analysis import extract_skills, gap_analysis, generate_feedback
from src.preprocessing import clean_text


def test_clean_text_strips_bullets_and_whitespace():
    raw = "•  Python  •  Django\n\n\tREST APIs   "
    out = clean_text(raw)
    assert "Python" in out and "Django" in out
    assert "  " not in out
    assert "•" not in out


def test_extract_skills_basic():
    text = "Experienced Python developer with Django, PostgreSQL and AWS."
    skills = extract_skills(text)
    assert {"python", "django", "postgresql", "aws"}.issubset(skills)


def test_gap_analysis_and_feedback():
    resume = "Python, Django, MongoDB, Git"
    jd = "Looking for Python, Django, PostgreSQL, Docker, Kubernetes"
    gap = gap_analysis(resume, jd)
    assert "postgresql" in gap.missing_skills
    assert "python" in gap.matched_skills
    assert 0 <= gap.match_percentage <= 100
    fb = generate_feedback(gap)
    assert isinstance(fb, list) and len(fb) >= 1
