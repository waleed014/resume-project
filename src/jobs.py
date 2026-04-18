"""Load, embed and search the real-world job-description dataset.

The dataset is ``job_title_des.csv`` with columns::

    ,Job Title,Job Description

`build_job_index` embeds all JDs once and persists the vectors + metadata.
`JobIndex.match` then returns the top-k best-fitting jobs for any resume text.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


# Map every canonical resume category -> list of substring keywords that should
# appear in a JD title for the row to be considered "relevant" to that category.
# Used to prune the very large JD CSV folder (189 per-title files) down to only
# the JDs that are actually useful for ranking against our resume corpus.
RESUME_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Accountant": ["accountant", "accounts", "bookkeep", "auditor", "audit clerk"],
    "Advocate": ["advocate", "lawyer", "legal", "attorney", "paralegal", "solicitor"],
    "Agriculture": ["agricultur", "farm", "horticultur", "agronomist"],
    "Apparel": ["apparel", "fashion", "garment", "tailor", "textile"],
    "Architect": ["architect"],
    "Arts": ["artist", "painter", "sculpt", "musician", "actor", "actress", "dancer", "photographer"],
    "Automobile": ["automobile", "automotive", "vehicle", "car mechanic", "auto mechanic"],
    "Aviation": ["aviation", "pilot", "aircraft", "aerospace", "flight", "cabin crew", "air hostess"],
    "Avian": ["avian", "ornithol", "poultry"],
    "BPO": ["bpo", "call center", "call centre", "customer service", "telecaller", "tele caller"],
    "Banking": ["bank", "teller", "branch manager", "loan officer", "credit officer"],
    "Blockchain": ["blockchain", "crypto", "web3", "solidity", "smart contract"],
    "Building/Construction": ["construction", "builder", "site engineer", "contractor", "site supervisor"],
    "Business Analyst": ["business analyst", "business analysis"],
    "Civil Engineer": ["civil engineer"],
    "Consultant": ["consultant", "consulting", "advisor"],
    "Data Science": ["data scien", "machine learning", "ml engineer", "ai engineer", "data analy"],
    "Database": ["database", "dba", "sql server", "oracle developer", "data engineer"],
    "Designer": ["designer", "graphic design", "ux designer", "ui designer", "interior design"],
    "DevOps Engineer": ["devops", "site reliability", "sre", "platform engineer"],
    "Digital Media": ["digital media", "social media", "content creator", "seo", "digital marketing"],
    "DotNet Developer": ["dotnet", ".net", "asp.net", "c# developer"],
    "Education": ["teacher", "lecturer", "tutor", "professor", "instructor"],
    "ETL Developer": ["etl", "informatica"],
    "Electrical Engineer": ["electrical engineer", "electrician"],
    "Finance": ["finance", "financial analyst", "investment", "treasury"],
    "Food/Beverages": ["chef", "cook", "barista", "bartender", "waiter", "kitchen", "beverage"],
    "HR": ["human resources", "hr ", "recruiter", "recruitment", "talent acquisition", "hr manager"],
    "Health/Fitness": ["nurse", "doctor", "physician", "fitness", "personal trainer", "therapist", "physio"],
    "Information Technology": ["it support", "system administrator", "sysadmin", "network engineer", "help desk", "service desk"],
    "Java Developer": ["java developer", "java engineer", "spring developer"],
    "Mechanical Engineer": ["mechanical engineer"],
    "Network Security Engineer": ["network security", "cybersecurity", "security engineer", "infosec", "penetration tester"],
    "Operations Manager": ["operations manager", "operation manager"],
    "PMO": ["pmo", "project management office", "project manager", "scrum master", "program manager"],
    "Public Relations": ["public relations", "communications officer", "communications manager"],
    "Python Developer": ["python developer", "django developer", "flask developer"],
    "React Developer": ["react developer", "frontend developer", "front end developer", "front-end developer", "javascript developer"],
    "SAP Developer": ["sap"],
    "SQL Developer": ["sql developer", "sql programmer"],
    "Sales": ["sales", "account executive", "business development"],
    "Testing": ["qa engineer", "quality assurance", "test engineer", "tester", "sdet", "automation test"],
    "Web Designing": ["web design"],
    # Management is intentionally last so more specific titles
    # (Operations Manager, Project Manager, HR Manager, Sales Manager, ...)
    # win the substring race against the generic "manager" keyword.
    "Management": ["general manager", "operations director", "managing director", "head of", "executive director", "chief executive", "ceo", "cto", "cfo", "coo"],
}


def _match_category(title: str, category_keywords: dict[str, list[str]] | None = None) -> str | None:
    """Return the first resume category whose keyword appears in the title."""
    if not isinstance(title, str) or not title:
        return None
    t = title.lower()
    mapping = category_keywords or RESUME_CATEGORY_KEYWORDS
    for cat, kws in mapping.items():
        for kw in kws:
            if kw in t:
                return cat
    return None


def _strip_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return _WS_RE.sub(" ", text).strip()


def _read_one_jd_csv(path: Path) -> pd.DataFrame:
    """Load a single JD CSV and normalize to columns: title, description."""
    try:
        df = pd.read_csv(path, on_bad_lines="skip", low_memory=False)
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return pd.DataFrame(columns=["title", "description"])

    cols = {c.lower().strip(): c for c in df.columns}

    # Schema A: per-title scrape  -> title, description (+company,location,skills)
    # Schema B: aggregate         -> Job Title, Job Description
    # Schema C: job_dataset.csv   -> Title, Skills, Responsibilities, Keywords
    if "job title" in cols and "job description" in cols:
        out = pd.DataFrame({
            "title": df[cols["job title"]],
            "description": df[cols["job description"]],
        })
    elif "title" in cols and "description" in cols:
        out = pd.DataFrame({
            "title": df[cols["title"]],
            "description": df[cols["description"]],
        })
    elif "title" in cols and ("skills" in cols or "responsibilities" in cols):
        parts = []
        for key in ("responsibilities", "skills", "keywords"):
            if key in cols:
                parts.append(df[cols[key]].fillna("").astype(str))
        desc = parts[0]
        for p in parts[1:]:
            desc = desc.str.cat(p, sep=". ")
        out = pd.DataFrame({"title": df[cols["title"]], "description": desc})
    else:
        return pd.DataFrame(columns=["title", "description"])

    out["title"] = out["title"].fillna("Unknown").astype(str).str.strip()
    out["description"] = out["description"].astype(str).map(_strip_html)
    out = out.dropna(subset=["description"])
    out = out[out["description"].str.len() >= 30]
    return out

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
    category: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_jd_dataframe(
    csv_path: Path | str | None = None,
    *,
    filter_to_resume_categories: bool = False,
    per_category_limit: int | None = None,
) -> pd.DataFrame:
    """Load JDs from a single CSV or merge every CSV in a folder.

    - If ``csv_path`` is a file, only that file is loaded.
    - If ``csv_path`` (or the configured default) is a folder, *all* ``*.csv``
      files inside it are merged. This lets us ingest the per-job-title CSVs
      (Accountant.csv, Data Scientist.csv, ...) in addition to the aggregate
      ``job_title_des.csv`` and ``job_dataset.csv`` files.
    - When ``filter_to_resume_categories`` is True, only JD rows whose title
      maps to one of the canonical resume categories (via
      ``RESUME_CATEGORY_KEYWORDS``) are kept, and a ``category`` column is
      added. Folder-style sources (per-title CSVs) also use the *file name*
      as a fallback signal so e.g. ``Accountant.csv`` rows get tagged as
      ``Accountant`` even when the inner ``title`` field is generic.
    - ``per_category_limit`` caps rows per resume category (after the filter)
      so a few popular categories don't dwarf the rest.
    """
    path = Path(csv_path) if csv_path else SETTINGS.jd_csv_path
    if not path.exists():
        raise FileNotFoundError(f"Job description path not found: {path}")

    if path.is_file():
        df = _read_one_jd_csv(path).reset_index(drop=True)
        if filter_to_resume_categories:
            df["category"] = df["title"].map(_match_category)
            df = df.dropna(subset=["category"]).reset_index(drop=True)
            if per_category_limit:
                df = (
                    df.groupby("category", group_keys=False)
                    .head(per_category_limit)
                    .reset_index(drop=True)
                )
        return df

    # Directory: merge every CSV inside it.
    files = sorted(path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in folder: {path}")

    frames: list[pd.DataFrame] = []
    for f in files:
        sub = _read_one_jd_csv(f)
        if sub.empty:
            continue
        if filter_to_resume_categories:
            # Try the row title first; fall back to the file stem (e.g.
            # "Accountant.csv" -> "Accountant") when the row title doesn't
            # match any resume category.
            file_cat = _match_category(f.stem)
            sub["category"] = sub["title"].map(_match_category)
            if file_cat is not None:
                sub["category"] = sub["category"].fillna(file_cat)
            sub = sub.dropna(subset=["category"])
            if sub.empty:
                continue
        frames.append(sub)
    if not frames:
        raise ValueError(f"No usable JD rows found under: {path}")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["description"])
    df = df[df["description"].str.len() >= 30]
    dedupe_cols = ["title", "description"]
    if "category" in df.columns:
        # Don't lose category info during dedupe.
        df = df.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)
        if per_category_limit:
            df = (
                df.groupby("category", group_keys=False)
                .head(per_category_limit)
                .reset_index(drop=True)
            )
    else:
        df = df.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)
    logger.info(
        "Loaded %d JD rows from %d CSV file(s) in %s%s",
        len(df),
        len(files),
        path,
        " (filtered to resume categories)" if filter_to_resume_categories else "",
    )
    return df


def build_job_index(
    csv_path: Path | str | None = None,
    *,
    max_words: int = 512,
    dedupe: bool = True,
    filter_to_resume_categories: bool = False,
    per_category_limit: int | None = None,
    batch_size: int | None = None,
) -> int:
    """Embed every row in the JD CSV and save to disk. Returns row count."""
    df = load_jd_dataframe(
        csv_path,
        filter_to_resume_categories=filter_to_resume_categories,
        per_category_limit=per_category_limit,
    )
    if dedupe:
        df = df.drop_duplicates(subset=["title", "description"]).reset_index(drop=True)

    titles = df["title"].tolist()
    descriptions = df["description"].tolist()
    categories = df["category"].tolist() if "category" in df.columns else [None] * len(df)

    # Combine title + description so short descriptions still get useful signal.
    texts = [
        truncate_words(clean_text(f"{t}. {d}"), max_words)
        for t, d in zip(titles, descriptions)
    ]

    logger.info("Encoding %d job descriptions ...", len(texts))
    embeddings = encode_texts(
        texts, show_progress=True, normalize=True, batch_size=batch_size
    )

    SETTINGS.jd_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(SETTINGS.jd_embeddings_path, embeddings)
    with open(SETTINGS.jd_metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {"titles": titles, "descriptions": descriptions, "categories": categories},
            f,
        )

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
        categories: List[str | None] | None = None,
    ) -> None:
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.titles = titles
        self.descriptions = descriptions
        self.categories = categories or [None] * len(titles)

    @classmethod
    def load(cls) -> "JobIndex":
        if not SETTINGS.jd_embeddings_path.exists() or not SETTINGS.jd_metadata_path.exists():
            raise FileNotFoundError(
                "Job embeddings not built. Run `python -m scripts.build_job_index`."
            )
        emb = np.load(SETTINGS.jd_embeddings_path)
        with open(SETTINGS.jd_metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(
            emb,
            meta["titles"],
            meta["descriptions"],
            meta.get("categories"),
        )

    def __len__(self) -> int:
        return len(self.titles)

    def match(
        self,
        query_text: str,
        top_k: int = 10,
        title_contains: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[JobMatch]:
        q = encode_one(clean_text(query_text), normalize=True)
        scores = self.embeddings @ q

        idx_pool = np.arange(len(self))
        if category and any(c == category for c in self.categories):
            mask = np.array([c == category for c in self.categories])
            idx_pool = idx_pool[mask]
        if title_contains:
            needle = title_contains.lower()
            mask = np.array([needle in self.titles[i].lower() for i in idx_pool])
            idx_pool = idx_pool[mask]
        if idx_pool.size == 0:
            return []
        scores_pool = scores[idx_pool]

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
                    category=self.categories[gi] if gi < len(self.categories) else None,
                )
            )
        return out
