"""Central configuration for the HireFormer project."""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Project root = directory containing this file's parent's parent (repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Local data folders (created on first run)
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_DIR = DATA_DIR / "models"
EXTRACTED_DIR = DATA_DIR / "extracted"
JD_DIR = DATA_DIR / "job_descriptions"

for d in (DATA_DIR, CACHE_DIR, EMBEDDINGS_DIR, MODELS_DIR, EXTRACTED_DIR, JD_DIR):
    d.mkdir(parents=True, exist_ok=True)

# External raw resume sources from the workspace.
# These are the folders that hold the actual resumes (DOCX, PNG, JPG, WebP).
DEFAULT_RAW_SOURCES: List[Path] = [
    Path(r"C:\Users\devmu\Downloads\Resumes Datasets\resume_database"),
    Path(r"C:\Users\devmu\Downloads\Resumes Datasets\Scrapped_Resumes"),
    Path(r"C:\Users\devmu\Downloads\Resumes Datasets\Bing_images"),
]

# Real-world job-description dataset (CSV with columns: Job Title, Job Description).
DEFAULT_JD_CSV = Path(r"C:\Users\devmu\Downloads\Job Description Data sets\job_title_des.csv")


@dataclass
class Settings:
    """Runtime settings for HireFormer."""

    # Embedding model
    embedding_model: str = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
    embedding_batch_size: int = int(os.getenv("HF_EMBED_BATCH", "32"))
    max_seq_length: int = 384

    # Text extraction
    enable_ocr: bool = os.getenv("HF_ENABLE_OCR", "1") == "1"
    tesseract_cmd: str = os.getenv(
        "HF_TESSERACT_CMD",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    )
    min_text_length: int = 50

    # Ranking
    default_top_k: int = 10

    # Paths
    raw_sources: List[Path] = field(default_factory=lambda: list(DEFAULT_RAW_SOURCES))
    embeddings_path: Path = EMBEDDINGS_DIR / "resume_embeddings.npy"
    metadata_path: Path = EMBEDDINGS_DIR / "resume_metadata.json"
    texts_path: Path = EMBEDDINGS_DIR / "resume_texts.json"
    classifier_path: Path = MODELS_DIR / "domain_classifier.joblib"
    label_encoder_path: Path = MODELS_DIR / "label_encoder.joblib"
    extracted_cache: Path = EXTRACTED_DIR / "extracted_texts.jsonl"

    # Job description CSV + derived artifacts.
    jd_csv_path: Path = DEFAULT_JD_CSV
    jd_embeddings_path: Path = EMBEDDINGS_DIR / "jd_embeddings.npy"
    jd_metadata_path: Path = EMBEDDINGS_DIR / "jd_metadata.json"

    # Canonical category map (handles dataset folder-name inconsistencies).
    category_aliases: dict = field(default_factory=lambda: {
        "agricultural": "Agriculture",
        "agriculture": "Agriculture",
        "architects": "Architect",
        "architect": "Architect",
        "businessanalyst": "Business Analyst",
        "business analyst": "Business Analyst",
        "civilengineer": "Civil Engineer",
        "civil engineer": "Civil Engineer",
        "consult": "Consultant",
        "consultant": "Consultant",
        "data science": "Data Science",
        "datascience": "Data Science",
        "designer": "Designer",
        "design": "Designer",
        "designing": "Designer",
        "devopsengineer": "DevOps Engineer",
        "devops engineer": "DevOps Engineer",
        "digital": "Digital Media",
        "digital media": "Digital Media",
        "dot": "DotNet Developer",
        "dotnet developer": "DotNet Developer",
        "etl": "ETL Developer",
        "etl developer": "ETL Developer",
        "food": "Food/Beverages",
        "food_beverages": "Food/Beverages",
        "healthfitness": "Health/Fitness",
        "health_fitness": "Health/Fitness",
        "hr": "HR",
        "human resources": "HR",
        "it": "Information Technology",
        "information technology": "Information Technology",
        "javadeveloper": "Java Developer",
        "java developer": "Java Developer",
        "managment": "Management",
        "management": "Management",
        "mechanicalengineer": "Mechanical Engineer",
        "mechanical engineer": "Mechanical Engineer",
        "nse": "Network Security Engineer",
        "network security engineer": "Network Security Engineer",
        "operationmanager": "Operations Manager",
        "operations manager": "Operations Manager",
        "pbo": "PMO",
        "pmo": "PMO",
        "public": "Public Relations",
        "public relations": "Public Relations",
        "pythondeveloper": "Python Developer",
        "python developer": "Python Developer",
        "react": "React Developer",
        "react developer": "React Developer",
        "electricalengineer": "Electrical Engineer",
        "electrical engineering": "Electrical Engineer",
        "sapdeveloper": "SAP Developer",
        "sap developer": "SAP Developer",
        "sql": "SQL Developer",
        "sql developer": "SQL Developer",
        "webdesigning": "Web Designing",
        "web designing": "Web Designing",
        "building _construction": "Building/Construction",
        "bpo": "BPO",
    })


SETTINGS = Settings()


def canonical_category(name: str) -> str:
    """Normalize a folder name to a canonical category label."""
    if not name:
        return "Unknown"
    raw = name.strip()
    # Strip ' resumes' suffix used by some folders.
    lowered = raw.lower().removesuffix(" resumes").strip()
    return SETTINGS.category_aliases.get(lowered, raw.removesuffix(" resumes").strip().title())
