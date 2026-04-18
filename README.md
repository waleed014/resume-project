# HireFormer

> Transformer-based talent matching, resume domain classification, skill gap analysis, and job recommendation system.

[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?logo=streamlit)](https://hireformer.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Pipeline](#data-pipeline)
- [API Reference](#api-reference)
- [Frontend (Streamlit)](#frontend-streamlit)
- [Model Details](#model-details)
- [Evaluation & Results](#evaluation--results)
- [Deployment](#deployment)
- [Tests](#tests)
- [Future Work](#future-work)

---

## Overview

HireFormer is a graduation-project system that uses **sentence-transformer embeddings** to semantically match resumes to job descriptions. It processes **12,459 resumes** across **40 professional categories** and indexes **8,961 job descriptions** from real-world datasets, enabling three core workflows:

| Workflow | Input | Output |
|----------|-------|--------|
| **Recruiter** | Job description | Top-K ranked candidate resumes |
| **Applicant** | Resume (+ optional JD) | Domain prediction, skill gap analysis, feedback |
| **Job Search** | Resume | Top-K matching job descriptions |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                         │
│                                                              │
│  Raw Resumes (PDF/DOCX/TXT/Images)                           │
│       │                                                      │
│       ▼                                                      │
│  extraction.py ──► preprocessing.py ──► embeddings.py        │
│  (pdfplumber,       (normalize,         (all-mpnet-base-v2,  │
│   python-docx,       clean PII,          768-dim vectors)    │
│   Tesseract OCR)     truncate)                               │
│       │                                    │                 │
│       ▼                                    ▼                 │
│  ocr_cache/                    resume_embeddings.npy          │
│  (SHA1-sharded                 resume_metadata.json           │
│   sidecar cache)               resume_texts.json              │
│                                                              │
│  Job Description CSVs ──► jobs.py ──► jd_embeddings.npy      │
│  (73k+ rows, 189 CSVs)                jd_metadata.json       │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                      INFERENCE LAYER                         │
│                                                              │
│  ranking.py         classifier.py       gap_analysis.py      │
│  (cosine-sim        (LogReg on          (lexicon-based       │
│   top-K search)      embeddings,         skill extraction,   │
│                      40 categories)      matched/missing/    │
│                                          extra skills)       │
│       │                  │                    │              │
│       └──────────┬───────┘────────────────────┘              │
│                  ▼                                            │
│            pipeline.py                                       │
│       (rank_for_jd / analyze_applicant)                      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                        │
│                                                              │
│  ┌─────────────┐              ┌────────────────────┐         │
│  │  FastAPI     │              │  Streamlit          │        │
│  │  (api.py)    │◄────────────│  (streamlit_app.py) │        │
│  │  Port 8000   │              │  Port 8501          │        │
│  │  REST / JSON │              │  5-tab interactive  │        │
│  └─────────────┘              └────────────────────┘         │
└──────────────────────────────────────────────────────────────┘
```

---

## Features

### Core Capabilities

- **Multi-format text extraction** — PDF (`pdfplumber`), DOCX (`python-docx`), TXT, and image files (PNG/JPG/JPEG/WebP/BMP/TIFF) via Tesseract OCR
- **Semantic ranking** — Cosine similarity search using `sentence-transformers/all-mpnet-base-v2` (768-dimensional embeddings)
- **40-category domain classifier** — Logistic Regression trained on resume embeddings (accuracy: 82.2%, macro F1: 84.4%)
- **Skill gap analysis** — Lexicon-based extraction (100+ skills across 8 categories) with matched/missing/extra skill breakdown
- **Job recommendation** — Semantic matching against 8,961 real-world job descriptions
- **Actionable feedback** — Auto-generated improvement suggestions for applicants

### Infrastructure

- **Parallel OCR pipeline** — 16-worker ProcessPoolExecutor with Tesseract 5.4, processes ~9,300 image resumes in ~23 minutes
- **Resumable caching** — SHA1-sharded sidecar cache for OCR results (`data/ocr_cache/<xx>/<sha1>.txt`)
- **Category normalization** — Automatically maps 40+ folder-name variants to 40 canonical categories
- **REST API** — FastAPI with 13 endpoints, file upload support, Swagger UI
- **Interactive UI** — Streamlit with 5 tabs (Recruiter, Applicant, Job Search, Dataset Overview, About)

---

## Project Structure

```
resume-project/
├── app/
│   ├── api.py                  # FastAPI backend (13 endpoints)
│   ├── streamlit_app.py        # Streamlit frontend (5 tabs)
│   └── static/                 # Static assets
├── scripts/
│   ├── build_embeddings.py     # Walk dataset → extract → embed → save
│   ├── build_job_index.py      # Load JD CSVs → embed → save
│   ├── train_classifier.py     # Train LogReg domain classifier
│   ├── ocr_images_tesseract.py # Parallel Tesseract OCR (16 workers)
│   ├── ocr_images.py           # EasyOCR alternative (GPU)
│   ├── ocr_inspect.py          # Debug/inspect OCR cache
│   ├── evaluate_ranking.py     # Benchmark P@K, NDCG, MAP metrics
│   └── dataset_summary.py      # Per-category file counts
├── src/
│   ├── config.py               # Paths, settings, category aliases
│   ├── dataset.py              # Resume file iterator
│   ├── extraction.py           # PDF/DOCX/OCR text extraction
│   ├── preprocessing.py        # Text cleaning & normalization
│   ├── embeddings.py           # SentenceTransformer wrapper
│   ├── ranking.py              # CandidateIndex + cosine search
│   ├── classifier.py           # Domain classifier (train/predict)
│   ├── gap_analysis.py         # Skill extraction & gap analysis
│   ├── jobs.py                 # Job description index
│   ├── ocr_cache.py            # SHA1-sharded OCR cache
│   ├── pipeline.py             # High-level orchestration
│   └── evaluation.py           # IR metrics (P@K, NDCG, MAP)
├── tests/
│   └── test_basic.py           # Smoke tests
├── data/                       # Generated at build time
│   ├── embeddings/             # .npy arrays + metadata JSONs
│   ├── models/                 # Classifier + eval reports
│   └── ocr_cache/              # Cached OCR text files
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- 8 GB+ RAM (for embedding model)
- GPU optional (CUDA for faster embedding; OCR runs on CPU)

### Setup

```bash
git clone https://github.com/devmustafa4/resume-project.git
cd resume-project
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### OCR Setup (for image resumes)

Install Tesseract OCR:

- **Windows**: `winget install --id UB-Mannheim.TesseractOCR -e`
- **Linux**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`

---

## Data Pipeline

### Step 1: OCR Pre-processing (image resumes only)

```bash
# Pre-compute OCR text for all image resumes (parallel, resumable)
python -m scripts.ocr_images_tesseract --workers 16

# Inspect results
python -m scripts.ocr_inspect --samples 5
```

### Step 2: Build Resume Embeddings

```bash
# Quick smoke test (30 files per category, skip OCR)
python -m scripts.build_embeddings --skip-ocr --limit 30

# Full build (all 12,459 resumes)
python -m scripts.build_embeddings
```

Output: `data/embeddings/resume_embeddings.npy` (12459 × 768)

### Step 3: Build Job Description Index

```bash
# Index all JDs with category filtering
python -m scripts.build_job_index --filter-by-resume-categories --per-category-limit 300
```

Output: `data/embeddings/jd_embeddings.npy` (8961 × 768)

### Step 4: Train Domain Classifier

```bash
python -m scripts.train_classifier
```

Output: `data/models/domain_classifier.joblib` (accuracy: 82.2%)

### Step 5: Evaluate (optional)

```bash
python -m scripts.evaluate_ranking --queries 200
```

---

## API Reference

Start the backend:

```bash
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/info` | GET | Model name, index size, OCR status |
| `/categories` | GET | List all resume categories |
| `/stats/categories` | GET | Per-category resume counts |
| `/rank` | POST | Rank candidates from JD text |
| `/rank/upload` | POST | Rank candidates from JD file upload |
| `/applicant` | POST | Classify resume + gap analysis |
| `/applicant/upload` | POST | Same, with file upload |
| `/jobs/search` | POST | Find matching jobs for resume text |
| `/jobs/search/upload` | POST | Same, with file upload |
| `/extract` | POST | Extract text from any supported file |

### Example

```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{"job_description": "Python developer with Django and PostgreSQL", "top_k": 5}'
```

---

## Frontend (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

### Tabs

| Tab | Purpose | Features |
|-----|---------|----------|
| **Recruiter** | Find candidates for a job | Paste/upload JD, ranked results with Plotly chart |
| **Applicant** | Analyze a resume | Upload resume + optional JD, category prediction, skill gaps, feedback |
| **Find Jobs** | Job recommendations | Upload resume, matching JDs from dataset |
| **Dataset Overview** | Explore the data | Per-category distribution charts |
| **About** | Documentation | Rendered README |

---

## Model Details

### Embedding Model

| Property | Value |
|----------|-------|
| Model | `sentence-transformers/all-mpnet-base-v2` |
| Dimensions | 768 |
| Max Sequence Length | 384 tokens |
| Normalization | L2-normalized (cosine sim = dot product) |

### Domain Classifier

| Property | Value |
|----------|-------|
| Algorithm | Logistic Regression (`max_iter=2000`) |
| Input | 768-dim embedding vector |
| Classes | 40 categories |
| Training Data | 12,459 resumes (80/20 train/test split) |
| Accuracy | **82.2%** |
| Macro F1 | **84.4%** |
| Weighted F1 | **82.2%** |

### Category Merges Applied

Low-support and semantically ambiguous categories were merged to improve classifier performance:

| Original Category | Merged Into | Reason |
|---|---|---|
| Management (F1=0.49) | Consultant | Semantic overlap |
| Operations Manager (F1=0.76) | Consultant | Subset of management |
| Building (F1=0.00, n=1) | Building/Construction | Same domain, insufficient data |
| BPO (F1=0.29, n=6) | Information Technology | Too few samples |
| Avian (F1=0.67) | Aviation | Naming variant |

### Skill Lexicon

8 categories, 100+ skills used for gap analysis:

| Category | Example Skills |
|----------|---------------|
| Languages | Python, Java, C++, JavaScript, R, Go, Rust |
| Web Frameworks | Django, Flask, React, Angular, Vue, Spring |
| Data / ML | TensorFlow, PyTorch, Pandas, Scikit-learn, Spark |
| Cloud / DevOps | AWS, Azure, GCP, Docker, Kubernetes, Terraform |
| Databases | PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch |
| QA / Security | Selenium, Jest, OWASP, Penetration Testing |
| Soft Skills | Leadership, Communication, Problem Solving |
| Business | Agile, Scrum, JIRA, Six Sigma, PMP |

---

## Evaluation & Results

### Classifier Performance (highlights)

**Top performers:**

| Category | F1 | Support |
|----------|-----|---------|
| Health/Fitness | 1.000 | 27 |
| Apparel | 0.974 | 39 |
| DotNet Developer | 0.964 | 40 |
| Blockchain | 0.952 | 11 |
| Public Relations | 0.952 | 42 |
| DevOps Engineer | 0.950 | 59 |

**Weakest categories:**

| Category | F1 | Support | Notes |
|----------|-----|---------|-------|
| Sales | 0.600 | 34 | Overlaps with multiple domains |
| Consultant | 0.691 | 198 | Broad, merged from 3 categories |
| ETL Developer | 0.712 | 56 | Overlaps with Database/Data Science |

### Dataset Scale

| Metric | Count |
|--------|-------|
| Total resumes indexed | 12,459 |
| Resume categories | 40 |
| Job descriptions indexed | 8,961 |
| JD source CSVs | 189 |
| OCR-processed images | 9,327 |

---

## Deployment

### Streamlit Cloud

The app is configured for [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push code to GitHub
2. Connect the repo at [share.streamlit.io](https://share.streamlit.io)
3. Set main file path to `app/streamlit_app.py`
4. Data artifacts (`.npy`, `.joblib`) must be uploaded to the repo or rebuilt on first run

### Local

```bash
# Terminal 1 — backend
uvicorn app.api:app --host 127.0.0.1 --port 8000

# Terminal 2 — frontend
streamlit run app/streamlit_app.py
```

---

## Tests

```bash
pip install pytest
pytest -q
```

Three smoke tests:
- Text preprocessing (whitespace, bullets)
- Skill extraction accuracy
- Gap analysis + feedback generation

---

## Future Work

- **Fine-tuned embedding model** — Domain-adapt `all-mpnet-base-v2` on resume-JD pairs
- **FAISS vector index** — Approximate nearest neighbors for 50k+ scale
- **BERT classifier head** — Replace LogReg with fine-tuned classification layer
- **Structured resume parsing** — Extract name, education, experience, certifications
- **More resume data** — Increase samples for low-support categories
- **Live job scraping** — Real-time JD ingestion from job boards
