# HireFormer

Transformer-based talent matching, resume domain classification, and gap analysis.

> Implementation of the HireFormer graduation project (see [PROJECT_DOCUMENTATION.md](../Resumes%20Datasets/PROJECT_DOCUMENTATION.md)).

## Features

- **Text extraction** from PDF, DOCX, TXT, and images (Tesseract OCR).
- **Semantic ranking** of resumes against any job description using `sentence-transformers/all-mpnet-base-v2`.
- **43-class domain classifier** (Logistic Regression on top of embeddings).
- **Skill extraction + gap analysis** with actionable applicant feedback.
- **FastAPI** backend (`/rank`, `/applicant`, `/extract`, ...).
- **Streamlit** frontend with two views: Recruiter (rank candidates) and Applicant (resume feedback).

## Project layout

```
resume-project/
├── app/
│   ├── api.py              # FastAPI backend
│   └── streamlit_app.py    # Streamlit frontend
├── scripts/
│   ├── build_embeddings.py # Walk dataset → extract → embed → save
│   ├── train_classifier.py # Train domain classifier on embeddings
│   └── dataset_summary.py  # Per-category file counts
├── src/
│   ├── config.py           # Paths & runtime settings
│   ├── dataset.py          # Walks the raw resume sources
│   ├── extraction.py       # PDF / DOCX / OCR text extraction
│   ├── preprocessing.py    # Text cleaning + truncation
│   ├── embeddings.py       # SentenceTransformer wrapper
│   ├── ranking.py          # CandidateIndex + ranking
│   ├── classifier.py       # Domain classifier (train/load/predict)
│   ├── gap_analysis.py     # Skill extraction, gap analysis, feedback
│   └── pipeline.py         # High-level orchestration
├── tests/
│   └── test_basic.py
├── data/                   # (created on first run; embeddings, models, cache)
├── requirements.txt
└── README.md
```

## 1. Install

```powershell
cd resume-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # optional, only if you extend NER
```

For OCR on image-only resumes, install **Tesseract OCR**:
- Windows: <https://github.com/UB-Mannheim/tesseract/wiki>
- Then either add it to `PATH` or set `HF_TESSERACT_CMD` to the executable.

## 2. Configure data sources

Raw resume folders are read from [src/config.py](src/config.py) (`DEFAULT_RAW_SOURCES`):

```
C:\Users\devmu\Downloads\Resumes Datasets\resume_database
C:\Users\devmu\Downloads\Resumes Datasets\Scrapped_Resumes
C:\Users\devmu\Downloads\Resumes Datasets\Bing_images
```

Adjust if your data lives elsewhere.

## 3. Build the index

Quick smoke test (≤30 docs per category, skips OCR — fast):

```powershell
python -m scripts.build_embeddings --skip-ocr --limit 30
```

Full build (uses OCR — slow but covers image-only resumes):

```powershell
python -m scripts.build_embeddings
```

Outputs:
- `data/embeddings/resume_embeddings.npy`
- `data/embeddings/resume_metadata.json`
- `data/embeddings/resume_texts.json`

## 4. Train the domain classifier

```powershell
python -m scripts.train_classifier
```

Saves `data/models/domain_classifier.joblib` + a JSON classification report.

## 5. Run the apps

**Frontend (Streamlit):**

```powershell
streamlit run app/streamlit_app.py
```

**Backend (FastAPI):**

```powershell
uvicorn app.api:app --reload --port 8000
```

Open <http://localhost:8000/docs> for the interactive Swagger UI.

## 6. API quick reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/info` | GET | Model + index status |
| `/categories` | GET | Categories present in the index |
| `/rank` | POST | Rank candidates from JD text |
| `/rank/upload` | POST | Same, but upload a JD file |
| `/applicant` | POST | Classify resume + optional gap analysis |
| `/applicant/upload` | POST | Same, with resume file upload |
| `/extract` | POST | Extract text from any supported file |

Example:

```powershell
curl -X POST http://localhost:8000/rank `
  -H "Content-Type: application/json" `
  -d '{"job_description":"Python developer with Django and PostgreSQL","top_k":5}'
```

## 7. Tests

```powershell
pip install pytest
pytest -q
```

## Notes / next steps

- The classifier is a fast baseline (LogReg on embeddings). Drop in a fine-tuned BERT head per the project doc for higher accuracy.
- Replace the brute-force ranker with FAISS once the index grows past ~50k vectors.
- Job description scraping is left as a Phase 1 follow-up — see `PROJECT_DOCUMENTATION.md`.
