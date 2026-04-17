# HireFormer — Progress Tracker

This document mirrors the original [PROJECT_DOCUMENTATION.md](../Resumes%20Datasets/PROJECT_DOCUMENTATION.md) and tracks every deliverable as a checkbox. It is updated as the build proceeds.

**Legend**
- [x] Done — code / artifact exists in this repo
- [ ] Pending — not implemented yet
- [~] Partial — scaffolded but needs further work (documented below)

---

## 1. Core Objectives

- [x] **Semantic Resume–Job Matching** — `src/ranking.py` (`CandidateIndex.rank` using cosine similarity on MPNet embeddings) and `src/pipeline.py::rank_for_jd`.
- [x] **Resume Domain Classification** — `src/classifier.py` (Logistic Regression baseline on embeddings, exposed via `/applicant` and in Streamlit).
- [x] **Missing Qualification Detection** — `src/gap_analysis.py::gap_analysis` (matched / missing / extra skills + match %).
- [x] **Applicant Feedback Generation** — `src/gap_analysis.py::generate_feedback` + Streamlit Applicant tab.
- [x] **Reduce Time-to-Hire prototype** — FastAPI backend (`app/api.py`) + Streamlit app (`app/streamlit_app.py`) give a working end-to-end product.

---

## 2. Dataset

- [x] `resume_database/` ingest (21 categories, .docx / .png) — handled by `src/dataset.py::iter_resume_files`.
- [x] `Scrapped_Resumes/` ingest (43 categories) — same walker.
- [x] `Bing_images/` ingest (image-only resumes) — OCR via `pytesseract` in `src/extraction.py`.
- [ ] `archive/Resumes PDF/` — **intentionally excluded per user decision** (removed from `DEFAULT_RAW_SOURCES`).
- [x] 43-category normalization — `src/config.py::CATEGORY_ALIASES` maps folder names like `"DotNet Developer resumes"` → `"DotNet Developer"`.
- [x] **Real-world JD dataset** (`job_title_des.csv`, 53k+ rows) — `src/jobs.py` + `scripts/build_job_index.py` (replaces the "scrape JDs" task).

---

## 3. Module Breakdown

| Module | Status | Location |
|---|---|---|
| Text Extraction (PDF/DOCX/TXT/Image) | [x] | [src/extraction.py](resume-project/src/extraction.py) |
| Preprocessing (clean / normalize / truncate) | [x] | [src/preprocessing.py](resume-project/src/preprocessing.py) |
| Transformer Encoder (`all-mpnet-base-v2`) | [x] | [src/embeddings.py](resume-project/src/embeddings.py) |
| Similarity Ranking (cosine, optional FAISS) | [x] | [src/ranking.py](resume-project/src/ranking.py) |
| Domain Classifier (43-class LogReg) | [x] | [src/classifier.py](resume-project/src/classifier.py) |
| Gap Analyzer | [x] | [src/gap_analysis.py](resume-project/src/gap_analysis.py) |
| Feedback Generator | [x] | `generate_feedback` in [src/gap_analysis.py](resume-project/src/gap_analysis.py) |
| Web Interface — Streamlit | [x] | [app/streamlit_app.py](resume-project/app/streamlit_app.py) |
| Web Interface — FastAPI | [x] | [app/api.py](resume-project/app/api.py) |

---

## 4. Model Details

- [x] Base encoder: `sentence-transformers/all-mpnet-base-v2` (configured in `SETTINGS.embedding_model`).
- [x] Bi-encoder embeddings with cosine similarity.
- [x] Linear classification head (LogisticRegression on 768-d embeddings).
- [ ] **Contrastive fine-tuning** with `CosineSimilarityLoss` / `MultipleNegativesRankingLoss` — scripted path deferred (Phase 2 stretch).
- [ ] **Cross-encoder re-ranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) — deferred.

---

## 5. Implementation

- [x] `requirements.txt` with all required packages.
- [x] Virtual environment created at `resume-project/.venv` and all dependencies installed successfully.
- [x] `src/extraction.py::extract_text` covers `.pdf` / `.docx` / `.doc` / `.txt` / `.png` / `.jpg` / `.jpeg` / `.webp` / `.bmp` / `.tiff`.
- [x] `scripts/build_embeddings.py` — batch embedding generation with `--limit`, `--skip-ocr`, `--sources` flags.
- [x] `rank_candidates` equivalent — `CandidateIndex.rank` with optional category filter and FAISS backend.
- [x] Domain classifier training script — `scripts/train_classifier.py` with train/test split and classification report.
- [x] Gap analysis + feedback — see modules above.
- [x] Tesseract OCR supported via `pytesseract` (system install remains a user responsibility on Windows).

---

## 6. Example Output

- [x] Candidate ranking JSON matches the documented schema (rank / score / category / file / snippet) — see `RankedCandidate.to_dict()`.
- [x] Gap analysis JSON matches the documented schema (matched / missing / extra + match_percentage) — see `GapResult.to_dict()`.

---

## 7. Evaluation Metrics

- [x] `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `average_precision`, `mean_average_precision` — [src/evaluation.py](resume-project/src/evaluation.py).
- [x] `summarize(queries_relevant, queries_retrieved, ks=(1, 5, 10, 20))` aggregator.
- [x] `scripts/evaluate_ranking.py` — self-evaluation using category as the relevance signal on a random sample of N query resumes.
- [x] Classification accuracy / F1 — produced by `scripts/train_classifier.py` via `classification_report`.
- [ ] Target thresholds (P@10 ≥ 0.70, NDCG ≥ 0.75, MAP ≥ 0.65, classifier ≥ 0.85) — to be measured against the full index once it is built.

---

## 8. Fine-Tuning (Phase 2)

- [ ] Contrastive fine-tuning script — deferred.
- [ ] BERT classification fine-tuning — deferred (baseline LogReg in place).

---

## 9. Project Structure

This repository follows the proposed structure (with small pragmatic tweaks):

```
resume-project/
├── src/                 # extraction, preprocessing, embeddings, ranking,
│                        # classifier, gap_analysis, jobs, evaluation, pipeline
├── app/                 # streamlit_app.py, api.py
├── scripts/             # build_embeddings.py, train_classifier.py,
│                        # dataset_summary.py, build_job_index.py,
│                        # evaluate_ranking.py
├── tests/               # test_basic.py
├── data/                # (auto-created) embeddings, metadata, JD index, models
├── requirements.txt
├── README.md
└── PROGRESS.md          # this file
```

Notebook folder (`notebooks/01..04`) is not yet created — analysis is done via scripts / Streamlit "Dataset overview" tab instead.

- [x] `src/` with nine modules.
- [x] `app/` with Streamlit + FastAPI.
- [x] `scripts/` with five utility scripts.
- [x] `tests/` with basic smoke tests.
- [ ] `notebooks/` (01 data exploration, 02 embedding analysis, 03 model training, 04 evaluation) — optional.

---

## 10. Implementation Plan

### Phase 1 — Data Preparation & Exploration (Weeks 1–3)

- [x] Audit / normalize folder names across sources.
- [x] Dev environment (Python 3.13 venv, dependencies installed).
- [x] Text-extraction pipeline (PDF / DOCX / OCR).
- [x] Preprocessing module.
- [~] Exploratory data analysis — covered by the Streamlit **Dataset overview** tab + `scripts/dataset_summary.py`, but no dedicated `01_data_exploration.ipynb`.
- [x] Job descriptions collected — replaced by the pre-built `job_title_des.csv` (53k+ JDs).

### Phase 2 — Core Model Development (Weeks 4–7)

- [x] Build resume embeddings (`scripts/build_embeddings.py`).
- [x] Cosine-similarity ranker (`src/ranking.py`).
- [x] Domain classifier — baseline LogReg on embeddings.
- [x] Baseline evaluation tooling (`src/evaluation.py`, `scripts/evaluate_ranking.py`).
- [ ] Contrastive fine-tuning (deferred).
- [ ] BERT classifier fine-tuning (deferred).
- [ ] Fine-tuned vs baseline comparison (deferred).
- [ ] Cross-encoder re-ranking (deferred).

### Phase 3 — Advanced Features (Weeks 8–10)

- [x] Skill extraction (`src/gap_analysis.py::extract_skills`, lexicon-based).
- [x] Gap analysis.
- [x] Feedback generation.
- [x] FAISS vector index (opt-in via `SETTINGS.use_faiss`).
- [x] End-to-end pipeline (`src/pipeline.py`).

### Phase 4 — Interface & Evaluation (Weeks 11–13)

- [x] Streamlit recruiter view (Tab: **👔 Recruiter**).
- [x] Streamlit applicant view (Tab: **🎯 Applicant**, with optional "suggest matching jobs").
- [x] Streamlit "Find jobs for my resume" (Tab: **🧭**, uses `job_title_des.csv`).
- [x] Streamlit dataset overview (Tab: **📊**, Plotly bar chart).
- [x] Streamlit About (renders `README.md`).
- [x] FastAPI endpoints — `/rank`, `/rank/upload`, `/applicant`, `/applicant/upload`, `/classify`, `/extract`, `/jobs/search`, `/jobs/search/upload`, `/info`, `/health`.
- [ ] **Full-scale evaluation report across all 43 categories** — pending a full-index build.
- [ ] Ablation study — pending fine-tuning work.
- [x] Documentation — `README.md` + this file.
- [ ] Final project presentation / report — out of scope for this repo.

### Milestones

- [~] **M1: Data Ready** — code and tooling are ready; actually running the full build for all 21,600 files is left to the user (`scripts.build_embeddings` without `--limit`).
- [x] **M2: Baseline Working** — end-to-end ranking path is functional.
- [ ] **M3: Fine-Tuned Models** — deferred.
- [x] **M4: Full Feature Set** — ranking + classification + gap analysis + feedback + JD matching.
- [~] **M5: Prototype Complete** — web interface + tooling done; final evaluation report + presentation pending.

---

## Quick "what's left" summary

1. Run a full embedding build (no `--limit`) and a full JD-index build.
2. Run `scripts/evaluate_ranking.py` against the full index and record the report.
3. (Optional) Fine-tune the bi-encoder with contrastive pairs; add a cross-encoder re-ranker.
4. (Optional) Port the ad-hoc exploration into `notebooks/01..04`.
5. (Optional) Produce the final presentation deliverable.
