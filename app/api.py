"""FastAPI backend exposing HireFormer's matching, classification and gap APIs.

Run:
    uvicorn app.api:app --reload --port 8000
"""
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SETTINGS  # noqa: E402
from src.extraction import is_supported  # noqa: E402
from src.jobs import JobIndex  # noqa: E402
from src.pipeline import (  # noqa: E402
    analyze_applicant,
    load_index,
    load_job_index,
    rank_for_jd,
    text_from_file,
)
from src.preprocessing import clean_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hireformer.api")

app = FastAPI(
    title="HireFormer API",
    description="Transformer-based talent matching and resume ranking.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- models

class RankRequest(BaseModel):
    job_description: str = Field(..., min_length=10)
    top_k: int = Field(10, ge=1, le=100)
    category_filter: Optional[str] = None
    classify: bool = True


class ApplicantRequest(BaseModel):
    resume_text: str = Field(..., min_length=20)
    job_description: Optional[str] = None
    classify: bool = True
    suggest_jobs: bool = False
    jobs_top_k: int = Field(5, ge=1, le=50)


class JobSearchRequest(BaseModel):
    resume_text: str = Field(..., min_length=20)
    top_k: int = Field(10, ge=1, le=100)
    title_contains: Optional[str] = None


# --------------------------------------------------------------------------- helpers

_index_cache = {}


def _get_index():
    if "idx" not in _index_cache:
        _index_cache["idx"] = load_index()
    return _index_cache["idx"]


def _get_job_index() -> JobIndex:
    if "jobs" not in _index_cache:
        _index_cache["jobs"] = load_job_index()
    return _index_cache["jobs"]


def _save_upload_to_tmp(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "upload").suffix or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.file.read())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


# --------------------------------------------------------------------------- static UI

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")


# --------------------------------------------------------------------------- routes

@app.get("/")
def root():
    if _STATIC_DIR.exists():
        return RedirectResponse(url="/ui/")
    return {
        "name": "HireFormer API",
        "version": "0.1.0",
        "endpoints": [
            "/health", "/info",
            "/rank", "/rank/upload",
            "/applicant", "/applicant/upload",
            "/jobs/search", "/jobs/search/upload",
            "/extract", "/categories",
        ],
    }


@app.get("/api")
def api_root():
    return {
        "name": "HireFormer API",
        "docs": "/docs",
        "ui": "/ui/",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    try:
        idx = _get_index()
        n = len(idx)
        cats = sorted(set(idx.categories))
    except FileNotFoundError:
        n, cats = 0, []
    try:
        jobs = _get_job_index()
        n_jobs = len(jobs)
    except FileNotFoundError:
        n_jobs = 0
    return {
        "embedding_model": SETTINGS.embedding_model,
        "index_size": n,
        "job_index_size": n_jobs,
        "categories": cats,
        "ocr_enabled": SETTINGS.enable_ocr,
    }


@app.get("/categories")
def categories():
    try:
        idx = _get_index()
    except FileNotFoundError:
        return {"categories": []}
    return {"categories": sorted(set(idx.categories))}


@app.get("/stats/categories")
def stats_categories():
    """Return per-category resume counts for the dataset chart."""
    try:
        idx = _get_index()
    except FileNotFoundError:
        return []
    from collections import Counter
    counts = Counter(idx.categories)
    return [{"category": c, "count": n} for c, n in counts.most_common()]


@app.post("/rank")
def rank(req: RankRequest):
    try:
        idx = _get_index()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    report = rank_for_jd(
        req.job_description,
        top_k=req.top_k,
        category_filter=req.category_filter,
        index=idx,
        classify=req.classify,
    )
    return report.to_dict()


@app.post("/rank/upload")
def rank_upload(
    file: UploadFile = File(...),
    top_k: int = Form(10),
    category_filter: Optional[str] = Form(None),
    classify: bool = Form(True),
):
    """Upload a JD file (PDF/DOCX/TXT/Image) and rank candidates."""
    if not is_supported(file.filename or ""):
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    tmp = _save_upload_to_tmp(file)
    try:
        jd_text = text_from_file(tmp)
        if len(jd_text) < 20:
            raise HTTPException(
                status_code=422, detail="Could not extract enough text from the file."
            )
        idx = _get_index()
        report = rank_for_jd(
            jd_text,
            top_k=top_k,
            category_filter=category_filter,
            index=idx,
            classify=classify,
        )
        return report.to_dict()
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


@app.post("/applicant")
def applicant(req: ApplicantRequest):
    job_idx = None
    if req.suggest_jobs:
        try:
            job_idx = _get_job_index()
        except FileNotFoundError:
            job_idx = None
    report = analyze_applicant(
        req.resume_text,
        jd_text=req.job_description,
        classify=req.classify,
        suggest_jobs=req.suggest_jobs,
        jobs_top_k=req.jobs_top_k,
        job_index=job_idx,
    )
    return report.to_dict()


@app.post("/applicant/upload")
def applicant_upload(
    resume: UploadFile = File(...),
    job_description: Optional[str] = Form(None),
    classify: bool = Form(True),
    suggest_jobs: bool = Form(False),
    jobs_top_k: int = Form(5),
):
    if not is_supported(resume.filename or ""):
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    tmp = _save_upload_to_tmp(resume)
    try:
        resume_text = text_from_file(tmp)
        if len(resume_text) < 20:
            raise HTTPException(
                status_code=422, detail="Could not extract enough text from the resume."
            )
        job_idx = None
        if suggest_jobs:
            try:
                job_idx = _get_job_index()
            except FileNotFoundError:
                job_idx = None
        report = analyze_applicant(
            resume_text,
            jd_text=job_description,
            classify=classify,
            suggest_jobs=suggest_jobs,
            jobs_top_k=jobs_top_k,
            job_index=job_idx,
        )
        return report.to_dict()
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


@app.post("/jobs/search")
def jobs_search(req: JobSearchRequest):
    """Find the best-matching jobs for a resume (from the JD dataset)."""
    try:
        jobs = _get_job_index()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    matches = jobs.match(
        req.resume_text, top_k=req.top_k, title_contains=req.title_contains
    )
    return {"matches": [m.to_dict() for m in matches]}


@app.post("/jobs/search/upload")
def jobs_search_upload(
    resume: UploadFile = File(...),
    top_k: int = Form(10),
    title_contains: Optional[str] = Form(None),
):
    if not is_supported(resume.filename or ""):
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    tmp = _save_upload_to_tmp(resume)
    try:
        text = text_from_file(tmp)
        jobs = _get_job_index()
        matches = jobs.match(text, top_k=top_k, title_contains=title_contains)
        return {"matches": [m.to_dict() for m in matches]}
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


@app.post("/extract")
def extract(file: UploadFile = File(...)):
    """Utility: just extract + clean text from an uploaded file."""
    if not is_supported(file.filename or ""):
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    tmp = _save_upload_to_tmp(file)
    try:
        text = text_from_file(tmp)
        return {"filename": file.filename, "length": len(text), "text": text}
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
