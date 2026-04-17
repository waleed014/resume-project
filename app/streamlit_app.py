"""Streamlit frontend for HireFormer.

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SETTINGS  # noqa: E402
from src.pipeline import (  # noqa: E402
    analyze_applicant,
    load_index,
    load_job_index,
    rank_for_jd,
    text_from_file,
)

st.set_page_config(
    page_title="HireFormer — Talent Matching",
    page_icon="🧠",
    layout="wide",
)

UPLOAD_TYPES = ["pdf", "docx", "doc", "txt", "png", "jpg", "jpeg", "webp"]


@st.cache_resource(show_spinner="Loading resume index ...")
def _index():
    return load_index()


@st.cache_resource(show_spinner="Loading job-description index ...")
def _jobs():
    return load_job_index()


def _save_upload(upload) -> Path:
    suffix = Path(upload.name).suffix or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.getbuffer())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _try(getter):
    try:
        return getter()
    except FileNotFoundError:
        return None


st.title("🧠 HireFormer")
st.caption(
    "Transformer-based talent matching, resume domain classification, "
    "gap analysis and job recommendations."
)

idx = _try(_index)
jobs = _try(_jobs)

with st.sidebar:
    st.header("⚙️ Status")
    if idx is not None:
        st.success(f"Resume index: **{len(idx):,}** resumes")
        cats = sorted(set(idx.categories))
        st.write(f"Categories: {len(cats)}")
        with st.expander("Show categories"):
            st.write(", ".join(cats))
    else:
        st.warning(
            "No resume index yet.\n\n"
            "`python -m scripts.build_embeddings --skip-ocr --limit 30`"
        )

    if jobs is not None:
        st.success(f"Job index: **{len(jobs):,}** JDs")
    else:
        st.info(
            "No JD index.\n\n"
            "`python -m scripts.build_job_index --limit 500`"
        )

    st.write(f"Model: `{SETTINGS.embedding_model}`")

tab_recruiter, tab_applicant, tab_jobs, tab_explore, tab_about = st.tabs(
    [
        "👔 Recruiter",
        "🎯 Applicant",
        "🧭 Find jobs for my resume",
        "📊 Dataset overview",
        "📖 About",
    ]
)


# ---------------------- Recruiter ------------------------------------------------

with tab_recruiter:
    st.subheader("Find the best candidates for a job description")

    if idx is None:
        st.info("Build the resume index first to enable matching.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            jd_text = st.text_area(
                "Paste a job description",
                height=240,
                placeholder="Looking for a Python Developer with 3+ years experience in "
                "Django, REST APIs, PostgreSQL, Docker, and AWS...",
            )
        with col2:
            top_k = st.number_input("Top K", min_value=1, max_value=50, value=10)
            cats = sorted(set(idx.categories))
            cat_filter = st.selectbox("Filter by category", options=["(any)"] + cats)
            classify = st.checkbox("Also predict JD's category", value=True)

        jd_file = st.file_uploader(
            "...or upload a JD file (PDF/DOCX/TXT/Image)",
            type=UPLOAD_TYPES,
        )

        if st.button("🚀 Rank candidates", type="primary"):
            text = jd_text.strip()
            if jd_file and not text:
                with st.spinner("Extracting text from JD ..."):
                    p = _save_upload(jd_file)
                    try:
                        text = text_from_file(p)
                    finally:
                        try:
                            p.unlink()
                        except OSError:
                            pass

            if not text or len(text) < 20:
                st.error("Please provide a job description (text or file).")
            else:
                with st.spinner("Encoding JD and ranking candidates ..."):
                    report = rank_for_jd(
                        text,
                        top_k=int(top_k),
                        category_filter=None if cat_filter == "(any)" else cat_filter,
                        index=idx,
                        classify=classify,
                    )

                if report.predicted_categories:
                    st.markdown("**Predicted JD categories**")
                    st.dataframe(
                        pd.DataFrame(report.predicted_categories),
                        hide_index=True,
                        width='stretch',
                    )

                if not report.candidates:
                    st.warning("No matching candidates found.")
                else:
                    df = pd.DataFrame([c.to_dict() for c in report.candidates])
                    df["score"] = df["score"].round(4)
                    df["file"] = df["file"].apply(lambda p: Path(p).name)
                    st.markdown(f"### Top {len(df)} candidates")
                    st.dataframe(
                        df[["rank", "score", "category", "file", "snippet"]],
                        hide_index=True,
                        width='stretch',
                    )
                    fig = px.bar(
                        df, x="file", y="score", color="category",
                        title="Similarity scores",
                    )
                    st.plotly_chart(fig, width='stretch')

                    with st.expander("Inspect a candidate"):
                        choice = st.selectbox(
                            "Pick a rank", options=[c.rank for c in report.candidates]
                        )
                        cand = report.candidates[int(choice) - 1]
                        st.write(f"**File:** `{cand.file}`")
                        st.write(f"**Category:** {cand.category}")
                        st.write(f"**Score:** {cand.score:.4f}")
                        st.text_area("Snippet", value=cand.snippet, height=180, disabled=True)


# ---------------------- Applicant ------------------------------------------------

with tab_applicant:
    st.subheader("Get feedback on your resume")

    col1, col2 = st.columns(2)
    with col1:
        resume_file = st.file_uploader(
            "Upload your resume (PDF/DOCX/Image)",
            type=UPLOAD_TYPES,
            key="applicant_resume",
        )
        resume_text_in = st.text_area(
            "...or paste resume text", height=220, key="applicant_resume_text",
        )
    with col2:
        jd_text_in = st.text_area(
            "Target job description (optional, enables gap analysis)",
            height=300,
            key="applicant_jd",
        )

    want_jobs = st.checkbox(
        "Also suggest matching jobs from the JD dataset",
        value=jobs is not None,
        disabled=jobs is None,
    )

    if st.button("🔎 Analyze resume", type="primary", key="analyze_btn"):
        text = resume_text_in.strip()
        if resume_file and not text:
            with st.spinner("Extracting resume text ..."):
                p = _save_upload(resume_file)
                try:
                    text = text_from_file(p)
                finally:
                    try:
                        p.unlink()
                    except OSError:
                        pass

        if not text or len(text) < 30:
            st.error("Please provide a resume (file or text).")
        else:
            with st.spinner("Analyzing ..."):
                report = analyze_applicant(
                    text,
                    jd_text=jd_text_in.strip() or None,
                    classify=True,
                    suggest_jobs=bool(want_jobs and jobs is not None),
                    jobs_top_k=5,
                    job_index=jobs if want_jobs else None,
                )
            if report.predicted_categories:
                st.markdown("**Predicted resume categories**")
                st.dataframe(
                    pd.DataFrame(report.predicted_categories),
                    hide_index=True,
                    width='stretch',
                )

            if report.gap is not None:
                gap = report.gap
                st.markdown("### Gap analysis")
                st.metric("Skill match", f"{gap.match_percentage:.1f}%")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**✅ Matched**")
                    st.write(", ".join(gap.matched_skills) or "_none_")
                with c2:
                    st.markdown("**❌ Missing**")
                    st.write(", ".join(gap.missing_skills) or "_none_")
                with c3:
                    st.markdown("**➕ Extra**")
                    st.write(", ".join(gap.extra_skills) or "_none_")

                st.markdown("### Feedback")
                for line in report.feedback:
                    st.write(f"- {line}")

            with st.expander("Extracted resume text"):
                st.text_area(
                    "Resume text", value=report.resume_text, height=240, disabled=True
                )


# ---------------------- Jobs-for-resume -----------------------------------------

with tab_jobs:
    st.subheader("Find matching jobs for a resume")

    if jobs is None:
        st.info(
            "Build the job index first:\n\n"
            "`python -m scripts.build_job_index --limit 500`"
        )
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            resume_file2 = st.file_uploader(
                "Upload your resume", type=UPLOAD_TYPES, key="jobs_resume"
            )
            resume_text2 = st.text_area(
                "...or paste resume text", height=220, key="jobs_resume_text",
            )
        with col2:
            top_k2 = st.number_input("Top K jobs", min_value=1, max_value=50, value=10)
            title_contains = st.text_input(
                "Title filter (substring)", placeholder="e.g. python",
            )

        if st.button("🧭 Find jobs", type="primary"):
            text = resume_text2.strip()
            if resume_file2 and not text:
                with st.spinner("Extracting resume text ..."):
                    p = _save_upload(resume_file2)
                    try:
                        text = text_from_file(p)
                    finally:
                        try:
                            p.unlink()
                        except OSError:
                            pass

            if not text or len(text) < 30:
                st.error("Please provide a resume (file or text).")
            else:
                with st.spinner("Matching ..."):
                    matches = jobs.match(
                        text,
                        top_k=int(top_k2),
                        title_contains=title_contains.strip() or None,
                    )
                if not matches:
                    st.warning("No matching jobs found.")
                else:
                    df = pd.DataFrame([m.to_dict() for m in matches])
                    df["score"] = df["score"].round(4)
                    st.dataframe(
                        df[["rank", "score", "title", "description"]],
                        hide_index=True,
                        width='stretch',
                    )
                    with st.expander("Inspect a job"):
                        pick = st.selectbox("Pick a rank", [m.rank for m in matches])
                        m = matches[int(pick) - 1]
                        st.markdown(f"**{m.title}** — score={m.score:.4f}")
                        st.text_area(
                            "Description", value=m.description, height=260, disabled=True,
                        )


# ---------------------- Dataset overview ---------------------------------------

with tab_explore:
    st.subheader("Dataset overview")

    if idx is None:
        st.info("Build the resume index to see category distribution.")
    else:
        cats_df = (
            pd.Series(idx.categories, name="count")
            .value_counts()
            .rename_axis("category")
            .reset_index()
        )
        fig = px.bar(
            cats_df.head(50), x="category", y="count",
            title=f"Resume distribution ({len(idx):,} resumes, {cats_df.shape[0]} categories)",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width='stretch')
        st.dataframe(cats_df, hide_index=True, width='stretch')

    if jobs is not None:
        st.markdown("### Job descriptions")
        jt_df = (
            pd.Series(jobs.titles, name="count")
            .value_counts()
            .rename_axis("title")
            .reset_index()
        )
        st.write(
            f"Loaded **{len(jobs):,}** JDs across **{jt_df.shape[0]}** distinct titles."
        )
        st.dataframe(jt_df.head(50), hide_index=True, width='stretch')


# ---------------------- About ---------------------------------------------------

with tab_about:
    readme = ROOT / "README.md"
    if readme.exists():
        st.markdown(readme.read_text(encoding="utf-8"))
    else:
        st.markdown("### HireFormer — see `README.md` for details.")
