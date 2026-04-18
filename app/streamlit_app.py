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
from src.embeddings import encode_one  # noqa: E402
from src.gap_analysis import gap_analysis  # noqa: E402

st.set_page_config(
    page_title="HireFormer — AI Talent Matching",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Hide default Streamlit branding ── */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* ── Main container padding ── */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* ── Gradient hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 50%, #06B6D4 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero-banner h1 {
    color: #fff;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-banner p {
    color: rgba(255,255,255,0.85);
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0;
    max-width: 650px;
    line-height: 1.6;
}

/* ── Stat pills in sidebar ── */
.stat-pill {
    background: linear-gradient(135deg, rgba(108,99,255,0.15), rgba(59,130,246,0.10));
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
}
.stat-pill .stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #8B83FF;
    line-height: 1.2;
}
.stat-pill .stat-label {
    font-size: 0.8rem;
    color: rgba(232,230,240,0.6);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
}

/* ── Tabs styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: rgba(19,19,26,0.6);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(108,99,255,0.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-weight: 500;
    font-size: 0.9rem;
    color: rgba(232,230,240,0.6);
    background: transparent;
    border: none;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6C63FF, #3B82F6) !important;
    color: white !important;
    font-weight: 600;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 1.5rem;
    padding-bottom: 0.8rem;
    border-bottom: 2px solid rgba(108,99,255,0.2);
}
.section-header .icon {
    font-size: 1.6rem;
    background: linear-gradient(135deg, rgba(108,99,255,0.2), rgba(59,130,246,0.15));
    border-radius: 10px;
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.section-header h2 {
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0;
    color: #e8e6f0;
}
.section-header p {
    font-size: 0.88rem;
    color: rgba(232,230,240,0.5);
    margin: 0;
}

/* ── Cards ── */
.metric-card {
    background: linear-gradient(145deg, #1a1a2e, #16162a);
    border: 1px solid rgba(108,99,255,0.15);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(108,99,255,0.4);
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF, #3B82F6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-card .metric-label {
    font-size: 0.82rem;
    color: rgba(232,230,240,0.5);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 0.3rem;
    font-weight: 500;
}

/* ── Results card ── */
.result-card {
    background: rgba(19,19,26,0.7);
    border: 1px solid rgba(108,99,255,0.12);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s ease;
}
.result-card:hover {
    border-color: rgba(108,99,255,0.35);
}

/* ── Skill tags ── */
.skill-tag {
    display: inline-block;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 0.2rem;
}
.skill-matched {
    background: rgba(16,185,129,0.15);
    color: #34D399;
    border: 1px solid rgba(16,185,129,0.3);
}
.skill-missing {
    background: rgba(239,68,68,0.12);
    color: #F87171;
    border: 1px solid rgba(239,68,68,0.25);
}
.skill-extra {
    background: rgba(59,130,246,0.12);
    color: #60A5FA;
    border: 1px solid rgba(59,130,246,0.25);
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6C63FF, #3B82F6) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(108,99,255,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 25px rgba(108,99,255,0.4) !important;
    transform: translateY(-1px);
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(108,99,255,0.25) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: border-color 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(108,99,255,0.5) !important;
}

/* ── Text areas ── */
.stTextArea textarea {
    border-radius: 10px !important;
    border: 1px solid rgba(108,99,255,0.2) !important;
    background: rgba(10,10,15,0.6) !important;
    transition: border-color 0.2s ease;
}
.stTextArea textarea:focus {
    border-color: #6C63FF !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,0.15) !important;
}

/* ── Select boxes / number inputs ── */
.stSelectbox [data-baseweb="select"],
.stNumberInput input {
    border-radius: 10px !important;
    border-color: rgba(108,99,255,0.2) !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    border-radius: 10px !important;
    border: 1px solid rgba(108,99,255,0.15) !important;
    font-weight: 500 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d14, #111120) !important;
    border-right: 1px solid rgba(108,99,255,0.1) !important;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 1.1rem;
    font-weight: 700;
    color: rgba(232,230,240,0.7);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 0.75rem;
    margin-top: 1rem;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1a1a2e, #16162a);
    border: 1px solid rgba(108,99,255,0.15);
    border-radius: 14px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] {
    background: linear-gradient(135deg, #6C63FF, #3B82F6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Progress / Spinner ── */
.stSpinner > div {
    border-top-color: #6C63FF !important;
}

/* ── Plotly chart container ── */
[data-testid="stPlotlyChart"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(108,99,255,0.1);
}

/* ── Candidate card ── */
.candidate-card {
    background: linear-gradient(145deg, #1a1a2e, #16162a);
    border: 1px solid rgba(108,99,255,0.15);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 0.9rem;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.candidate-card:hover {
    transform: translateY(-2px);
    border-color: rgba(108,99,255,0.4);
}
.candidate-card .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.7rem;
}
.candidate-card .card-rank {
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF, #3B82F6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    min-width: 36px;
}
.candidate-card .card-file {
    font-size: 1rem;
    font-weight: 600;
    color: #e8e6f0;
    flex: 1;
    margin-left: 0.8rem;
    word-break: break-all;
}
.candidate-card .card-category {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    background: rgba(108,99,255,0.15);
    color: #8B83FF;
    border: 1px solid rgba(108,99,255,0.3);
}
.candidate-card .card-score-bar {
    height: 6px;
    border-radius: 3px;
    background: rgba(108,99,255,0.1);
    margin: 0.5rem 0;
    overflow: hidden;
}
.candidate-card .card-score-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #6C63FF, #3B82F6);
}
.candidate-card .card-score-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: rgba(232,230,240,0.7);
}
.candidate-card .card-snippet {
    font-size: 0.82rem;
    color: rgba(232,230,240,0.45);
    line-height: 1.5;
    margin-top: 0.5rem;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

UPLOAD_TYPES = ["pdf", "docx", "doc", "txt", "png", "jpg", "jpeg", "webp"]

# ── Plotly theme ────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e8e6f0"),
    title_font=dict(size=16, color="#e8e6f0"),
    xaxis=dict(gridcolor="rgba(108,99,255,0.08)", zerolinecolor="rgba(108,99,255,0.15)"),
    yaxis=dict(gridcolor="rgba(108,99,255,0.08)", zerolinecolor="rgba(108,99,255,0.15)"),
    colorway=["#6C63FF", "#3B82F6", "#06B6D4", "#10B981", "#F59E0B", "#EF4444",
              "#8B5CF6", "#EC4899", "#14B8A6", "#F97316"],
    margin=dict(l=40, r=20, t=60, b=40),
)

GRADIENT_COLORS = px.colors.make_colorscale(["#6C63FF", "#3B82F6", "#06B6D4", "#10B981"])


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


def _section_header(icon: str, title: str, subtitle: str = ""):
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="section-header">'
        f'<div class="icon">{icon}</div>'
        f"<div><h2>{title}</h2>{sub}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _metric_cards(metrics: list[tuple[str, str]]):
    cols = st.columns(len(metrics))
    for col, (value, label) in zip(cols, metrics):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def _skill_tags(skills: list[str], css_class: str) -> str:
    return " ".join(
        f'<span class="skill-tag {css_class}">{s}</span>' for s in skills
    ) if skills else "<em style='color:rgba(232,230,240,0.4)'>none</em>"


# ── Hero banner ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-banner">'
    "<h1>🧠 HireFormer</h1>"
    "<p>AI-powered talent matching — classify resumes, rank candidates, "
    "analyze skill gaps, and discover the best-fit jobs using "
    "transformer embeddings.</p>"
    "</div>",
    unsafe_allow_html=True,
)

idx = _try(_index)
jobs = _try(_jobs)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## System")
    if idx is not None:
        n_cats = len(sorted(set(idx.categories)))
        st.markdown(
            f'<div class="stat-pill">'
            f'<div class="stat-value">{len(idx):,}</div>'
            f'<div class="stat-label">Indexed Resumes</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="stat-pill">'
            f'<div class="stat-value">{n_cats}</div>'
            f'<div class="stat-label">Categories</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        with st.expander("View categories"):
            cats = sorted(set(idx.categories))
            for c in cats:
                st.markdown(f"- {c}")
    else:
        st.warning("No resume index found.")

    if jobs is not None:
        st.markdown(
            f'<div class="stat-pill">'
            f'<div class="stat-value">{len(jobs):,}</div>'
            f'<div class="stat-label">Job Descriptions</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No JD index loaded.")

    st.markdown("## Model")
    st.code(SETTINGS.embedding_model, language=None)

tab_recruiter, tab_applicant, tab_jobs, tab_explore, tab_about = st.tabs(
    [
        "Recruiter",
        "Applicant",
        "Job Finder",
        "Analytics",
        "About",
    ]
)


# ── Recruiter tab ───────────────────────────────────────────────────────────────

with tab_recruiter:
    _section_header("👔", "Recruiter", "Find the best candidates for any role")

    mode = st.radio(
        "Mode",
        options=[
            "Score uploaded CVs against a JD",
            "Search the indexed resume database",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── Mode A: upload many CVs + one JD ────────────────────────────────────
    if mode == "Score uploaded CVs against a JD":
        col_jd, col_cv = st.columns(2, gap="large")
        with col_jd:
            st.markdown("##### Job description")
            jd_text_multi = st.text_area(
                "Paste the job description",
                height=220,
                placeholder="Paste the JD here, or upload a file below ...",
                key="multi_jd_text",
                label_visibility="collapsed",
            )
            jd_file_multi = st.file_uploader(
                "Or upload a JD file",
                type=UPLOAD_TYPES,
                key="multi_jd_file",
                help="Supports PDF, DOCX, TXT, and image files",
            )
        with col_cv:
            st.markdown("##### Candidate CVs")
            cv_files = st.file_uploader(
                "Upload one or more CVs",
                type=UPLOAD_TYPES,
                accept_multiple_files=True,
                key="multi_cv_files",
                help="Drop multiple resumes (PDF, DOCX, TXT, images) at once",
            )
            st.caption(
                f"{len(cv_files)} CV(s) selected" if cv_files else "No CVs uploaded yet"
            )

        run_multi = st.button(
            "📊  Score candidates", type="primary",
            key="multi_run_btn", use_container_width=True,
        )

        if run_multi:
            jd_text_resolved = (jd_text_multi or "").strip()
            if not jd_text_resolved and jd_file_multi:
                with st.spinner("Extracting text from JD ..."):
                    p = _save_upload(jd_file_multi)
                    try:
                        jd_text_resolved = text_from_file(p)
                    finally:
                        try:
                            p.unlink()
                        except OSError:
                            pass

            if not jd_text_resolved or len(jd_text_resolved) < 10:
                st.error("Please provide a job description (paste text or upload a file).")
            elif not cv_files:
                st.error("Please upload at least one CV.")
            else:
                from src.preprocessing import clean_text, truncate_words

                jd_clean = truncate_words(clean_text(jd_text_resolved), 1024)

                with st.spinner("Encoding JD ..."):
                    jd_vec = encode_one(jd_clean)

                rows: list[dict] = []
                details: list[dict] = []

                progress = st.progress(0.0, text="Scoring CVs ...")
                for i, up in enumerate(cv_files, start=1):
                    fname = up.name
                    raw_bytes = bytes(up.getbuffer())
                    try:
                        path = _save_upload(up)
                        try:
                            cv_text = text_from_file(path)
                        finally:
                            try:
                                path.unlink()
                            except OSError:
                                pass
                    except Exception as exc:  # noqa: BLE001
                        rows.append({
                            "file": fname,
                            "relevance": 0.0,
                            "skill_match_%": 0.0,
                            "matched": 0,
                            "missing": 0,
                            "matched_skills": "",
                            "missing_skills": f"⚠️ extraction failed: {exc}",
                            "verdict": "Error",
                        })
                        progress.progress(i / len(cv_files), text=f"Scoring {fname} ...")
                        continue

                    if not cv_text or len(cv_text) < 30:
                        rows.append({
                            "file": fname,
                            "relevance": 0.0,
                            "skill_match_%": 0.0,
                            "matched": 0,
                            "missing": 0,
                            "matched_skills": "",
                            "missing_skills": "⚠️ no readable text found",
                            "verdict": "Empty",
                        })
                        progress.progress(i / len(cv_files), text=f"Scoring {fname} ...")
                        continue

                    cv_clean = truncate_words(clean_text(cv_text), 1024)
                    cv_vec = encode_one(cv_clean)
                    relevance = float((jd_vec * cv_vec).sum())  # both normalized

                    gap = gap_analysis(cv_clean, jd_clean)

                    if relevance >= 0.65 and gap.match_percentage >= 60:
                        verdict = "🟢 Strong fit"
                    elif relevance >= 0.45 or gap.match_percentage >= 40:
                        verdict = "🟡 Possible fit"
                    else:
                        verdict = "🔴 Weak fit"

                    rows.append({
                        "file": fname,
                        "relevance": round(relevance, 4),
                        "skill_match_%": round(gap.match_percentage, 1),
                        "matched": len(gap.matched_skills),
                        "missing": len(gap.missing_skills),
                        "matched_skills": ", ".join(gap.matched_skills[:15]),
                        "missing_skills": ", ".join(gap.missing_skills[:15]),
                        "verdict": verdict,
                    })
                    details.append({
                        "file": fname,
                        "bytes": raw_bytes,
                        "relevance": relevance,
                        "gap": gap,
                        "snippet": cv_clean[:600],
                    })
                    progress.progress(i / len(cv_files), text=f"Scoring {fname} ...")
                progress.empty()

                df = pd.DataFrame(rows).sort_values(
                    ["relevance", "skill_match_%"], ascending=False,
                ).reset_index(drop=True)
                df.insert(0, "rank", df.index + 1)

                _metric_cards([
                    (str(len(df)), "CVs scored"),
                    (f"{df['relevance'].max():.3f}" if len(df) else "—", "Best relevance"),
                    (f"{df['relevance'].mean():.3f}" if len(df) else "—", "Avg relevance"),
                    (
                        str(int((df["verdict"] == "🟢 Strong fit").sum())),
                        "Strong fits",
                    ),
                ])
                st.markdown("")

                st.markdown("#### Per-CV report")
                st.dataframe(
                    df,
                    hide_index=True,
                    width='stretch',
                    column_config={
                        "rank": st.column_config.NumberColumn("Rank", width="small"),
                        "relevance": st.column_config.ProgressColumn(
                            "Relevance", min_value=0, max_value=1, format="%.4f",
                            help="Cosine similarity between JD and CV embeddings",
                        ),
                        "skill_match_%": st.column_config.ProgressColumn(
                            "Skill match %", min_value=0, max_value=100, format="%.1f",
                        ),
                        "matched_skills": st.column_config.TextColumn(
                            "Matched skills", width="large",
                        ),
                        "missing_skills": st.column_config.TextColumn(
                            "Missing skills", width="large",
                        ),
                    },
                )

                st.download_button(
                    "⬇️  Download report as CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="recruiter_cv_report.csv",
                    mime="text/csv",
                )

                if len(df):
                    fig = px.bar(
                        df, x="file", y="relevance", color="verdict",
                        title="CV relevance vs JD",
                        hover_data=["skill_match_%", "matched", "missing"],
                    )
                    fig.update_layout(**PLOTLY_LAYOUT, xaxis_tickangle=-45, bargap=0.15)
                    st.plotly_chart(fig, width='stretch')

                if details:
                    st.markdown("#### Download CVs")
                    dl_cols = st.columns(min(len(details), 4))
                    for i, d in enumerate(details):
                        with dl_cols[i % len(dl_cols)]:
                            st.download_button(
                                f"⬇️ {d['file']}",
                                data=d["bytes"],
                                file_name=d["file"],
                                key=f"dl_multi_{i}",
                            )

    # ── Mode B: rank from prebuilt index (legacy behavior) ──────────────────
    else:
        if idx is None:
            st.info("Build the resume index first to enable database search.")
        else:
            col_input, col_opts = st.columns([3, 1], gap="large")
            with col_input:
                jd_text = st.text_area(
                    "Job description",
                    height=240,
                    placeholder="Looking for a Python Developer with 3+ years experience in "
                    "Django, REST APIs, PostgreSQL, Docker, and AWS...",
                    label_visibility="collapsed",
                )
                jd_file = st.file_uploader(
                    "Or upload a JD file",
                    type=UPLOAD_TYPES,
                    help="Supports PDF, DOCX, TXT, and image files",
                )
            with col_opts:
                st.markdown("##### Options")
                top_k = st.slider("Number of candidates", min_value=1, max_value=50, value=10)
                cats = sorted(set(idx.categories))
                cat_filter = st.selectbox("Category filter", options=["All categories"] + cats)
                classify = st.toggle("Predict JD category", value=True)
                st.markdown("")
                run_recruiter = st.button("🚀  Rank candidates", type="primary", use_container_width=True)

            if run_recruiter:
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

                if not text or len(text) < 3:
                    st.error("Please provide a job description (text or file).")
                else:
                    with st.spinner("Encoding JD and ranking candidates ..."):
                        report = rank_for_jd(
                            text,
                            top_k=int(top_k),
                            category_filter=None if cat_filter == "All categories" else cat_filter,
                            index=idx,
                            classify=classify,
                        )

                    if report.predicted_categories:
                        st.markdown("#### Predicted JD categories")
                        cat_df = pd.DataFrame(report.predicted_categories)
                        st.dataframe(
                            cat_df,
                            hide_index=True,
                            width='stretch',
                            column_config={
                                "confidence": st.column_config.ProgressColumn(
                                    "Confidence", min_value=0, max_value=1, format="%.2f"
                                ),
                            },
                        )

                    if not report.candidates:
                        st.warning("No matching candidates found.")
                    else:
                        raw_paths = [c.file for c in report.candidates]
                        scores = [round(c.score, 4) for c in report.candidates]
                        categories = [c.category for c in report.candidates]

                        _metric_cards([
                            (str(len(report.candidates)), "Candidates"),
                            (f"{max(scores):.3f}", "Best Score"),
                            (f"{sum(scores)/len(scores):.3f}", "Avg Score"),
                            (str(len(set(categories))), "Categories"),
                        ])
                        st.markdown("")

                        for cand, raw_p in zip(report.candidates, raw_paths):
                            cv_path = Path(raw_p)
                            fname = cv_path.name
                            pct = int(cand.score * 100)
                            snippet = cand.snippet[:200] if cand.snippet else ""

                            st.markdown(
                                f'<div class="candidate-card">'
                                f'<div class="card-header">'
                                f'<span class="card-rank">#{cand.rank}</span>'
                                f'<span class="card-file">{fname}</span>'
                                f'<span class="card-category">{cand.category}</span>'
                                f'</div>'
                                f'<div class="card-score-bar"><div class="card-score-fill" style="width:{pct}%"></div></div>'
                                f'<span class="card-score-label">Score: {cand.score:.4f}</span>'
                                f'<p class="card-snippet">{snippet}</p>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            if cv_path.exists():
                                st.download_button(
                                    f"⬇️  Download CV",
                                    data=cv_path.read_bytes(),
                                    file_name=fname,
                                    key=f"dl_db_{cand.rank}",
                                )


# ── Applicant tab ───────────────────────────────────────────────────────────────

with tab_applicant:
    _section_header("🎯", "Resume Analyzer", "Get AI-powered feedback on your resume")

    col_resume, col_jd = st.columns(2, gap="large")
    with col_resume:
        st.markdown("##### Upload your resume")
        resume_file = st.file_uploader(
            "Upload your resume",
            type=UPLOAD_TYPES,
            key="applicant_resume",
            help="Supports PDF, DOCX, TXT, and image files",
        )
    with col_jd:
        st.markdown("##### Target job description")
        jd_text_in = st.text_area(
            "Paste job description",
            height=160,
            placeholder="Paste the job description here for gap analysis (optional) ...",
            key="applicant_jd_text",
            label_visibility="collapsed",
        )
        jd_file_applicant = st.file_uploader(
            "Or upload a JD file (optional)",
            type=UPLOAD_TYPES,
            key="applicant_jd_file",
            help="Upload a JD to enable skill gap analysis",
        )

    col_opt, col_btn = st.columns([3, 1])
    with col_opt:
        want_jobs = st.toggle(
            "Suggest matching jobs from JD dataset",
            value=jobs is not None,
            disabled=jobs is None,
        )
    with col_btn:
        run_applicant = st.button(
            "🔎  Analyze resume", type="primary", key="analyze_btn",
            use_container_width=True,
        )

    if run_applicant:
        text = ""
        if resume_file:
            with st.spinner("Extracting resume text ..."):
                p = _save_upload(resume_file)
                try:
                    text = text_from_file(p)
                finally:
                    try:
                        p.unlink()
                    except OSError:
                        pass

        jd_text_extracted = jd_text_in.strip() if jd_text_in else ""
        if not jd_text_extracted and jd_file_applicant:
            with st.spinner("Extracting JD text ..."):
                p2 = _save_upload(jd_file_applicant)
                try:
                    jd_text_extracted = text_from_file(p2)
                finally:
                    try:
                        p2.unlink()
                    except OSError:
                        pass

        if not text or len(text) < 30:
            st.error("Please upload a resume file.")
        else:
            with st.spinner("Analyzing ..."):
                report = analyze_applicant(
                    text,
                    jd_text=jd_text_extracted.strip() or None,
                    classify=True,
                    suggest_jobs=bool(want_jobs and jobs is not None),
                    jobs_top_k=5,
                    job_index=jobs if want_jobs else None,
                )

            if report.predicted_categories:
                st.markdown("#### Predicted resume categories")
                cat_df = pd.DataFrame(report.predicted_categories)
                st.dataframe(
                    cat_df,
                    hide_index=True,
                    width='stretch',
                    column_config={
                        "confidence": st.column_config.ProgressColumn(
                            "Confidence", min_value=0, max_value=1, format="%.2f"
                        ),
                    },
                )

            if report.gap is not None:
                gap = report.gap
                st.markdown("---")
                st.markdown("#### Skill gap analysis")

                _metric_cards([
                    (f"{gap.match_percentage:.0f}%", "Skill Match"),
                    (str(len(gap.matched_skills)), "Matched"),
                    (str(len(gap.missing_skills)), "Missing"),
                    (str(len(gap.extra_skills)), "Extra"),
                ])
                st.markdown("")

                c1, c2, c3 = st.columns(3, gap="medium")
                with c1:
                    st.markdown("##### ✅ Matched skills")
                    st.markdown(
                        _skill_tags(gap.matched_skills, "skill-matched"),
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown("##### ❌ Missing skills")
                    st.markdown(
                        _skill_tags(gap.missing_skills, "skill-missing"),
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown("##### ➕ Extra skills")
                    st.markdown(
                        _skill_tags(gap.extra_skills, "skill-extra"),
                        unsafe_allow_html=True,
                    )

                if report.feedback:
                    st.markdown("---")
                    st.markdown("#### Recommendations")
                    for line in report.feedback:
                        st.markdown(f"- {line}")

            with st.expander("📄  Extracted resume text"):
                st.text_area(
                    "Resume text", value=report.resume_text, height=240, disabled=True
                )


# ── Job Finder tab ──────────────────────────────────────────────────────────────

with tab_jobs:
    _section_header("🧭", "Job Finder", "Discover roles that match your profile")

    if jobs is None:
        st.info("Build the job index first to enable job matching.")
    else:
        col_input, col_opts = st.columns([3, 1], gap="large")
        with col_input:
            st.markdown("##### Upload your resume")
            resume_file2 = st.file_uploader(
                "Upload your resume", type=UPLOAD_TYPES, key="jobs_resume",
                help="Supports PDF, DOCX, TXT, and image files",
            )
        with col_opts:
            st.markdown("##### Options")
            top_k2 = st.slider("Number of jobs", min_value=1, max_value=50, value=10, key="jobs_k")
            title_contains = st.text_input(
                "Title filter", placeholder="e.g. python developer",
            )
            st.markdown("")
            run_jobs = st.button(
                "🧭  Find jobs", type="primary", use_container_width=True,
            )

        if run_jobs:
            text = ""
            if resume_file2:
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
                st.error("Please upload a resume file.")
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

                    _metric_cards([
                        (str(len(df)), "Jobs Found"),
                        (f"{df['score'].max():.3f}", "Best Match"),
                        (f"{df['score'].mean():.3f}", "Avg Score"),
                    ])
                    st.markdown("")

                    st.dataframe(
                        df[["rank", "score", "title", "description"]],
                        hide_index=True,
                        width='stretch',
                        column_config={
                            "rank": st.column_config.NumberColumn("Rank", width="small"),
                            "score": st.column_config.ProgressColumn(
                                "Score", min_value=0, max_value=1, format="%.4f"
                            ),
                        },
                    )
                    with st.expander("🔍  Inspect a job"):
                        pick = st.selectbox("Pick a rank", [m.rank for m in matches])
                        m = matches[int(pick) - 1]
                        c1, c2 = st.columns(2)
                        c1.metric("Title", m.title)
                        c2.metric("Score", f"{m.score:.4f}")
                        st.text_area(
                            "Description", value=m.description, height=260, disabled=True,
                        )


# ── Analytics tab ───────────────────────────────────────────────────────────────

with tab_explore:
    _section_header("📊", "Analytics", "Explore the dataset at a glance")

    if idx is None:
        st.info("Build the resume index to see category distribution.")
    else:
        cats_df = (
            pd.Series(idx.categories, name="count")
            .value_counts()
            .rename_axis("category")
            .reset_index()
        )

        _metric_cards([
            (f"{len(idx):,}", "Total Resumes"),
            (str(cats_df.shape[0]), "Categories"),
            (str(cats_df["count"].max()), "Largest Category"),
            (str(cats_df["count"].min()), "Smallest Category"),
        ])
        st.markdown("")

        fig = px.bar(
            cats_df.head(50), x="category", y="count",
            title="Resume distribution by category",
            color="count",
            color_continuous_scale=[[0, "#6C63FF"], [0.5, "#3B82F6"], [1, "#06B6D4"]],
        )
        fig.update_layout(**PLOTLY_LAYOUT, xaxis_tickangle=-45, bargap=0.2)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

        with st.expander("📋  Full category table"):
            st.dataframe(cats_df, hide_index=True, width='stretch')

    if jobs is not None:
        st.markdown("---")
        st.markdown("#### Job descriptions")
        jt_df = (
            pd.Series(jobs.titles, name="count")
            .value_counts()
            .rename_axis("title")
            .reset_index()
        )

        _metric_cards([
            (f"{len(jobs):,}", "Total JDs"),
            (str(jt_df.shape[0]), "Distinct Titles"),
        ])
        st.markdown("")

        fig2 = px.bar(
            jt_df.head(30), x="title", y="count",
            title="Job descriptions by title",
            color="count",
            color_continuous_scale=[[0, "#8B5CF6"], [0.5, "#EC4899"], [1, "#F97316"]],
        )
        fig2.update_layout(**PLOTLY_LAYOUT, xaxis_tickangle=-45, bargap=0.2)
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig2, width='stretch')

        with st.expander("📋  Full JD table"):
            st.dataframe(jt_df, hide_index=True, width='stretch')


# ── About tab ──────────────────────────────────────────────────────────────────

with tab_about:
    readme = ROOT / "README.md"
    if readme.exists():
        st.markdown(readme.read_text(encoding="utf-8"))
    else:
        st.markdown("### HireFormer — see `README.md` for details.")
