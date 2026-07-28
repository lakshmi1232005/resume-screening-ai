"""
app.py
========
Phase 10 -- Final integration. Wires together every prior phase into a
single Streamlit app:

    Phase 2  ResumeParser          -- parse uploaded resume PDF
    Phase 3  TextPreprocessor      -- used internally by Phase 4/5
    Phase 4  SemanticMatcher       -- resume <-> JD embedding match %
    Phase 5  ATSScorer             -- rule-based 0-100 ATS score
    Phase 6  SkillGapAnalyzer      -- missing skills vs the JD
    Phase 7  ImprovementEngine     -- prioritized improvement suggestions
    Phase 8  DashboardBuilder      -- Plotly charts for all of the above
    Phase 9  ReportGenerator       -- downloadable PDF version of the dashboard

Two modes:
    - Single Candidate: full dashboard + suggestions + PDF download for
      one resume against one JD.
    - Multi-Candidate Ranking: screen several resumes against one JD at
      once and see DashboardBuilder's leaderboard, with per-candidate
      PDF reports available on demand.

Why Streamlit re-runs the whole script on every interaction, and why
that's safe here:
    Every module in this project is stateless (see their own
    docstrings), and the two model loaders (models/nlp_loader.py,
    models/embedding_loader.py) use @lru_cache so the actual expensive
    model loading only happens once per process, not once per rerun.
"""

import os
import time

import streamlit as st

from config import ALLOWED_JD_EXTENSIONS, ALLOWED_RESUME_EXTENSIONS, UPLOADS_DIR
from modules.ats_scorer import ATSScorer
from modules.dashboard import DashboardBuilder
from modules.improvement_engine import ImprovementEngine
from modules.report_generator import ReportGenerator
from modules.resume_parser import ResumeParser
from modules.semantic_matcher import SemanticMatcher
from modules.skill_gap_analyzer import SkillGapAnalyzer
from utils.pdf_utils import PDFExtractionError, extract_text_from_pdf, extract_text_from_txt
from utils.validators import FileValidationError, validate_file

st.set_page_config(page_title="AI Resume Screening System", page_icon="\U0001F4C4", layout="wide")

# ----------------------------------------------------------------------
# STATELESS PIPELINE OBJECTS -- created once per rerun, cheap to build.
# The actual expensive ML models they lean on are cached in
# models/nlp_loader.py and models/embedding_loader.py via @lru_cache.
# ----------------------------------------------------------------------
parser = ResumeParser()
matcher = SemanticMatcher()
scorer = ATSScorer()
gap_analyzer = SkillGapAnalyzer()
improvement_engine = ImprovementEngine()
dashboard = DashboardBuilder()
report_generator = ReportGenerator()


# ------------------------------------------------------------------
# SHARED HELPERS
# ------------------------------------------------------------------
def save_upload(uploaded_file, allowed_extensions: tuple) -> str:
    """
    Persists a Streamlit UploadedFile to UPLOADS_DIR and validates it,
    since every downstream module (ResumeParser, pdf_utils) expects a
    real file path on disk, not an in-memory buffer.

    Raises:
        FileValidationError: via validate_file, on a bad file.
    """
    # Prefix with a timestamp so two people uploading "resume.pdf" in
    # the same session don't clobber each other's file on disk.
    safe_name = f"{int(time.time() * 1000)}_{uploaded_file.name}"
    dest_path = os.path.join(UPLOADS_DIR, safe_name)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    validate_file(dest_path, allowed_extensions)
    return dest_path


def get_jd_text(jd_file, jd_pasted_text: str) -> str:
    """Resolves the job description to raw text, from an upload or pasted text."""
    if jd_file is not None:
        jd_path = save_upload(jd_file, ALLOWED_JD_EXTENSIONS)
        if jd_path.lower().endswith(".pdf"):
            return extract_text_from_pdf(jd_path)
        return extract_text_from_txt(jd_path)
    return jd_pasted_text.strip()


def run_pipeline(resume_path: str, jd_text: str) -> dict:
    """
    Runs Phases 2, 4, 5, 6, 7 for one resume against one JD.

    Returns:
        dict: {"resume_data", "match_score", "ats_result", "gap_result",
               "improvement_result"} -- everything DashboardBuilder and
               ReportGenerator need.
    """
    resume_data = parser.parse(resume_path)
    match_score = matcher.compute_match_percentage(resume_data["raw_text"], jd_text)
    ats_result = scorer.compute_ats_score(resume_data, jd_text)
    gap_result = gap_analyzer.analyze(resume_data["skills"], jd_text)
    improvement_result = improvement_engine.generate_suggestions(resume_data, jd_text)
    return {
        "resume_data": resume_data,
        "match_score": match_score,
        "ats_result": ats_result,
        "gap_result": gap_result,
        "improvement_result": improvement_result,
    }


def render_single_candidate_dashboard(result: dict, jd_title: str):
    """Renders the full score/chart/suggestion dashboard for one candidate."""
    resume_data = result["resume_data"]
    match_score = result["match_score"]
    ats_result = result["ats_result"]
    gap_result = result["gap_result"]
    improvement_result = result["improvement_result"]
    final_score = dashboard.compute_final_score(match_score, ats_result["overall_score"])

    st.subheader(resume_data.get("name") or "Candidate")
    contact_bits = [b for b in (resume_data.get("email"), resume_data.get("phone")) if b]
    if contact_bits:
        st.caption(" | ".join(contact_bits))

    c1, c2, c3 = st.columns(3)
    c1.metric("Match Score", f"{match_score:.0f}")
    c2.metric("ATS Score", f"{ats_result['overall_score']:.0f}")
    c3.metric("Final Score", f"{final_score:.0f}")

    tab_overview, tab_skills, tab_suggestions = st.tabs(
        ["\U0001F4CA ATS Breakdown", "\U0001F9E9 Skill Gap", "\U0001F4A1 Suggestions"]
    )

    with tab_overview:
        st.plotly_chart(
            dashboard.build_subscore_breakdown_chart(ats_result["sub_scores"]),
            use_container_width=True,
        )

    with tab_skills:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                dashboard.build_skill_category_chart(gap_result["category_breakdown"]),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                dashboard.build_missing_skills_chart(gap_result["missing_skills"]),
                use_container_width=True,
            )
        st.metric("Skill Coverage", f"{gap_result['coverage_percentage']:.0f}%")

    with tab_suggestions:
        st.plotly_chart(
            dashboard.build_suggestion_priority_chart(improvement_result["suggestions"]),
            use_container_width=True,
        )
        for s in improvement_result["suggestions"]:
            with st.expander(f"[{s['priority'].upper()}] {s['category']} \u2014 {s['issue']}"):
                st.markdown(f"**Suggestion:** {s['suggestion']}")
                st.markdown(f"**Why it matters:** {s['why']}")

    st.divider()
    if st.button("\U0001F4C4 Generate PDF Report", key=f"pdf_{resume_data.get('name')}_{id(result)}"):
        with st.spinner("Building PDF report..."):
            pdf_path = report_generator.generate_report(
                resume_data, ats_result, gap_result, improvement_result,
                match_score, jd_title=jd_title,
            )
        with open(pdf_path, "rb") as f:
            st.download_button(
                "\u2b07\ufe0f Download Report", f, file_name=os.path.basename(pdf_path),
                mime="application/pdf", key=f"dl_{pdf_path}",
            )


# ------------------------------------------------------------------
# SIDEBAR -- mode + job description input (shared across both modes)
# ------------------------------------------------------------------
st.title("\U0001F4C4 AI-Powered Resume Screening System")
st.caption("Parse, score, and improve resumes against a job description -- powered by NLP + semantic matching.")

with st.sidebar:
    st.header("Setup")
    mode = st.radio("Mode", ["Single Candidate", "Multi-Candidate Ranking"])

    st.subheader("Job Description")
    jd_input_mode = st.radio("JD input method", ["Paste text", "Upload file"], horizontal=True)
    jd_pasted_text, jd_file = "", None
    if jd_input_mode == "Paste text":
        jd_pasted_text = st.text_area("Paste the job description", height=200)
    else:
        jd_file = st.file_uploader("Upload JD (PDF or TXT)", type=["pdf", "txt"])

# ------------------------------------------------------------------
# MODE 1: SINGLE CANDIDATE
# ------------------------------------------------------------------
if mode == "Single Candidate":
    resume_file = st.file_uploader("Upload a resume (PDF)", type=["pdf"])
    run = st.button("Screen Resume", type="primary")

    if run:
        if resume_file is None:
            st.error("Please upload a resume PDF.")
        else:
            try:
                jd_text = get_jd_text(jd_file, jd_pasted_text)
                if not jd_text:
                    st.error("Please paste or upload a job description.")
                else:
                    resume_path = save_upload(resume_file, ALLOWED_RESUME_EXTENSIONS)
                    with st.spinner("Parsing resume and running analysis..."):
                        result = run_pipeline(resume_path, jd_text)
                    st.session_state["single_result"] = result
                    st.session_state["single_jd_title"] = "Screened Role"
            except (FileValidationError, PDFExtractionError) as exc:
                st.error(str(exc))

    if "single_result" in st.session_state:
        render_single_candidate_dashboard(
            st.session_state["single_result"], st.session_state.get("single_jd_title", "")
        )

# ------------------------------------------------------------------
# MODE 2: MULTI-CANDIDATE RANKING
# ------------------------------------------------------------------
else:
    resume_files = st.file_uploader(
        "Upload multiple resumes (PDF)", type=["pdf"], accept_multiple_files=True
    )
    run = st.button("Screen All Candidates", type="primary")

    if run:
        if not resume_files:
            st.error("Please upload at least one resume PDF.")
        else:
            try:
                jd_text = get_jd_text(jd_file, jd_pasted_text)
                if not jd_text:
                    st.error("Please paste or upload a job description.")
                else:
                    results = []
                    progress = st.progress(0.0, text="Screening candidates...")
                    for i, rf in enumerate(resume_files):
                        try:
                            resume_path = save_upload(rf, ALLOWED_RESUME_EXTENSIONS)
                            results.append(run_pipeline(resume_path, jd_text))
                        except (FileValidationError, PDFExtractionError) as exc:
                            st.warning(f"Skipped '{rf.name}': {exc}")
                        progress.progress((i + 1) / len(resume_files), text=f"Screened {i + 1}/{len(resume_files)}")
                    progress.empty()
                    st.session_state["multi_results"] = results
            except (FileValidationError, PDFExtractionError) as exc:
                st.error(str(exc))

    if "multi_results" in st.session_state and st.session_state["multi_results"]:
        results = st.session_state["multi_results"]
        candidates = [
            {
                "name": r["resume_data"].get("name") or f"Candidate {i + 1}",
                "match_score": r["match_score"],
                "ats_score": r["ats_result"]["overall_score"],
            }
            for i, r in enumerate(results)
        ]
        ranked = dashboard.rank_candidates(candidates)

        st.plotly_chart(dashboard.build_candidate_ranking_chart(ranked), use_container_width=True)
        st.dataframe(ranked, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Candidate Detail")
        names = [r["resume_data"].get("name") or f"Candidate {i + 1}" for i, r in enumerate(results)]
        selected_name = st.selectbox("Select a candidate for the full dashboard + report", names)
        selected_result = results[names.index(selected_name)]
        render_single_candidate_dashboard(selected_result, jd_title="Screened Role")
