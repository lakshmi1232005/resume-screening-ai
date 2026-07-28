"""
modules/dashboard.py
=======================
Builds interactive Plotly visualizations from the outputs of Phases
5-7 (ATS score, skill gaps, improvement suggestions), plus multi-
candidate ranking for screening several resumes against one JD.

Why this module returns Plotly Figures instead of calling Streamlit
directly:
    Keeping chart-building logic separate from the UI framework means
    it's testable on its own (no running Streamlit server needed) and
    reusable elsewhere -- e.g. Phase 9 (PDF report) can export these
    same figures as static images instead of re-implementing charts
    in a second place. Phase 10 (app.py) will simply do:
        st.plotly_chart(builder.build_score_gauge(...), use_container_width=True)

Design note on color coding:
    Every chart in this module uses the SAME green/amber/red logic for
    "good/ok/poor" (via `_score_color`) and the same criticality/
    priority color mapping. Consistent color language across every
    chart means a user learns "red = needs attention" once and it
    holds throughout the whole dashboard, instead of every chart
    inventing its own scale.
"""

import plotly.graph_objects as go

from config import ATS_WEIGHTS, FINAL_SCORE_WEIGHTS

# ----------------------------------------------------------------------
# SHARED COLOR LANGUAGE
# ----------------------------------------------------------------------
COLOR_GOOD = "#2ecc71"
COLOR_OK = "#f1c40f"
COLOR_POOR = "#e74c3c"
COLOR_NEUTRAL = "#95a5a6"

# Score bands: >= GOOD_THRESHOLD -> green, >= OK_THRESHOLD -> amber, else red.
GOOD_THRESHOLD = 75
OK_THRESHOLD = 50

CRITICALITY_COLORS = {"critical": COLOR_POOR, "important": COLOR_OK, "preferred": COLOR_NEUTRAL}
PRIORITY_COLORS = {"high": COLOR_POOR, "medium": COLOR_OK, "low": COLOR_NEUTRAL}


def _score_color(score: float) -> str:
    """Maps a 0-100 score to the shared green/amber/red color language."""
    if score >= GOOD_THRESHOLD:
        return COLOR_GOOD
    if score >= OK_THRESHOLD:
        return COLOR_OK
    return COLOR_POOR


class DashboardBuilder:
    """
    Builds Plotly figures for candidate score visualization.

    Stateless by design: every method takes the relevant Phase 5/6/7
    result dict(s) directly and returns a ready-to-render go.Figure.
    No file I/O, no Streamlit calls -- safe to unit test in isolation.

    Usage:
        builder = DashboardBuilder()
        gauge_fig = builder.build_score_gauge(ats_result["overall_score"], "ATS Score")
        subscore_fig = builder.build_subscore_breakdown_chart(ats_result["sub_scores"])
    """

    # ------------------------------------------------------------------
    # 1. SCORE GAUGE -- generic, reused for ATS score, match %, final score
    # ------------------------------------------------------------------
    def build_score_gauge(self, score: float, title: str) -> go.Figure:
        """A single 0-100 gauge, colored by the shared score bands."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": _score_color(score)},
                "steps": [
                    {"range": [0, OK_THRESHOLD], "color": "#fdecea"},
                    {"range": [OK_THRESHOLD, GOOD_THRESHOLD], "color": "#fff8e1"},
                    {"range": [GOOD_THRESHOLD, 100], "color": "#eafaf1"},
                ],
            },
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        return fig

    # ------------------------------------------------------------------
    # 2. ATS SUB-SCORE BREAKDOWN (Phase 5)
    # ------------------------------------------------------------------
    def build_subscore_breakdown_chart(self, sub_scores: dict) -> go.Figure:
        """
        Horizontal bar chart of the 7 ATS sub-scores, ordered by their
        weight in the overall score (heaviest-weighted first) so the
        reading order itself communicates "these matter most".
        """
        ordered_keys = sorted(sub_scores.keys(), key=lambda k: ATS_WEIGHTS.get(k, 0), reverse=True)
        labels = [k.replace("_", " ").title() for k in ordered_keys]
        values = [sub_scores[k] for k in ordered_keys]
        colors = [_score_color(v) for v in values]

        fig = go.Figure(go.Bar(
            x=values, y=labels, orientation="h",
            marker_color=colors,
            text=[f"{v:.0f}" for v in values], textposition="outside",
        ))
        fig.update_layout(
            title="ATS Sub-Score Breakdown",
            xaxis=dict(range=[0, 100], title="Score"),
            height=350, margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    # ------------------------------------------------------------------
    # 3. SKILL CATEGORY COVERAGE (Phase 6)
    # ------------------------------------------------------------------
    def build_skill_category_chart(self, category_breakdown: dict) -> go.Figure:
        """
        Horizontal bar chart of coverage % per skill category (e.g.
        "Cloud / DevOps: 33%"), with the underlying matched/required
        counts shown as bar labels for context beyond the percentage.
        """
        if not category_breakdown:
            return self._empty_state_figure("No JD skills detected to compare against.")

        categories = list(category_breakdown.keys())
        coverage = [category_breakdown[c]["coverage"] for c in categories]
        matched = [category_breakdown[c]["matched"] for c in categories]
        required = [category_breakdown[c]["required"] for c in categories]
        colors = [_score_color(v) for v in coverage]
        labels = [f"{m}/{r} skills" for m, r in zip(matched, required)]

        fig = go.Figure(go.Bar(
            x=coverage, y=categories, orientation="h",
            marker_color=colors, text=labels, textposition="outside",
        ))
        fig.update_layout(
            title="Skill Coverage by Category",
            xaxis=dict(range=[0, 100], title="Coverage %"),
            height=max(250, 60 * len(categories)), margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    # ------------------------------------------------------------------
    # 4. MISSING SKILLS, COLORED BY CRITICALITY (Phase 6)
    # ------------------------------------------------------------------
    def build_missing_skills_chart(self, missing_skills: list) -> go.Figure:
        """
        One bar per missing skill, colored by criticality (critical =
        red, important = amber, preferred = grey) so the most urgent
        gaps are visually obvious at a glance, not just first in a list.
        """
        if not missing_skills:
            return self._empty_state_figure("No missing skills detected -- full coverage!")

        # Reverse so the chart reads top-to-bottom in the same
        # critical -> preferred order the data already comes sorted in.
        ordered = list(reversed(missing_skills))
        skills = [s["skill"].title() for s in ordered]
        criticalities = [s["criticality"] for s in ordered]
        colors = [CRITICALITY_COLORS[c] for c in criticalities]

        fig = go.Figure(go.Bar(
            x=[1] * len(skills), y=skills, orientation="h",
            marker_color=colors,
            text=[c.title() for c in criticalities], textposition="inside",
        ))
        fig.update_layout(
            title="Missing Skills by Importance",
            xaxis=dict(visible=False),
            height=max(250, 40 * len(skills)), margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
        )
        return fig

    # ------------------------------------------------------------------
    # 5. IMPROVEMENT SUGGESTIONS BY PRIORITY (Phase 7)
    # ------------------------------------------------------------------
    def build_suggestion_priority_chart(self, suggestions: list) -> go.Figure:
        """Donut chart summarizing how many suggestions fall in each priority tier."""
        if not suggestions:
            return self._empty_state_figure("No suggestions -- this resume looks strong!")

        counts = {"high": 0, "medium": 0, "low": 0}
        for s in suggestions:
            counts[s["priority"]] += 1

        present = [(k, v) for k, v in counts.items() if v > 0]
        labels = [k.title() for k, v in present]
        values = [v for k, v in present]
        colors = [PRIORITY_COLORS[k] for k, v in present]

        fig = go.Figure(go.Pie(labels=labels, values=values, marker_colors=colors, hole=0.5))
        fig.update_layout(
            title="Improvement Suggestions by Priority",
            height=300, margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    # ------------------------------------------------------------------
    # 6. MULTI-CANDIDATE RANKING (multi-resume screening view)
    # ------------------------------------------------------------------
    def compute_final_score(self, match_score: float, ats_score: float) -> float:
        """
        Blends the semantic match % (Phase 4) and the ATS score
        (Phase 5) using config.FINAL_SCORE_WEIGHTS -- the single
        number recruiters use to rank candidates against each other.
        """
        weights = FINAL_SCORE_WEIGHTS
        blended = match_score * weights["match_score"] + ats_score * weights["ats_score"]
        return round(blended, 2)

    def rank_candidates(self, candidates: list) -> list:
        """
        Args:
            candidates: list of {"name": str, "match_score": float, "ats_score": float}

        Returns:
            list: same dicts, each augmented with "final_score", sorted
            descending (best candidate first). Ties broken by name for
            stable, reproducible ordering.
        """
        ranked = [{**c, "final_score": self.compute_final_score(c["match_score"], c["ats_score"])}
                  for c in candidates]
        ranked.sort(key=lambda c: (-c["final_score"], c.get("name") or ""))
        return ranked

    def build_candidate_ranking_chart(self, ranked_candidates: list) -> go.Figure:
        """
        Grouped bar chart comparing Match / ATS / Final score across
        multiple candidates -- the "leaderboard" view for screening
        many resumes against one job description at once.
        """
        if not ranked_candidates:
            return self._empty_state_figure("No candidates to compare yet.")

        names = [c.get("name") or f"Candidate {i + 1}" for i, c in enumerate(ranked_candidates)]
        fig = go.Figure()
        fig.add_bar(name="Match %", x=names, y=[c["match_score"] for c in ranked_candidates])
        fig.add_bar(name="ATS Score", x=names, y=[c["ats_score"] for c in ranked_candidates])
        fig.add_bar(name="Final Score", x=names, y=[c["final_score"] for c in ranked_candidates])
        fig.update_layout(
            title="Candidate Ranking", barmode="group",
            yaxis=dict(range=[0, 100], title="Score"),
            height=400, margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    # ------------------------------------------------------------------
    # SHARED HELPER
    # ------------------------------------------------------------------
    def _empty_state_figure(self, message: str) -> go.Figure:
        """
        A blank figure with a centered message, used whenever there's
        genuinely nothing to plot (e.g. zero missing skills, zero
        candidates yet) -- avoids rendering a confusing empty chart.
        """
        fig = go.Figure()
        fig.add_annotation(text=message, showarrow=False, font=dict(size=16))
        fig.update_layout(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            height=200, margin=dict(l=10, r=10, t=10, b=10),
        )
        return fig
