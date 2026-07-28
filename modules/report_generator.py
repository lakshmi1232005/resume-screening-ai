"""
modules/report_generator.py
==============================
Builds a downloadable PDF report from the outputs of Phases 2-7
(parsed resume, ATS score, skill gap analysis, improvement
suggestions) plus the Phase 4 semantic match score.

Why reportlab + platypus (not the raw canvas API):
    The report is fundamentally a flowing document -- tables, headings,
    and wrapped paragraphs that need to break cleanly across pages
    without the generator manually tracking a "current y position".
    Platypus (SimpleDocTemplate + a story list of Flowables) handles
    pagination, page breaks, and text wrapping for us, the same way
    dashboard.py leans on Plotly instead of hand-drawing charts.

Design note on color coding:
    This module deliberately imports its color language from
    dashboard.py (COLOR_GOOD/OK/POOR/NEUTRAL, the score thresholds,
    and the criticality/priority color maps) instead of redefining
    its own. A candidate who looks at the on-screen dashboard and then
    downloads the PDF should see "red = needs attention" mean exactly
    the same thing in both places -- one color language, two
    renderings (Plotly figures on screen, reportlab tables on paper).

Output location:
    Reports are saved under config.REPORTS_DIR by default, matching
    where the rest of the app already expects generated PDFs to live
    (see config.py's folder-path section).
"""

import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import ATS_WEIGHTS, REPORTS_DIR
from modules.dashboard import (
    COLOR_GOOD,
    COLOR_NEUTRAL,
    COLOR_OK,
    COLOR_POOR,
    CRITICALITY_COLORS,
    PRIORITY_COLORS,
    _score_color,
)

# reportlab wants colors.Color / colors.HexColor objects, not raw hex
# strings -- wrap the shared palette once here rather than sprinkling
# HexColor(...) calls throughout the flowable-building code below.
_HEX = {
    COLOR_GOOD: colors.HexColor(COLOR_GOOD),
    COLOR_OK: colors.HexColor(COLOR_OK),
    COLOR_POOR: colors.HexColor(COLOR_POOR),
    COLOR_NEUTRAL: colors.HexColor(COLOR_NEUTRAL),
}


def _rl_color(hex_str: str) -> colors.Color:
    """Looks up (or lazily creates) the reportlab Color for a hex string."""
    if hex_str not in _HEX:
        _HEX[hex_str] = colors.HexColor(hex_str)
    return _HEX[hex_str]


# Light tint backgrounds for table rows/cells, mirroring the pale
# green/amber/red step colors used behind the gauges in dashboard.py
# (build_score_gauge's "steps"), so a screenshot of the gauge and a
# printed table row feel like the same design system.
_TINT = {
    COLOR_GOOD: colors.HexColor("#eafaf1"),
    COLOR_OK: colors.HexColor("#fff8e1"),
    COLOR_POOR: colors.HexColor("#fdecea"),
    COLOR_NEUTRAL: colors.HexColor("#f4f5f5"),
}

PAGE_MARGIN = 0.65 * inch


class ReportGenerator:
    """
    Assembles a multi-section PDF report for a single candidate/JD pair.

    Stateless by design, same as DashboardBuilder: every method takes
    the relevant Phase 2/4/5/6/7 result dict(s) directly and returns
    either a Flowable (or list of Flowables) or, for the public entry
    point, a path to the saved PDF file on disk.

    Usage:
        generator = ReportGenerator()
        pdf_path = generator.generate_report(
            resume_data=resume_data,
            ats_result=ats_result,
            gap_result=gap_result,
            improvement_result=improvement_result,
            match_score=match_score,
            jd_title="Senior Backend Engineer",
        )
    """

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._register_custom_styles()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def generate_report(
        self,
        resume_data: dict,
        ats_result: dict,
        gap_result: dict,
        improvement_result: dict,
        match_score: float,
        jd_title: str = "",
        output_path: str = None,
    ) -> str:
        """
        Build the full PDF report and save it to disk.

        Args:
            resume_data: Output of ResumeParser.parse() (Phase 2).
            ats_result: Output of ATSScorer.score() (Phase 5).
            gap_result: Output of SkillGapAnalyzer.analyze() (Phase 6).
            improvement_result: Output of ImprovementEngine.generate_suggestions()
                                 (Phase 7).
            match_score: Semantic match percentage from SemanticMatcher
                          (Phase 4), 0-100.
            jd_title: Optional job title/label shown in the report header.
            output_path: Optional explicit save path. If omitted, a
                          filename is generated from the candidate name
                          and timestamp and saved under REPORTS_DIR.

        Returns:
            str: Absolute path to the saved PDF file.
        """
        final_score = self._compute_final_score(match_score, ats_result["overall_score"])

        if output_path is None:
            output_path = self._default_output_path(resume_data.get("name"))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            leftMargin=PAGE_MARGIN, rightMargin=PAGE_MARGIN,
            topMargin=PAGE_MARGIN, bottomMargin=PAGE_MARGIN,
            title="Resume Screening Report",
        )

        story = []
        story += self._build_header(resume_data, jd_title)
        story += self._build_score_summary(match_score, ats_result["overall_score"], final_score)
        story += self._build_ats_breakdown(ats_result["sub_scores"])
        story.append(PageBreak())
        story += self._build_skill_gap_section(gap_result)
        story.append(PageBreak())
        story += self._build_suggestions_section(improvement_result)
        story += self._build_footer()

        doc.build(story)
        return output_path

    # ------------------------------------------------------------------
    # STYLES
    # ------------------------------------------------------------------
    def _register_custom_styles(self):
        """Adds report-specific paragraph styles on top of the sample stylesheet."""
        self.styles.add(ParagraphStyle(
            name="ReportTitle", parent=self.styles["Title"],
            fontSize=20, spaceAfter=4, textColor=colors.HexColor("#2c3e50"),
        ))
        self.styles.add(ParagraphStyle(
            name="ReportSubtitle", parent=self.styles["Normal"],
            fontSize=10, textColor=colors.HexColor("#7f8c8d"), spaceAfter=2,
        ))
        self.styles.add(ParagraphStyle(
            name="SectionHeading", parent=self.styles["Heading2"],
            fontSize=14, spaceBefore=14, spaceAfter=8,
            textColor=colors.HexColor("#2c3e50"),
        ))
        self.styles.add(ParagraphStyle(
            name="BigScoreLabel", parent=self.styles["Normal"],
            fontSize=10, alignment=TA_CENTER, textColor=colors.HexColor("#555555"),
        ))
        self.styles.add(ParagraphStyle(
            name="BigScoreValue", parent=self.styles["Normal"],
            fontSize=26, alignment=TA_CENTER, leading=30,
        ))
        self.styles.add(ParagraphStyle(
            name="CellWrap", parent=self.styles["Normal"], fontSize=9, leading=12,
        ))
        self.styles.add(ParagraphStyle(
            name="CellWrapBold", parent=self.styles["Normal"],
            fontSize=9, leading=12, fontName="Helvetica-Bold",
        ))

    # ------------------------------------------------------------------
    # 1. HEADER
    # ------------------------------------------------------------------
    def _build_header(self, resume_data: dict, jd_title: str) -> list:
        name = resume_data.get("name") or "Candidate"
        generated_on = datetime.now().strftime("%B %d, %Y at %I:%M %p")

        flow = [
            Paragraph("AI Resume Screening Report", self.styles["ReportTitle"]),
            Paragraph(f"Candidate: <b>{name}</b>", self.styles["ReportSubtitle"]),
        ]
        if jd_title:
            flow.append(Paragraph(f"Role: <b>{jd_title}</b>", self.styles["ReportSubtitle"]))
        contact_bits = [b for b in (resume_data.get("email"), resume_data.get("phone")) if b]
        if contact_bits:
            flow.append(Paragraph(" &nbsp;|&nbsp; ".join(contact_bits), self.styles["ReportSubtitle"]))
        flow.append(Paragraph(f"Generated on {generated_on}", self.styles["ReportSubtitle"]))
        flow.append(Spacer(1, 6))
        flow.append(HRFlowable(width="100%", color=colors.HexColor("#dddddd"), thickness=1))
        flow.append(Spacer(1, 10))
        return flow

    # ------------------------------------------------------------------
    # 2. TOP-LINE SCORE SUMMARY (Match / ATS / Final)
    # ------------------------------------------------------------------
    def _build_score_summary(self, match_score: float, ats_score: float, final_score: float) -> list:
        cards = [
            self._score_card("Match Score", match_score),
            self._score_card("ATS Score", ats_score),
            self._score_card("Final Score", final_score),
        ]
        table = Table([cards], colWidths=[(letter[0] - 2 * PAGE_MARGIN) / 3] * 3)
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX", (0, 0), (0, 0), 0.5, colors.HexColor("#e0e0e0")),
            ("BOX", (1, 0), (1, 0), 0.5, colors.HexColor("#e0e0e0")),
            ("BOX", (2, 0), (2, 0), 0.5, colors.HexColor("#e0e0e0")),
            ("BACKGROUND", (0, 0), (0, 0), _TINT[_score_color(match_score)]),
            ("BACKGROUND", (1, 0), (1, 0), _TINT[_score_color(ats_score)]),
            ("BACKGROUND", (2, 0), (2, 0), _TINT[_score_color(final_score)]),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ]))
        return [table, Spacer(1, 4)]

    def _score_card(self, label: str, score: float) -> Table:
        """A single label-over-number 'card', colored by the shared score bands."""
        color = _rl_color(_score_color(score))
        value_style = ParagraphStyle(
            name="BigScoreValueColored", parent=self.styles["BigScoreValue"], textColor=color,
        )
        inner = Table(
            [[Paragraph(label, self.styles["BigScoreLabel"])],
             [Paragraph(f"{score:.0f}", value_style)]],
            colWidths=[(letter[0] - 2 * PAGE_MARGIN) / 3 - 12],
        )
        inner.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        return inner

    # ------------------------------------------------------------------
    # 3. ATS SUB-SCORE BREAKDOWN (Phase 5)
    # ------------------------------------------------------------------
    def _build_ats_breakdown(self, sub_scores: dict) -> list:
        flow = [Paragraph("ATS Sub-Score Breakdown", self.styles["SectionHeading"])]

        # Same ordering rule as DashboardBuilder.build_subscore_breakdown_chart:
        # heaviest-weighted sub-score first.
        ordered_keys = sorted(sub_scores.keys(), key=lambda k: ATS_WEIGHTS.get(k, 0), reverse=True)

        rows = [["Sub-Score", "Weight", "Score", ""]]
        row_colors = []
        for key in ordered_keys:
            value = sub_scores[key]
            label = key.replace("_", " ").title()
            weight_pct = f"{ATS_WEIGHTS.get(key, 0) * 100:.0f}%"
            bar = self._mini_bar(value)
            rows.append([label, weight_pct, f"{value:.0f}", bar])
            row_colors.append(_score_color(value))

        table = Table(rows, colWidths=[1.8 * inch, 0.8 * inch, 0.7 * inch, 2.6 * inch], hAlign="LEFT")
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
        ]
        for i, color_hex in enumerate(row_colors, start=1):
            style.append(("TEXTCOLOR", (2, i), (2, i), _rl_color(color_hex)))
            style.append(("FONTNAME", (2, i), (2, i), "Helvetica-Bold"))
        table.setStyle(TableStyle(style))
        flow.append(table)
        return flow

    def _mini_bar(self, value: float) -> Table:
        """A tiny horizontal bar (filled/empty cell pair) as an inline 0-100 indicator."""
        width_total = 2.4 * inch
        filled = max(0.02, min(1.0, value / 100.0)) * width_total
        empty = width_total - filled
        bar = Table([["", ""]], colWidths=[filled, empty], rowHeights=[8])
        bar.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), _rl_color(_score_color(value))),
            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#f0f0f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        return bar

    # ------------------------------------------------------------------
    # 4. SKILL GAP SECTION (Phase 6)
    # ------------------------------------------------------------------
    def _build_skill_gap_section(self, gap_result: dict) -> list:
        flow = [Paragraph("Skill Gap Analysis", self.styles["SectionHeading"])]

        coverage = gap_result["coverage_percentage"]
        summary = (
            f"Matched <b>{len(gap_result['matched_skills'])}</b> of "
            f"<b>{gap_result['total_required_skills']}</b> JD skills "
            f"(<font color='{_score_color(coverage)}'><b>{coverage:.0f}% coverage</b></font>), "
            f"including <b>{gap_result['critical_gap_count']}</b> critical gap(s)."
        )
        flow.append(Paragraph(summary, self.styles["Normal"]))
        flow.append(Spacer(1, 10))

        # -- Category breakdown table --
        category_breakdown = gap_result.get("category_breakdown", {})
        if category_breakdown:
            flow.append(Paragraph("Coverage by Category", self.styles["Heading3"]))
            rows = [["Category", "Matched / Required", "Coverage"]]
            style = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
            for i, (category, stats) in enumerate(category_breakdown.items(), start=1):
                rows.append([
                    category,
                    f"{stats['matched']} / {stats['required']}",
                    f"{stats['coverage']:.0f}%",
                ])
                style.append(("TEXTCOLOR", (2, i), (2, i), _rl_color(_score_color(stats["coverage"]))))
                style.append(("FONTNAME", (2, i), (2, i), "Helvetica-Bold"))
            table = Table(rows, colWidths=[2.6 * inch, 1.8 * inch, 1.7 * inch], hAlign="LEFT")
            table.setStyle(TableStyle(style))
            flow.append(table)
            flow.append(Spacer(1, 12))

        # -- Missing skills, colored by criticality (same language as
        #    dashboard.py's build_missing_skills_chart) --
        missing = gap_result.get("missing_skills", [])
        flow.append(Paragraph("Missing Skills", self.styles["Heading3"]))
        if not missing:
            flow.append(Paragraph("No missing skills detected -- full coverage!", self.styles["Normal"]))
        else:
            rows = [["Skill", "Category", "Criticality", "Related Skills You Have"]]
            style = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
            for i, entry in enumerate(missing, start=1):
                related = ", ".join(entry.get("related_skills_you_have") or []) or "-"
                rows.append([
                    Paragraph(entry["skill"].title(), self.styles["CellWrap"]),
                    Paragraph(entry["category"], self.styles["CellWrap"]),
                    Paragraph(entry["criticality"].title(), self.styles["CellWrapBold"]),
                    Paragraph(related, self.styles["CellWrap"]),
                ])
                crit_color = CRITICALITY_COLORS.get(entry["criticality"], COLOR_NEUTRAL)
                style.append(("BACKGROUND", (2, i), (2, i), _TINT[crit_color]))
                style.append(("TEXTCOLOR", (2, i), (2, i), _rl_color(crit_color)))
            table = Table(rows, colWidths=[1.2 * inch, 1.3 * inch, 0.9 * inch, 2.7 * inch], hAlign="LEFT")
            table.setStyle(TableStyle(style))
            flow.append(table)

        return flow

    # ------------------------------------------------------------------
    # 5. IMPROVEMENT SUGGESTIONS (Phase 7)
    # ------------------------------------------------------------------
    def _build_suggestions_section(self, improvement_result: dict) -> list:
        flow = [Paragraph("Improvement Suggestions", self.styles["SectionHeading"])]

        suggestions = improvement_result.get("suggestions", [])
        if not suggestions:
            flow.append(Paragraph("No suggestions -- this resume looks strong!", self.styles["Normal"]))
            return flow

        high_count = improvement_result.get("high_priority_count", 0)
        flow.append(Paragraph(
            f"<b>{len(suggestions)}</b> suggestion(s) found, including "
            f"<font color='{PRIORITY_COLORS['high']}'><b>{high_count} high-priority</b></font> item(s).",
            self.styles["Normal"],
        ))
        flow.append(Spacer(1, 8))

        # Same urgency ordering the priority donut in dashboard.py implies:
        # high -> medium -> low.
        priority_rank = {"high": 0, "medium": 1, "low": 2}
        ordered = sorted(suggestions, key=lambda s: priority_rank.get(s["priority"], 3))

        for s in ordered:
            flow.append(KeepTogether(self._suggestion_block(s)))
            flow.append(Spacer(1, 8))
        return flow

    def _suggestion_block(self, suggestion: dict) -> list:
        color_hex = PRIORITY_COLORS.get(suggestion["priority"], COLOR_NEUTRAL)
        header = Table(
            [[
                Paragraph(f"<b>{suggestion['priority'].upper()}</b>", self.styles["CellWrapBold"]),
                Paragraph(f"<b>{suggestion['category']}</b> &mdash; {suggestion['issue']}", self.styles["CellWrap"]),
            ]],
            colWidths=[0.7 * inch, 5.9 * inch],
        )
        header.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), _rl_color(color_hex)),
            ("TEXTCOLOR", (0, 0), (0, 0), colors.white),
            ("BACKGROUND", (1, 0), (1, 0), _TINT[color_hex]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (0, 0), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (1, 0), (1, 0), 8),
        ]))
        body = Paragraph(
            f"<b>Suggestion:</b> {suggestion['suggestion']}<br/><br/>"
            f"<b>Why it matters:</b> {suggestion['why']}",
            self.styles["CellWrap"],
        )
        wrapper = Table([[header], [body]], colWidths=[6.6 * inch])
        wrapper.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
            ("TOPPADDING", (0, 1), (-1, 1), 8), ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
            ("LEFTPADDING", (0, 1), (-1, 1), 10), ("RIGHTPADDING", (0, 1), (-1, 1), 10),
        ]))
        return [wrapper]

    # ------------------------------------------------------------------
    # 6. FOOTER
    # ------------------------------------------------------------------
    def _build_footer(self) -> list:
        return [
            Spacer(1, 16),
            HRFlowable(width="100%", color=colors.HexColor("#dddddd"), thickness=1),
            Spacer(1, 6),
            Paragraph(
                "Generated automatically by the AI-Powered Resume Screening System. "
                "Scores and suggestions are advisory -- use judgment alongside them.",
                self.styles["ReportSubtitle"],
            ),
        ]

    # ------------------------------------------------------------------
    # SHARED HELPERS
    # ------------------------------------------------------------------
    def _compute_final_score(self, match_score: float, ats_score: float) -> float:
        """
        Delegates to DashboardBuilder's blend so the PDF's "Final Score"
        always matches the number shown on screen -- one formula
        (config.FINAL_SCORE_WEIGHTS), used everywhere it's needed.
        """
        from modules.dashboard import DashboardBuilder
        return DashboardBuilder().compute_final_score(match_score, ats_score)

    def _default_output_path(self, candidate_name: str) -> str:
        safe_name = "".join(c if c.isalnum() else "_" for c in (candidate_name or "candidate")).strip("_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name or 'candidate'}_report_{timestamp}.pdf"
        return os.path.join(REPORTS_DIR, filename)
