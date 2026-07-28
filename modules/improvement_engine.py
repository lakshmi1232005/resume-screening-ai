"""
modules/improvement_engine.py
================================
Generates specific, prioritized, explainable suggestions for improving
a resume against a target job description.

Why this module exists (and why it's rule-based, not a black box):
    Phases 5 and 6 tell a candidate WHAT is wrong ("your ATS score is
    62", "you're missing Docker"). This module tells them WHAT TO DO
    about it and WHY it matters -- each suggestion is grounded in a
    concrete, inspectable rule (a missing keyword, a weak phrase, a
    missing metric), never a vague "improve your resume" platitude.
    That also means every suggestion can be traced back to the exact
    data point that triggered it, which is what makes it trustworthy
    and useful rather than generic filler.

Design approach:
    1. Reuse Phase 5 (ATSScorer) and Phase 6 (SkillGapAnalyzer) as
       building blocks rather than re-deriving their logic -- single
       source of truth for scoring/gap detection.
    2. Add NEW rule-based checks this module owns directly: weak
       passive phrasing (config.WEAK_PHRASES), missing quantification
       (no numbers/metrics in bullet entries), and structural gaps
       (missing contact info, missing sections).
    3. Every suggestion carries: priority, category, issue, suggestion,
       and why -- so the dashboard (Phase 8) and PDF report (Phase 9)
       can render them directly without extra formatting logic.
"""

import re

from config import STRONG_ACTION_VERBS, WEAK_PHRASES
from modules.ats_scorer import ATSScorer
from modules.skill_gap_analyzer import SkillGapAnalyzer

# Priority ordering used to sort the final suggestion list so the
# highest-impact fixes surface first.
PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}

# How many individual missing-skill suggestions to emit before
# collapsing the rest into a single grouped suggestion. Keeps the
# output scannable instead of a wall of near-duplicate bullets.
MAX_INDIVIDUAL_SKILL_SUGGESTIONS = 5

# ATS sub-score thresholds below which we consider a section "weak"
# enough to generate a suggestion for it.
WEAK_SCORE_THRESHOLD = 60.0


class ImprovementEngine:
    """
    Generates a prioritized list of improvement suggestions for a
    resume, given the parsed resume data and a target job description.

    Usage:
        engine = ImprovementEngine()
        result = engine.generate_suggestions(resume_data, jd_text)
        for s in result["suggestions"]:
            print(s["priority"], s["category"], s["suggestion"], "--", s["why"])
    """

    def __init__(self):
        self.ats_scorer = ATSScorer()
        self.skill_gap_analyzer = SkillGapAnalyzer()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def generate_suggestions(self, resume_data: dict, jd_text: str) -> dict:
        """
        Run the full improvement analysis.

        Args:
            resume_data: The dict returned by ResumeParser.parse().
            jd_text: Raw job description text.

        Returns:
            dict: {
                "suggestions": [
                    {"priority": "high"|"medium"|"low", "category": str,
                     "issue": str, "suggestion": str, "why": str},
                    ...
                ],
                "high_priority_count": int,
                "ats_score": float,      # for context, re-uses Phase 5
                "skill_coverage": float, # for context, re-uses Phase 6
            }
        """
        ats_result = self.ats_scorer.compute_ats_score(resume_data, jd_text)
        gap_result = self.skill_gap_analyzer.analyze(resume_data.get("skills", []), jd_text)

        suggestions = []
        suggestions.extend(self._contact_and_structure_suggestions(resume_data, ats_result))
        suggestions.extend(self._skill_gap_suggestions(gap_result))
        suggestions.extend(self._weak_language_suggestions(resume_data))
        suggestions.extend(self._quantification_suggestions(resume_data))
        suggestions.extend(self._section_score_suggestions(resume_data, ats_result))

        suggestions.sort(key=lambda s: PRIORITY_RANK[s["priority"]])
        high_priority_count = sum(1 for s in suggestions if s["priority"] == "high")

        return {
            "suggestions": suggestions,
            "high_priority_count": high_priority_count,
            "ats_score": ats_result["overall_score"],
            "skill_coverage": gap_result["coverage_percentage"],
        }

    # ------------------------------------------------------------------
    # CONTACT INFO & STRUCTURAL COMPLETENESS
    # ------------------------------------------------------------------
    def _contact_and_structure_suggestions(self, resume_data: dict, ats_result: dict) -> list:
        """
        Flags missing contact info -- these are near-zero-effort fixes
        with outsized impact, since a missing email/phone can get a
        resume silently discarded by a human recruiter even if the ATS
        parses everything else perfectly.
        """
        suggestions = []

        if not resume_data.get("email"):
            suggestions.append({
                "priority": "high",
                "category": "Contact Info",
                "issue": "No email address detected on the resume.",
                "suggestion": "Add a professional email address near the top of the resume.",
                "why": "Recruiters and ATS systems both rely on a visible email address as the "
                       "primary contact method -- without one, a strong resume can go straight "
                       "into a dead-end pile.",
            })

        if not resume_data.get("phone"):
            suggestions.append({
                "priority": "medium",
                "category": "Contact Info",
                "issue": "No phone number detected on the resume.",
                "suggestion": "Add a phone number to the header, alongside your email.",
                "why": "Some recruiters prefer a quick call before scheduling a formal interview; "
                       "a missing number removes that option entirely.",
            })

        if not resume_data.get("linkedin") and not resume_data.get("github"):
            suggestions.append({
                "priority": "low",
                "category": "Contact Info",
                "issue": "No LinkedIn or GitHub link detected.",
                "suggestion": "Add a LinkedIn profile (and a GitHub link if you have public "
                               "code/projects) to your header.",
                "why": "These links give recruiters an easy way to verify your background and "
                       "see additional work beyond what fits on one page.",
            })

        return suggestions

    # ------------------------------------------------------------------
    # SKILL GAP -> SUGGESTIONS (built on Phase 6)
    # ------------------------------------------------------------------
    def _skill_gap_suggestions(self, gap_result: dict) -> list:
        """
        Converts Phase 6's missing-skill list into actionable
        suggestions, prioritized by the criticality Phase 6 already
        assigned. When a related skill is on the resume, the
        suggestion nudges toward REFRAMING existing experience rather
        than implying the candidate must learn something from scratch.
        """
        suggestions = []
        missing = gap_result["missing_skills"]

        criticality_to_priority = {"critical": "high", "important": "medium", "preferred": "low"}

        for entry in missing[:MAX_INDIVIDUAL_SKILL_SUGGESTIONS]:
            skill = entry["skill"]
            priority = criticality_to_priority[entry["criticality"]]
            related = entry.get("related_skills_you_have", [])

            if related:
                suggestion_text = (
                    f"Highlight your experience with {', '.join(r.title() for r in related)} more "
                    f"prominently, and consider whether it's fair to also list \"{skill.title()}\" "
                    f"if you've genuinely used it -- these are closely related skills."
                )
                why = (
                    f"The job description mentions \"{skill}\", which your resume doesn't "
                    f"explicitly list, but you already show related experience with "
                    f"{', '.join(r.title() for r in related)}. Making that connection explicit "
                    f"helps both a human reader and an ATS keyword scan."
                )
            else:
                suggestion_text = f"Consider adding \"{skill.title()}\" to your Skills section if you have genuine experience with it."
                article = "an" if entry["criticality"][0] in "aeiou" else "a"
                why = (
                    f"The job description references \"{skill}\" as {article} "
                    f"{entry['criticality']} skill, and it doesn't currently appear anywhere on "
                    f"your resume."
                )

            suggestions.append({
                "priority": priority,
                "category": "Skills",
                "issue": f"Missing {entry['criticality']} skill: {skill}",
                "suggestion": suggestion_text,
                "why": why,
            })

        remaining = missing[MAX_INDIVIDUAL_SKILL_SUGGESTIONS:]
        if remaining:
            remaining_names = ", ".join(e["skill"].title() for e in remaining)
            suggestions.append({
                "priority": "low",
                "category": "Skills",
                "issue": f"{len(remaining)} additional skill(s) mentioned in the JD are not on your resume.",
                "suggestion": f"Review whether any of these apply to you: {remaining_names}.",
                "why": "These appeared less prominently in the job description, but each one "
                       "you can genuinely claim adds another keyword match.",
            })

        return suggestions

    # ------------------------------------------------------------------
    # WEAK / PASSIVE LANGUAGE DETECTION
    # ------------------------------------------------------------------
    def _weak_language_suggestions(self, resume_data: dict) -> list:
        """
        Scans Experience and Projects bullet entries for passive,
        low-impact phrasing (config.WEAK_PHRASES, e.g. "responsible
        for", "helped with") and suggests swapping in a strong action
        verb instead (config.STRONG_ACTION_VERBS).
        """
        suggestions = []
        sample_verbs = ", ".join(STRONG_ACTION_VERBS[:6])

        for section_name in ("experience", "projects"):
            entries = resume_data.get(section_name, [])
            flagged_entries = []
            for entry in entries:
                entry_lower = entry.lower()
                matched_phrases = [phrase for phrase in WEAK_PHRASES if phrase in entry_lower]
                if matched_phrases:
                    flagged_entries.append((entry, matched_phrases))

            if not flagged_entries:
                continue

            example_entry, example_phrases = flagged_entries[0]
            suggestions.append({
                "priority": "medium",
                "category": section_name.title(),
                "issue": f"{len(flagged_entries)} bullet(s) in {section_name.title()} use weak, "
                         f"passive phrasing (e.g. \"{example_phrases[0]}\").",
                "suggestion": f"Replace phrases like \"{example_phrases[0]}\" with a strong action "
                              f"verb at the start of the bullet -- e.g. {sample_verbs}.",
                "why": "Passive phrasing (\"was responsible for\", \"helped with\") describes your "
                       "presence in a task, not your impact on it. Leading with a strong verb "
                       "makes each bullet read as an accomplishment rather than a duty.",
            })

        return suggestions

    # ------------------------------------------------------------------
    # QUANTIFICATION / METRICS CHECK
    # ------------------------------------------------------------------
    def _quantification_suggestions(self, resume_data: dict) -> list:
        """
        Flags Experience/Projects entries that contain no numbers at
        all -- no percentages, counts, durations, or scale indicators.
        Quantified impact ("reduced load time by 40%") is consistently
        one of the highest-signal things a resume can show, and it's
        also one of the most commonly skipped.
        """
        suggestions = []

        for section_name in ("experience", "projects"):
            entries = resume_data.get(section_name, [])
            if not entries:
                continue

            unquantified = [e for e in entries if not re.search(r"\d", e)]
            if not unquantified:
                continue

            suggestions.append({
                "priority": "medium",
                "category": section_name.title(),
                "issue": f"{len(unquantified)} of {len(entries)} bullet(s) in {section_name.title()} "
                         f"have no numbers or metrics.",
                "suggestion": "Add a concrete number where possible -- team size, percentage "
                              "improvement, users affected, time saved, or scale (e.g. \"improved "
                              "query performance by 35%\" instead of \"improved query performance\").",
                "why": "Quantified bullets are far easier for both a recruiter and an ATS to judge "
                       "at a glance, and they make a claim of impact concrete instead of vague.",
            })

        return suggestions

    # ------------------------------------------------------------------
    # ATS SUB-SCORE -> SECTION-LEVEL SUGGESTIONS
    # ------------------------------------------------------------------
    def _section_score_suggestions(self, resume_data: dict, ats_result: dict) -> list:
        """
        Turns any weak ATS sub-score (below WEAK_SCORE_THRESHOLD) that
        isn't already covered by the more specific checks above
        (skills, weak language, quantification) into a suggestion --
        currently: education and certifications, the two purely
        presence-based sub-scores.
        """
        suggestions = []
        sub_scores = ats_result["sub_scores"]

        if sub_scores.get("education", 100) < WEAK_SCORE_THRESHOLD and not resume_data.get("education"):
            suggestions.append({
                "priority": "high",
                "category": "Education",
                "issue": "No Education section was detected on the resume.",
                "suggestion": "Add an Education section with your degree, institution, and "
                               "graduation year (or expected date).",
                "why": "Most ATS systems and recruiters treat a missing Education section as a "
                       "red flag or an incomplete resume, even for experienced candidates.",
            })

        if sub_scores.get("certifications", 100) < WEAK_SCORE_THRESHOLD and not resume_data.get("certifications"):
            suggestions.append({
                "priority": "low",
                "category": "Certifications",
                "issue": "No certifications listed.",
                "suggestion": "If you hold any relevant certifications (cloud platforms, "
                               "specific tools, or coursework certificates), add a "
                               "Certifications section listing them.",
                "why": "Certifications are a small bonus signal for ATS scoring, and can help "
                       "differentiate you when a JD lists specific tools or platforms you've "
                       "formally trained on.",
            })

        return suggestions
