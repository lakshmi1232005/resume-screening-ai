"""
modules/skill_gap_analyzer.py
================================
Identifies which JD-required skills the resume is missing, how
important each gap is, and what the candidate could highlight or
learn to close it.

Why this is a separate module from ats_scorer.py:
    Phase 5's `_score_skills()` answers ONE question: "what percentage
    of JD skills does the resume cover?" -- a single number. That's
    enough for a score, but not enough to help a candidate DO
    anything about it. This module answers three more useful
    questions instead:
        1. WHICH specific skills are missing (not just how many)?
        2. HOW important is each gap -- is this a dealbreaker
           ("required", "must have") or a bonus ("nice to have")?
        3. Does the candidate already have something RELATED that
           they could lean on or reframe (e.g. missing "tensorflow"
           but the resume already lists "pytorch")?

    Phase 7 (Improvement Engine) will consume this module's output
    directly to generate specific suggestions like "Consider adding
    Docker -- it's a required skill for this role and you don't
    mention it, though your Kubernetes experience is a related plus."
"""

import re

from config import CRITICAL_INDICATORS, PREFERRED_INDICATORS, SKILL_CATEGORIES, SKILL_DATABASE

# Ordering used to sort missing skills so the most urgent gaps surface
# first (critical > important > preferred).
CRITICALITY_RANK = {"critical": 0, "important": 1, "preferred": 2}

# How many "related skills you already have" to surface per gap.
# Kept small -- this is meant to be a quick, scannable hint, not an
# exhaustive list.
MAX_RELATED_SKILLS = 3


class SkillGapAnalyzer:
    """
    Compares a resume's extracted skills against a job description to
    find missing skills, rank them by importance, and surface related
    skills the candidate already has.

    Usage:
        analyzer = SkillGapAnalyzer()
        result = analyzer.analyze(resume_data["skills"], jd_text)
        print(result["coverage_percentage"], result["missing_skills"])
    """

    def __init__(self):
        # Reverse lookup: skill -> category, built once so we don't
        # re-scan SKILL_CATEGORIES for every missing skill.
        self._skill_to_category = {
            skill: category
            for category, skills in SKILL_CATEGORIES.items()
            for skill in skills
        }

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def analyze(self, resume_skills: list, jd_text: str) -> dict:
        """
        Run the full skill gap analysis.

        Args:
            resume_skills: List of skills extracted from the resume
                            (as returned by ResumeParser -- see
                            resume_data["skills"]).
            jd_text: Raw job description text.

        Returns:
            dict: {
                "total_required_skills": int,
                "matched_skills": [{"skill", "criticality", "category"}],
                "missing_skills": [{"skill", "criticality", "category",
                                     "related_skills_you_have": [...]}],
                "coverage_percentage": float (0-100),
                "critical_gap_count": int,
                "category_breakdown": {
                    category: {"required": int, "matched": int, "coverage": float}
                },
            }
        """
        resume_skills_set = {s.lower().strip() for s in resume_skills}
        jd_skill_criticality = self._extract_jd_skills_with_criticality(jd_text)

        matched_skills = []
        missing_skills = []

        for skill, criticality in jd_skill_criticality.items():
            category = self._skill_to_category.get(skill, "Other")
            if skill in resume_skills_set:
                matched_skills.append({
                    "skill": skill,
                    "criticality": criticality,
                    "category": category,
                })
            else:
                missing_skills.append({
                    "skill": skill,
                    "criticality": criticality,
                    "category": category,
                    "related_skills_you_have": self._find_related_skills(skill, category, resume_skills_set),
                })

        # Most urgent gaps first: critical, then important, then
        # preferred. Ties broken alphabetically for stable output.
        missing_skills.sort(key=lambda e: (CRITICALITY_RANK[e["criticality"]], e["skill"]))
        matched_skills.sort(key=lambda e: (CRITICALITY_RANK[e["criticality"]], e["skill"]))

        total_required = len(jd_skill_criticality)
        coverage = (len(matched_skills) / total_required * 100) if total_required else 100.0
        critical_gap_count = sum(1 for s in missing_skills if s["criticality"] == "critical")

        return {
            "total_required_skills": total_required,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "coverage_percentage": round(coverage, 2),
            "critical_gap_count": critical_gap_count,
            "category_breakdown": self._build_category_breakdown(jd_skill_criticality, resume_skills_set),
        }

    # ------------------------------------------------------------------
    # JD SKILL EXTRACTION + CRITICALITY CLASSIFICATION
    # ------------------------------------------------------------------
    def _extract_jd_skills_with_criticality(self, jd_text: str) -> dict:
        """
        Find every SKILL_DATABASE skill mentioned in the JD, and
        classify each one as "critical", "preferred", or "important"
        (default) based on nearby language in the SAME sentence.

        Returns:
            dict: {skill: criticality}
        """
        sentences = self._split_sentences(jd_text)
        skill_criticality = {}

        for sentence in sentences:
            sentence_lower = sentence.lower()
            has_critical_language = any(phrase in sentence_lower for phrase in CRITICAL_INDICATORS)
            has_preferred_language = any(phrase in sentence_lower for phrase in PREFERRED_INDICATORS)

            for skill in SKILL_DATABASE:
                pattern = r"\b" + re.escape(skill) + r"\b"
                if not re.search(pattern, sentence_lower):
                    continue

                if has_critical_language:
                    criticality = "critical"
                elif has_preferred_language:
                    criticality = "preferred"
                else:
                    criticality = "important"

                # A skill can appear in multiple sentences with
                # different framing (e.g. listed once in a "Required"
                # bullet and again in a summary paragraph). When that
                # happens, keep the MOST critical classification seen
                # -- better to flag a gap as too important than to
                # silently downgrade a real requirement.
                existing = skill_criticality.get(skill)
                if existing is None or CRITICALITY_RANK[criticality] < CRITICALITY_RANK[existing]:
                    skill_criticality[skill] = criticality

        return skill_criticality

    def _split_sentences(self, text: str) -> list:
        """
        Lightweight sentence/bullet splitter. Job descriptions are
        often bulleted rather than written in full grammatical
        sentences, so we split on sentence-ending punctuation AND
        newlines/bullets -- whichever comes first -- rather than
        relying on a full NLP sentence tokenizer (overkill for this).
        """
        if not text:
            return []
        # Split on '.', '!', '?', ';', or line breaks.
        raw_pieces = re.split(r"[.!?;\n]+", text)
        return [p.strip() for p in raw_pieces if p.strip()]

    # ------------------------------------------------------------------
    # RELATED SKILL SUGGESTIONS
    # ------------------------------------------------------------------
    def _find_related_skills(self, missing_skill: str, category: str, resume_skills_set: set) -> list:
        """
        For a missing JD skill, check whether the candidate already
        has OTHER skills in the same category (e.g. missing
        "tensorflow" but resume lists "pytorch" and "keras" -- both
        Data Science / ML skills). This is a much more actionable
        signal than a flat "missing" label.
        """
        category_skills = SKILL_CATEGORIES.get(category, [])
        related = [s for s in category_skills if s != missing_skill and s in resume_skills_set]
        return related[:MAX_RELATED_SKILLS]

    # ------------------------------------------------------------------
    # CATEGORY-LEVEL BREAKDOWN (useful for the dashboard in Phase 8)
    # ------------------------------------------------------------------
    def _build_category_breakdown(self, jd_skill_criticality: dict, resume_skills_set: set) -> dict:
        """
        Roll up coverage by category, e.g. "Cloud / DevOps: 1/4 (25%)".
        Only includes categories that the JD actually mentions at
        least one skill from -- an empty category tells the candidate
        nothing useful.
        """
        breakdown = {}
        for skill, _criticality in jd_skill_criticality.items():
            category = self._skill_to_category.get(skill, "Other")
            entry = breakdown.setdefault(category, {"required": 0, "matched": 0})
            entry["required"] += 1
            if skill in resume_skills_set:
                entry["matched"] += 1

        for entry in breakdown.values():
            entry["coverage"] = round(entry["matched"] / entry["required"] * 100, 2)

        return breakdown
