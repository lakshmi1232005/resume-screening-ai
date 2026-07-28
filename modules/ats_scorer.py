"""
modules/ats_scorer.py
=======================
Computes the ATS (Applicant Tracking System) Score for a resume.

Real-world ATS systems (Workday, Taleo, Greenhouse, etc.) don't use AI
"understanding" -- they use rule-based checks: is the resume readable?
Do keywords match the job posting? Are standard sections present? This
module recreates that rule-based logic across 7 sub-scores, then
combines them into one weighted score out of 100 using the weights
defined in config.ATS_WEIGHTS.

Each sub-score is independently interpretable (e.g. "your Skills score
is 40/100 because you're missing 6 of 10 JD-required skills"), which
Phase 7 (Improvement Engine) will use to generate specific, explainable
suggestions.
"""

import re

from config import ATS_WEIGHTS, SKILL_DATABASE, STRONG_ACTION_VERBS
from modules.text_preprocessor import TextPreprocessor

# A "healthy" resume word-count range. Too short usually means missing
# detail (weak ATS parsing target); too long risks the ATS truncating
# content or a human reviewer skimming past key details.
IDEAL_WORD_COUNT_RANGE = (200, 1200)
ACCEPTABLE_WORD_COUNT_RANGE = (100, 1600)

TOTAL_POSSIBLE_SECTIONS = 7  # summary, skills, education, experience, projects, certifications, achievements


class ATSScorer:
    """
    Computes an ATS score (0-100) for a resume against a job description.

    Usage:
        scorer = ATSScorer()
        result = scorer.compute_ats_score(resume_data, jd_text)
        print(result["overall_score"], result["sub_scores"])
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def compute_ats_score(self, resume_data: dict, jd_text: str) -> dict:
        """
        Compute the overall ATS score plus a breakdown of each sub-score.

        Args:
            resume_data: The dict returned by ResumeParser.parse().
            jd_text: Raw job description text.

        Returns:
            dict: {
                "overall_score": float (0-100),
                "sub_scores": {formatting, keyword_match, skills,
                                education, projects, experience,
                                certifications} each 0-100
            }
        """
        sub_scores = {
            "formatting": self._score_formatting(resume_data),
            "keyword_match": self._score_keyword_match(resume_data["raw_text"], jd_text),
            "skills": self._score_skills(resume_data["skills"], jd_text),
            "education": self._score_education(resume_data["education"]),
            "projects": self._score_projects(resume_data["projects"]),
            "experience": self._score_experience(resume_data["experience"]),
            "certifications": self._score_certifications(resume_data["certifications"]),
        }

        # Weighted sum using the weights from config.py (they sum to 1.0,
        # enforced by an assert in config.py, so this always yields 0-100).
        overall = sum(sub_scores[key] * weight for key, weight in ATS_WEIGHTS.items())

        return {
            "overall_score": round(overall, 2),
            "sub_scores": {k: round(v, 2) for k, v in sub_scores.items()},
        }

    # ------------------------------------------------------------------
    # SUB-SCORE 1: FORMATTING (10% weight)
    # ------------------------------------------------------------------
    def _score_formatting(self, resume_data: dict) -> float:
        """
        Rewards structural completeness that real ATS parsers rely on:
            - Contact info present (email + phone): 30 pts
            - Standard sections detected: up to 40 pts
            - Healthy word count (not too sparse, not bloated): 30 pts
        """
        points = 0.0

        if resume_data.get("email"):
            points += 15
        if resume_data.get("phone"):
            points += 15

        sections_found = resume_data.get("sections_found", [])
        points += min(len(sections_found) / TOTAL_POSSIBLE_SECTIONS, 1.0) * 40

        word_count = len(resume_data.get("raw_text", "").split())
        ideal_min, ideal_max = IDEAL_WORD_COUNT_RANGE
        ok_min, ok_max = ACCEPTABLE_WORD_COUNT_RANGE
        if ideal_min <= word_count <= ideal_max:
            points += 30
        elif ok_min <= word_count < ideal_min or ideal_max < word_count <= ok_max:
            points += 15
        # else: 0 pts -- resume is unusually short or long

        return min(points, 100.0)

    # ------------------------------------------------------------------
    # SUB-SCORE 2: KEYWORD MATCH (25% weight)
    # ------------------------------------------------------------------
    def _score_keyword_match(self, resume_text: str, jd_text: str) -> float:
        """
        Percentage of the JD's meaningful (lemmatized) keywords that
        also appear somewhere in the resume. Uses clean_full/lemmatized
        tokens (not raw words) so "managing" in the resume still
        matches "manage" in the JD.
        """
        resume_tokens = set(self.preprocessor.get_lemmatized_tokens(resume_text))
        jd_tokens = set(self.preprocessor.get_lemmatized_tokens(jd_text))

        # Ignore very short tokens (1-2 chars) -- usually stray noise
        # (initials, units) rather than meaningful keywords.
        jd_keywords = {t for t in jd_tokens if len(t) > 2}
        if not jd_keywords:
            return 0.0

        matched = jd_keywords.intersection(resume_tokens)
        return (len(matched) / len(jd_keywords)) * 100

    # ------------------------------------------------------------------
    # SUB-SCORE 3: SKILLS (20% weight)
    # ------------------------------------------------------------------
    def _score_skills(self, resume_skills: list, jd_text: str) -> float:
        """
        Percentage of the JD's REQUIRED skills (detected via
        SKILL_DATABASE) that the resume's parsed skills list covers.
        """
        jd_text_lower = jd_text.lower()
        jd_skills = set()
        for skill in SKILL_DATABASE:
            pattern = r"\b" + re.escape(skill.lower()) + r"\b"
            if re.search(pattern, jd_text_lower):
                jd_skills.add(skill)

        if not jd_skills:
            # The JD didn't explicitly name any known skills -- we can't
            # fairly penalize the resume for missing something the JD
            # never specified, so we don't count this sub-score against it.
            return 100.0

        resume_skills_set = {s.lower() for s in resume_skills}
        matched = jd_skills.intersection(resume_skills_set)
        return (len(matched) / len(jd_skills)) * 100

    # ------------------------------------------------------------------
    # SUB-SCORE 4: EDUCATION (10% weight)
    # ------------------------------------------------------------------
    def _score_education(self, education_entries: list) -> float:
        """
        Simple presence check for Phase 5: full marks if an education
        section was found and non-empty, zero otherwise. (Relevance-
        based scoring, e.g. degree field matching the JD, is a natural
        future enhancement but out of scope for this phase.)
        """
        return 100.0 if education_entries else 0.0

    # ------------------------------------------------------------------
    # SUB-SCORE 5: PROJECTS (15% weight)
    # ------------------------------------------------------------------
    def _score_projects(self, project_entries: list) -> float:
        """
        Rewards not just HAVING projects, but describing them well:
            - Any projects present: 50 pts baseline
            - Uses strong action verbs (e.g. "developed", "built"): +25 pts
            - Includes numbers/metrics (e.g. "95% accuracy", "3 engineers"): +25 pts
        """
        return self._score_entries_quality(project_entries)

    # ------------------------------------------------------------------
    # SUB-SCORE 6: EXPERIENCE (15% weight)
    # ------------------------------------------------------------------
    def _score_experience(self, experience_entries: list) -> float:
        """Same quality heuristic as projects, applied to work experience."""
        return self._score_entries_quality(experience_entries)

    def _score_entries_quality(self, entries: list) -> float:
        """
        Shared scoring heuristic for bullet-point sections (projects,
        experience): presence + action-verb usage + quantifiable metrics.
        """
        if not entries:
            return 0.0

        points = 50.0  # baseline for having any entries at all
        combined_text = " ".join(entries).lower()

        has_action_verb = any(verb in combined_text for verb in STRONG_ACTION_VERBS)
        has_numbers = bool(re.search(r"\d", combined_text))

        if has_action_verb:
            points += 25
        if has_numbers:
            points += 25

        return min(points, 100.0)

    # ------------------------------------------------------------------
    # SUB-SCORE 7: CERTIFICATIONS (5% weight -- a bonus category)
    # ------------------------------------------------------------------
    def _score_certifications(self, certification_entries: list) -> float:
        """
        Certifications are a bonus signal with diminishing returns:
            0 certs -> 0 pts
            1 cert  -> 60 pts
            2 certs -> 80 pts
            3+ certs -> 100 pts
        """
        if not certification_entries:
            return 0.0
        count = len(certification_entries)
        return min(60 + (count - 1) * 20, 100.0)
