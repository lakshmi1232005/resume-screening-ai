"""
modules/resume_parser.py
==========================
Core resume parsing logic.

Given a PDF resume, this module extracts:
    - Contact info: name, email, phone, LinkedIn, GitHub
    - Structured sections: skills, education, experience, projects,
      certifications, achievements, summary

Design approach:
    1. Extract raw text from the PDF (utils/pdf_utils.py).
    2. Run regex patterns over the WHOLE text for contact details
       (email/phone/links are format-based, so regex is reliable and
       fast -- no need for heavy ML here).
    3. Use spaCy's Named Entity Recognition (NER) to guess the
       candidate's NAME (names don't follow a fixed format, so regex
       is unreliable -- NER is the right tool).
    4. Split the resume into SECTIONS by detecting section header
       lines (e.g. "Skills", "Education") using the synonym map in
       config.py. Each section's raw content is then further parsed
       (e.g. skills are matched against a keyword database).

Beginner note:
    Resume formats vary wildly, so no parser (including this one) will
    be 100% perfect on every resume. This parser uses a "best effort"
    approach: it tries multiple strategies and gracefully returns
    partial results (with empty lists/None) rather than crashing when
    a section isn't found.
"""

import re
from typing import Optional

from config import SECTION_HEADERS, SKILL_DATABASE
from models.nlp_loader import get_spacy_model
from utils.pdf_utils import extract_text_from_pdf

# ----------------------------------------------------------------------
# REGEX PATTERNS
# ----------------------------------------------------------------------
# Compiled once at module load time for performance (avoids recompiling
# the same pattern on every resume parsed).

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# Matches common phone formats: +91 98765 43210, (123) 456-7890,
# 123-456-7890, 9876543210, etc. Intentionally permissive.
PHONE_PATTERN = re.compile(
    r"(\+?\d{1,3}[\s\-.]?)?(\(?\d{2,4}\)?[\s\-.]?){2,4}\d{3,4}"
)

LINKEDIN_PATTERN = re.compile(r"(https?://)?(www\.)?linkedin\.com/in/[A-Za-z0-9\-_/]+", re.IGNORECASE)
GITHUB_PATTERN = re.compile(r"(https?://)?(www\.)?github\.com/[A-Za-z0-9\-_/]+", re.IGNORECASE)


class ResumeParser:
    """
    Parses a resume PDF into structured data.

    Usage:
        parser = ResumeParser()
        data = parser.parse("uploads/john_doe_resume.pdf")
        print(data["name"], data["email"], data["skills"])
    """

    def __init__(self):
        # Load the spaCy model once per ResumeParser instance. Thanks to
        # the @lru_cache in nlp_loader.py, even creating many ResumeParser
        # instances won't reload the model from disk multiple times.
        self.nlp = get_spacy_model()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def parse(self, file_path: str) -> dict:
        """
        Parse a resume PDF and return a structured dictionary.

        Args:
            file_path: Path to the resume PDF file.

        Returns:
            dict: Structured resume data (see keys below).
        """
        raw_text = extract_text_from_pdf(file_path)
        sections = self._split_into_sections(raw_text)

        return {
            "raw_text": raw_text,
            "name": self._extract_name(raw_text),
            "email": self._extract_email(raw_text),
            "phone": self._extract_phone(raw_text),
            "linkedin": self._extract_linkedin(raw_text),
            "github": self._extract_github(raw_text),
            "summary": sections.get("summary", ""),
            "skills": self._extract_skills(raw_text, sections.get("skills", "")),
            "education": self._section_to_entries(sections.get("education", "")),
            "projects": self._section_to_entries(sections.get("projects", "")),
            "experience": self._section_to_entries(sections.get("experience", "")),
            "certifications": self._section_to_entries(sections.get("certifications", "")),
            "achievements": self._section_to_entries(sections.get("achievements", "")),
            "sections_found": list(sections.keys()),  # useful for error handling later
        }

    # ------------------------------------------------------------------
    # CONTACT INFO EXTRACTION (regex-based -- these have fixed formats)
    # ------------------------------------------------------------------
    def _extract_email(self, text: str) -> Optional[str]:
        match = EMAIL_PATTERN.search(text)
        return match.group(0) if match else None

    def _extract_phone(self, text: str) -> Optional[str]:
        # Only search the first ~15 lines: phone numbers usually appear
        # in the header, and searching the whole document risks false
        # positives from dates, years, or ID numbers in later sections.
        header_text = "\n".join(text.splitlines()[:15])
        match = PHONE_PATTERN.search(header_text)
        if match:
            digits_only = re.sub(r"\D", "", match.group(0))
            # A real phone number has at least 7 digits; anything shorter
            # is likely a false positive (e.g. a lone year like "2023").
            if len(digits_only) >= 7:
                return match.group(0).strip()
        return None

    def _extract_linkedin(self, text: str) -> Optional[str]:
        match = LINKEDIN_PATTERN.search(text)
        return match.group(0) if match else None

    def _extract_github(self, text: str) -> Optional[str]:
        match = GITHUB_PATTERN.search(text)
        return match.group(0) if match else None

    # ------------------------------------------------------------------
    # NAME EXTRACTION (NER-based -- names have no fixed format)
    # ------------------------------------------------------------------
    def _extract_name(self, text: str) -> Optional[str]:
        """
        Guess the candidate's name using spaCy NER.

        Strategy: a resume's name is almost always in the first 1-3
        lines, at the very top, before any contact details. We run NER
        only on this small header chunk (not the whole document) for
        two reasons: (1) speed, and (2) accuracy -- running NER on the
        whole resume risks picking up names of referees, companies, or
        even people mentioned in project descriptions instead of the
        candidate themselves.
        """
        header_lines = [ln.strip() for ln in text.splitlines()[:5] if ln.strip()]
        header_text = "\n".join(header_lines)
        if not header_text:
            return None

        # IMPORTANT: run NER line-by-line rather than on the whole joined
        # header block. If we feed spaCy multiple lines at once (e.g.
        # "Lakshmi S\nDoddaballapur"), it can merge an adjacent line
        # (like a city name right below the name, with no blank line
        # separating them) into a single PERSON entity, giving a wrong
        # result like "Lakshmi S Doddaballapur". Processing one line at
        # a time keeps each entity scoped to that single line.
        for line in header_lines:
            doc = self.nlp(line)
            person_entities = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
            if person_entities:
                # Return the first PERSON entity found on the first line
                # that contains one -- typically the name is the very
                # first thing on the resume.
                return person_entities[0]

        # Fallback heuristic: if NER found nothing (common with all-caps
        # names, which NER sometimes misses), assume the very first
        # non-empty line is the name -- but only if it "looks like" a
        # name (short, no @ or digits, which would indicate it's
        # actually an email/phone line instead).
        first_line = header_lines[0]
        looks_like_name = (
            len(first_line.split()) <= 4
            and "@" not in first_line
            and not any(char.isdigit() for char in first_line)
        )
        return first_line if looks_like_name else None

    # ------------------------------------------------------------------
    # SECTION SPLITTING
    # ------------------------------------------------------------------
    def _split_into_sections(self, text: str) -> dict:
        """
        Split resume text into named sections (skills, education, etc.)
        by detecting section header lines.

        Returns:
            dict: {section_key: section_content_text}
                  Only sections that were actually found are included.
        """
        lines = text.splitlines()

        # Build a flat lookup: synonym (lowercased) -> canonical section key.
        # e.g. "professional summary" -> "summary", "work experience" -> "experience"
        synonym_to_key = {}
        for key, synonyms in SECTION_HEADERS.items():
            for syn in synonyms:
                synonym_to_key[syn.lower()] = key

        # Find every line that looks like a section header.
        # Heuristic for "looks like a header": short line (<=40 chars),
        # and its cleaned text exactly matches one of our known synonyms.
        header_positions = []  # list of (line_index, section_key)
        for i, line in enumerate(lines):
            cleaned = line.strip().lower().strip(":").strip()
            if len(cleaned) <= 40 and cleaned in synonym_to_key:
                header_positions.append((i, synonym_to_key[cleaned]))

        # Slice the text between each header and the next one.
        sections = {}
        for idx, (line_no, key) in enumerate(header_positions):
            start = line_no + 1
            end = header_positions[idx + 1][0] if idx + 1 < len(header_positions) else len(lines)
            content = "\n".join(lines[start:end]).strip()
            if content:
                # If the same section name appears twice, keep the longer content.
                if key not in sections or len(content) > len(sections[key]):
                    sections[key] = content

        return sections

    def _section_to_entries(self, section_text: str) -> list:
        """
        Convert a raw section's text into a list of individual entries
        (roughly: bullet points or paragraphs), stripping empty lines.

        This is intentionally simple for Phase 2. Later phases (7:
        Improvement Engine) will analyze each entry's quality in depth.
        """
        if not section_text:
            return []
        # Split on newlines, strip bullet characters (-, *, •) and whitespace.
        entries = []
        for line in section_text.splitlines():
            cleaned = line.strip().lstrip("-*•").strip()
            if cleaned:
                entries.append(cleaned)
        return entries

    # ------------------------------------------------------------------
    # SKILLS EXTRACTION
    # ------------------------------------------------------------------
    def _extract_skills(self, full_text: str, skills_section_text: str) -> list:
        """
        Extract skills by matching against the curated SKILL_DATABASE
        (config.py). We search the WHOLE resume (not just the "Skills"
        section) because candidates often mention skills inside project
        or experience bullet points too (e.g. "Built a REST API using
        Flask" -- "flask" wouldn't be in the Skills section necessarily).

        Matching is case-insensitive and uses word boundaries so "r"
        (the language) doesn't match inside unrelated words like "your".
        """
        search_text = f"{full_text}\n{skills_section_text}".lower()
        found_skills = []

        for skill in SKILL_DATABASE:
            # Build a word-boundary regex per skill so "c" doesn't match
            # "connect", "c++" is matched literally (special chars escaped
            # via re.escape), etc.
            pattern = r"\b" + re.escape(skill.lower()) + r"\b"
            if re.search(pattern, search_text):
                found_skills.append(skill)

        return sorted(set(found_skills))
