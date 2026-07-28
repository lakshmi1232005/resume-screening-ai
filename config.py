"""
config.py
==========
Central configuration file for the AI-Powered Resume Screening System.

Why this file exists:
    Every module (parser, scorer, dashboard, report generator, etc.) needs
    shared settings -- folder paths, model names, scoring weights, and
    skill keyword lists. Instead of hardcoding these values in multiple
    files (which is error-prone and hard to maintain), we define them
    ONCE here and import them everywhere else.

Beginner note:
    You will see `from config import X` in almost every other file in
    this project. If you ever want to change a folder path or a scoring
    weight, this is the ONLY file you need to edit.
"""

import os

# ------------------------------------------------------------------
# BASE DIRECTORY
# ------------------------------------------------------------------
# os.path.dirname(os.path.abspath(__file__)) gives us the absolute path
# to the folder this config.py file lives in. We use this as the "root"
# so that all other paths work no matter where the project is run from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# FOLDER PATHS
# ------------------------------------------------------------------
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")   # Raw uploaded resumes / JD files
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")   # Intermediate outputs (parsed JSON, rewritten text, etc.)
REPORTS_DIR = os.path.join(BASE_DIR, "reports")   # Final generated PDF reports
ASSETS_DIR = os.path.join(BASE_DIR, "assets")     # Logos, icons, static images for the UI

# Make sure these folders always exist. exist_ok=True means "don't
# raise an error if the folder is already there" -- safe to run every time.
for _folder in (UPLOADS_DIR, OUTPUTS_DIR, REPORTS_DIR, ASSETS_DIR):
    os.makedirs(_folder, exist_ok=True)

# ------------------------------------------------------------------
# FILE UPLOAD LIMITS
# ------------------------------------------------------------------
MAX_FILE_SIZE_MB = 10                     # Reject uploads larger than this
ALLOWED_RESUME_EXTENSIONS = (".pdf",)      # Resumes must be PDF
ALLOWED_JD_EXTENSIONS = (".pdf", ".txt")   # Job description can be PDF or TXT

# ------------------------------------------------------------------
# NLP / MODEL SETTINGS
# ------------------------------------------------------------------
SPACY_MODEL_NAME = "en_core_web_sm"                    # Lightweight English spaCy model
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"        # Embedding model for semantic similarity

# ------------------------------------------------------------------
# ATS SCORING WEIGHTS
# ------------------------------------------------------------------
# The ATS score (out of 100) is a weighted sum of several sub-scores.
# All weights must add up to 1.0 (i.e. 100%).
ATS_WEIGHTS = {
    "formatting": 0.10,     # Is the resume well-structured / parseable?
    "keyword_match": 0.25,  # Overlap of important keywords with the JD
    "skills": 0.20,         # Skill coverage against the JD
    "education": 0.10,      # Education section present & relevant
    "projects": 0.15,       # Quality/presence of project descriptions
    "experience": 0.15,     # Quality/presence of work experience
    "certifications": 0.05, # Bonus for relevant certifications
}

# Sanity check performed at import time: catches config typos immediately
# instead of failing silently later during scoring.
assert abs(sum(ATS_WEIGHTS.values()) - 1.0) < 1e-6, "ATS_WEIGHTS must sum to 1.0"

# ------------------------------------------------------------------
# FINAL SCORE BLEND
# ------------------------------------------------------------------
# The "Final Score" shown in candidate ranking blends the semantic
# match score (how well the resume's meaning matches the JD) with the
# rule-based ATS score.
FINAL_SCORE_WEIGHTS = {
    "match_score": 0.5,
    "ats_score": 0.5,
}
assert abs(sum(FINAL_SCORE_WEIGHTS.values()) - 1.0) < 1e-6, "FINAL_SCORE_WEIGHTS must sum to 1.0"

# ------------------------------------------------------------------
# SKILL KEYWORD DATABASE
# ------------------------------------------------------------------
# A curated master list of common technical & soft skills used for:
#   1) Extracting skills from resumes/JDs via keyword matching
#   2) Suggesting "recommended skills" during skill-gap analysis
#
# NOTE: This list is intentionally broad but not exhaustive. It will be
# expanded in Phase 2 (Resume Parser) and Phase 6 (Skill Gap Analysis).
SKILL_DATABASE = [
    # Programming languages
    "python", "java", "c++", "c", "javascript", "typescript", "sql", "r", "go", "rust",
    # Web development
    "html", "css", "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    # Data science / ML
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "nlp",
    "machine learning", "deep learning", "data analysis", "data visualization",
    "computer vision", "opencv",
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "oracle",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "jenkins", "terraform",
    # Tools
    "git", "github", "jira", "linux", "excel", "power bi", "tableau",
    # Soft skills
    "communication", "leadership", "teamwork", "problem solving", "time management",
]

# ------------------------------------------------------------------
# SKILL CATEGORIES (Phase 6 -- Skill Gap Analysis)
# ------------------------------------------------------------------
# Groups SKILL_DATABASE by family so the Skill Gap Analyzer can:
#   1) Report gaps at a category level (e.g. "0/3 Cloud / DevOps skills")
#   2) Suggest "related skills you already have" when a specific JD
#      skill is missing but the resume has something in the same
#      family (e.g. missing "tensorflow" but resume already lists
#      "pytorch" -- a much more useful signal than a flat miss).
#
# NOTE: kept in sync with SKILL_DATABASE above -- every skill in
# SKILL_DATABASE belongs to exactly one category here. The assertion
# below catches drift between the two lists at import time.
SKILL_CATEGORIES = {
    "Programming Languages": ["python", "java", "c++", "c", "javascript", "typescript", "sql", "r", "go", "rust"],
    "Web Development": ["html", "css", "react", "angular", "vue", "node.js", "django", "flask", "fastapi"],
    "Data Science / ML": [
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "nlp",
        "machine learning", "deep learning", "data analysis", "data visualization",
        "computer vision", "opencv",
    ],
    "Databases": ["mysql", "postgresql", "mongodb", "redis", "oracle"],
    "Cloud / DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "jenkins", "terraform"],
    "Tools": ["git", "github", "jira", "linux", "excel", "power bi", "tableau"],
    "Soft Skills": ["communication", "leadership", "teamwork", "problem solving", "time management"],
}

_categorized_skills = sorted(s for skills in SKILL_CATEGORIES.values() for s in skills)
assert _categorized_skills == sorted(SKILL_DATABASE), (
    "SKILL_CATEGORIES must contain exactly the same skills as SKILL_DATABASE "
    "-- did you add/remove a skill in one list but not the other?"
)

# ------------------------------------------------------------------
# JD REQUIREMENT-LEVEL INDICATORS (Phase 6 -- Skill Gap Analysis)
# ------------------------------------------------------------------
# Phrases used to guess whether a skill mentioned in the JD is a hard
# "must-have" requirement or just a "nice-to-have" preference, by
# checking for these phrases in the same sentence as the skill
# mention. A skill with no nearby indicator either way defaults to a
# middle "important" criticality rather than being over- or under-
# weighted by a guess.
CRITICAL_INDICATORS = [
    "required", "require", "must have", "must-have", "mandatory", "essential",
    "minimum qualification", "minimum requirement", "prerequisite",
    "need to have", "should have", "strong knowledge of", "proficient in",
    "expert in", "hands-on experience",
]
PREFERRED_INDICATORS = [
    "preferred", "nice to have", "nice-to-have", "a plus", "is a plus",
    "bonus", "desirable", "good to have", "familiarity with",
    "exposure to", "advantageous", "optional", "added advantage",
]

# ------------------------------------------------------------------
# ACTION VERBS
# ------------------------------------------------------------------
# Used by the Resume Improvement Engine (Phase 7) to detect weak,
# passive phrasing (e.g. "worked on", "helped with") and recommend
# strong action verbs instead.
STRONG_ACTION_VERBS = [
    "developed", "designed", "built", "implemented", "led", "managed",
    "optimized", "automated", "engineered", "architected", "launched",
    "improved", "increased", "reduced", "achieved", "delivered",
    "spearheaded", "streamlined", "created", "analyzed", "deployed",
]

WEAK_PHRASES = [
    "worked on", "helped with", "responsible for", "involved in",
    "assisted with", "participated in", "was part of",
]

# ------------------------------------------------------------------
# SECTION HEADER SYNONYMS
# ------------------------------------------------------------------
# Resumes use inconsistent section titles. This map helps the parser
# (Phase 2) recognize a section regardless of exact wording.
SECTION_HEADERS = {
    "summary": ["summary", "objective", "professional summary", "profile"],
    "skills": ["skills", "technical skills", "core competencies", "soft skills"],
    "education": ["education", "academic background", "qualifications"],
    "experience": ["experience", "work experience", "employment history", "professional experience", "internship experience", "internship", "internships"],
    "projects": ["projects", "academic projects", "personal projects"],
    "certifications": ["certifications", "certificates", "licenses"],
    "achievements": ["achievements", "awards", "honors"],
}
