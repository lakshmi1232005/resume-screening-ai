"""
modules/semantic_matcher.py
=============================
Computes semantic similarity between a resume and a job description
using Sentence Transformer embeddings + cosine similarity.

Why semantic similarity (in addition to keyword matching)?
    Keyword matching (Phase 5-6) catches EXACT word overlap, but misses
    meaning. A resume that says "constructed predictive models" and a
    JD that asks for "machine learning experience" share zero exact
    keywords, yet clearly describe related work. Sentence embeddings
    capture that MEANING-level similarity, which keyword matching can't.

Why we use `clean_light()` (not `clean_full()`) before embedding:
    See the docstring in modules/text_preprocessor.py -- embedding
    models need natural sentence structure, not lemmatized/stopword-
    stripped text, to produce accurate similarity scores.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models.embedding_loader import get_sentence_transformer
from modules.text_preprocessor import TextPreprocessor

# ----------------------------------------------------------------------
# CALIBRATION CONSTANTS for converting raw cosine similarity -> 0-100%
# ----------------------------------------------------------------------
# Raw cosine similarity from MiniLM embeddings does NOT naturally span
# the full 0-1 range for real-world resume/JD pairs. Empirically:
#   - Completely unrelated texts (e.g. a chef resume vs a software JD)
#     still tend to score around ~0.10-0.20 (embeddings share some
#     baseline "this is English text" similarity).
#   - Excellent, highly relevant matches rarely exceed ~0.70-0.80.
# If we displayed the raw cosine value directly, EVERY resume would
# look like a mediocre "40-60%" match regardless of actual quality,
# which is confusing and unhelpful for users. So we rescale the
# realistic range (MIN_SIM to MAX_SIM) to a full 0-100% scale.
MIN_SIM = 0.15
MAX_SIM = 0.75


class SemanticMatcher:
    """
    Computes embedding-based semantic similarity between resume text
    and job description text.

    Usage:
        matcher = SemanticMatcher()
        percentage = matcher.compute_match_percentage(resume_text, jd_text)
    """

    def __init__(self):
        self.model = get_sentence_transformer()
        self.preprocessor = TextPreprocessor()
        # Simple in-memory cache: avoids re-encoding the SAME text
        # (e.g. the job description, which stays constant while many
        # resumes are compared against it in a ranking session).
        self._embedding_cache = {}

    # ------------------------------------------------------------------
    # EMBEDDING
    # ------------------------------------------------------------------
    def compute_embedding(self, text: str) -> np.ndarray:
        """
        Convert text into a dense embedding vector.

        Args:
            text: Raw text (resume or JD). Will be lightly cleaned
                  (whitespace/artifacts only) before encoding.

        Returns:
            np.ndarray: The embedding vector (384-dim for MiniLM-L6-v2).
        """
        cleaned = self.preprocessor.clean_light(text)

        if cleaned in self._embedding_cache:
            return self._embedding_cache[cleaned]

        if not cleaned:
            # Guard against empty text (e.g. a section that wasn't
            # found in the resume) -- return a zero vector instead of
            # crashing. Cosine similarity with a zero vector is 0,
            # which is the semantically correct "no match" result.
            dim = self.model.get_sentence_embedding_dimension()
            embedding = np.zeros(dim, dtype=np.float32)
        else:
            embedding = self.model.encode(cleaned, convert_to_numpy=True)

        self._embedding_cache[cleaned] = embedding
        return embedding

    # ------------------------------------------------------------------
    # RAW COSINE SIMILARITY
    # ------------------------------------------------------------------
    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute raw cosine similarity between two texts' embeddings.

        Returns:
            float: Cosine similarity, typically in range [0, 1] for
                   related text (can technically go slightly negative
                   for very dissimilar embeddings, though rare in practice).
        """
        emb_a = self.compute_embedding(text_a)
        emb_b = self.compute_embedding(text_b)

        # sklearn's cosine_similarity expects 2D arrays (one row per
        # sample), so we reshape our single vectors to shape (1, dim).
        similarity = cosine_similarity(
            emb_a.reshape(1, -1), emb_b.reshape(1, -1)
        )[0][0]
        return float(similarity)

    # ------------------------------------------------------------------
    # CALIBRATED MATCH PERCENTAGE (the number shown to users)
    # ------------------------------------------------------------------
    def compute_match_percentage(self, resume_text: str, jd_text: str) -> float:
        """
        Compute the resume-to-JD match percentage (0-100), rescaled
        from raw cosine similarity using the MIN_SIM/MAX_SIM calibration
        so scores are intuitive (unrelated ~0-10%, strong match ~80-100%).

        Returns:
            float: Match percentage, rounded to 2 decimal places.
        """
        raw_similarity = self.compute_similarity(resume_text, jd_text)
        raw_similarity = max(raw_similarity, 0.0)  # clip any negative values to 0

        scaled = (raw_similarity - MIN_SIM) / (MAX_SIM - MIN_SIM)
        scaled = max(0.0, min(1.0, scaled))  # clip to [0, 1]

        return round(scaled * 100, 2)

    # ------------------------------------------------------------------
    # SECTION-LEVEL BREAKDOWN (useful for the dashboard in Phase 8)
    # ------------------------------------------------------------------
    def compute_section_similarity(self, resume_sections: dict, jd_text: str) -> dict:
        """
        Compute the match percentage of EACH resume section individually
        against the job description. This gives a more granular view
        than a single overall score -- e.g. "your Projects section
        matches well (78%) but your Skills section is weak (32%)".

        Args:
            resume_sections: dict of {section_name: content}, where
                              content can be a string or a list of
                              strings (bullet entries).
            jd_text: The job description's raw text.

        Returns:
            dict: {section_name: match_percentage}
        """
        results = {}
        for section_name, content in resume_sections.items():
            # Normalize list-based sections (e.g. projects, experience
            # bullet lists) into a single string before embedding.
            if isinstance(content, list):
                content = " ".join(content)

            if not content or not content.strip():
                results[section_name] = 0.0
                continue

            results[section_name] = self.compute_match_percentage(content, jd_text)

        return results
