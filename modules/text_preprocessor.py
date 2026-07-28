"""
modules/text_preprocessor.py
==============================
Text cleaning and normalization utilities.

IMPORTANT DESIGN DECISION -- please read before using this module:
--------------------------------------------------------------------
This project uses text in TWO very different ways downstream, and each
needs a DIFFERENT kind of cleaning:

1. SEMANTIC MATCHING (Phase 4 -- Sentence Transformers)
   Sentence embedding models like `all-MiniLM-L6-v2` are trained on
   natural, grammatical sentences. Feeding them a lemmatized,
   stopword-stripped "bag of words" (e.g. "develop model predict churn"
   instead of "Developed a model to predict customer churn") actually
   HURTS similarity quality, because the model loses sentence structure
   and context it relies on.
   -> Use `clean_light()` for this: just whitespace/encoding cleanup,
      no lemmatization or stopword removal.

2. KEYWORD / ATS MATCHING (Phase 5-6 -- keyword overlap, TF-IDF style
   scoring, skill-gap analysis)
   Here we WANT lowercase, lemmatized, stopword-free tokens, because we
   are doing exact/fuzzy keyword comparison, not measuring sentence
   meaning. Lemmatizing "managing" / "managed" / "manages" down to
   "manage" helps us match keywords that would otherwise be missed by
   naive string comparison.
   -> Use `clean_full()` or `get_lemmatized_tokens()` for this.

Beginner takeaway: always ask "am I about to feed this to the
sentence-transformer, or am I comparing keywords?" and pick the
matching function below.
"""

import re
import string

from models.nlp_loader import get_spacy_model


class TextPreprocessor:
    """
    Provides text cleaning utilities at two levels:
        - light cleaning (for embeddings / semantic similarity)
        - full cleaning (for keyword / ATS matching)
    """

    # spaCy's small English model (en_core_web_sm) occasionally produces
    # lemmas that are technically "correct" grammar but wrong for our
    # purposes -- most notably "data" -> "datum". Since "data" is one of
    # the single most common words in tech resumes/JDs ("data analysis",
    # "data science", "big data"), silently mangling it into "datum"
    # would break keyword/skill matching downstream. This map patches
    # known cases after lemmatization.
    LEMMA_CORRECTIONS = {
        "datum": "data",
    }

    def __init__(self):
        self.nlp = get_spacy_model()

    # ------------------------------------------------------------------
    # LIGHT CLEANING -- safe to use before feeding text to the
    # Sentence Transformer (Phase 4). Preserves grammar and meaning.
    # ------------------------------------------------------------------
    def clean_light(self, text: str) -> str:
        """
        Minimal cleanup that preserves sentence structure and meaning:
            - Collapse repeated whitespace/newlines into single spaces
            - Strip stray bullet characters (•, -, *) left over from PDF extraction
            - Remove non-printable / control characters
            - Strip leading/trailing whitespace

        Does NOT lowercase, lemmatize, or remove stopwords -- those
        would hurt semantic embedding quality.
        """
        if not text:
            return ""

        # Remove non-printable control characters (common artifact of
        # PDF text extraction, e.g. form-feed characters between pages).
        text = "".join(ch for ch in text if ch in string.printable or ch.isalpha())

        # Strip common bullet markers so they don't get treated as words.
        text = re.sub(r"[•▪●·]", " ", text)

        # Collapse ANY run of whitespace (spaces, tabs, newlines) into
        # a single space. This turns multi-line PDF text into clean,
        # readable paragraphs for the embedding model.
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # ------------------------------------------------------------------
    # FULL CLEANING -- for keyword matching / ATS scoring (Phase 5-6).
    # Aggressively normalizes text down to comparable root words.
    # ------------------------------------------------------------------
    def clean_full(self, text: str) -> str:
        """
        Full NLP preprocessing pipeline, returned as a single string:
            1. Light clean (whitespace/bullets/control chars)
            2. Lowercase
            3. Remove punctuation
            4. Remove stopwords (e.g. "the", "is", "and")
            5. Lemmatize (e.g. "managing" -> "manage", "projects" -> "project")

        Returns:
            str: Cleaned text as space-joined lemmas, e.g.
                 "develop customer churn prediction model python"
        """
        tokens = self.get_lemmatized_tokens(text)
        return " ".join(tokens)

    def get_lemmatized_tokens(self, text: str) -> list:
        """
        Same pipeline as clean_full(), but returns a LIST of tokens
        instead of a joined string. Useful when you need to do set
        operations (e.g. skill-gap comparison, keyword overlap counts)
        rather than string comparison.

        Returns:
            list[str]: Lemmatized, lowercased, stopword-free tokens.
        """
        text = self.clean_light(text)
        if not text:
            return []

        doc = self.nlp(text.lower())

        tokens = []
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space:
                continue  # drop stopwords, punctuation, whitespace tokens
            lemma = token.lemma_.strip()
            if not lemma:
                continue  # safety: drop empty lemmas
            # Apply known lemma corrections (e.g. "datum" -> "data").
            lemma = self.LEMMA_CORRECTIONS.get(lemma, lemma)
            tokens.append(lemma)
        return tokens

    # ------------------------------------------------------------------
    # HELPER: remove a fixed set of "resume noise" characters that
    # commonly survive PDF extraction (page numbers, odd bullet glyphs).
    # ------------------------------------------------------------------
    def remove_pdf_artifacts(self, text: str) -> str:
        """
        Strips a few common PDF-extraction artifacts:
            - Standalone page numbers on their own line (e.g. "Page 2 of 3")
            - Repeated header/footer separators (e.g. "----------")
        """
        text = re.sub(r"(?im)^page\s*\d+(\s*of\s*\d+)?$", "", text)
        text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
        return text
