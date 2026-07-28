"""
models/nlp_loader.py
=====================
Loads and caches shared ML models (currently: the spaCy NLP pipeline).

Why this file exists:
    Loading spaCy's model from disk takes noticeable time (usually under
    a second, but it adds up). If every function that needs NLP loaded
    its own fresh copy of the model, we'd waste time and memory reloading
    it over and over -- especially inside a Streamlit app, which re-runs
    your script on every user interaction.

    `functools.lru_cache` ensures `get_spacy_model()` only actually loads
    the model the FIRST time it's called. Every call after that instantly
    returns the same cached object.

Beginner note:
    Later phases (semantic similarity) will add a similar loader here for
    the SentenceTransformer model, following the same caching pattern.
"""

from functools import lru_cache

import spacy

from config import SPACY_MODEL_NAME


@lru_cache(maxsize=1)
def get_spacy_model():
    """
    Load (once) and return the spaCy English NLP pipeline.

    Returns:
        spacy.lang.en.English: The loaded spaCy pipeline.

    Raises:
        OSError: If the model isn't installed. We re-raise with a
                 friendlier, actionable error message.
    """
    try:
        return spacy.load(SPACY_MODEL_NAME)
    except OSError as exc:
        # This is the #1 beginner error: forgetting to download the model.
        raise OSError(
            f"spaCy model '{SPACY_MODEL_NAME}' is not installed.\n"
            f"Fix this by running:\n"
            f"    python -m spacy download {SPACY_MODEL_NAME}\n"
        ) from exc
