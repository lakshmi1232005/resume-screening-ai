"""
models/embedding_loader.py
============================
Loads and caches the Sentence Transformer embedding model.

Same reasoning as models/nlp_loader.py: loading `all-MiniLM-L6-v2` from
disk (or downloading it on first run) takes a few seconds. We only want
to pay that cost ONCE per app session, not on every single similarity
calculation -- so we cache it with @lru_cache.

Note on first run:
    The very first time this runs on a machine, sentence-transformers
    will download the model (~80MB) from Hugging Face and cache it
    locally (usually under ~/.cache/huggingface). This requires an
    internet connection once; after that, it loads from the local cache
    and works offline.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import SENTENCE_TRANSFORMER_MODEL


@lru_cache(maxsize=1)
def get_sentence_transformer() -> SentenceTransformer:
    """
    Load (once) and return the cached Sentence Transformer model.

    Returns:
        SentenceTransformer: The loaded embedding model.

    Raises:
        Exception: Re-raised with a friendlier message if the model
                   can't be downloaded/loaded (e.g. no internet on
                   first run, or a corrupted cache).
    """
    try:
        return SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Sentence Transformer model "
            f"'{SENTENCE_TRANSFORMER_MODEL}'.\n"
            f"If this is the first run, make sure you have an internet "
            f"connection so the model (~80MB) can be downloaded once.\n"
            f"Original error: {exc}"
        ) from exc
