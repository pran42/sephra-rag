"""The embedding model, loaded once and reused for every query."""

from __future__ import annotations

import logging
from functools import lru_cache

from sephra import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load BAAI/bge-small-en-v1.5 once per process (about 2 seconds)."""
    from sentence_transformers import SentenceTransformer  # heavy import, done lazily

    logger.info("Loading embedding model %s", config.EMBEDDING_MODEL_NAME)
    return load_hf_model(SentenceTransformer, config.EMBEDDING_MODEL_NAME)


def load_hf_model(model_class, name: str, **kwargs):
    """Load a Hugging Face model from the local cache first.

    Without this, every start-up makes network calls to check for model
    updates, which made the first query ~10 s slower on this machine.
    Falls back to downloading if the model is not cached yet.
    """
    try:
        return model_class(name, local_files_only=True, **kwargs)
    except Exception:  # not cached yet (or cache incomplete): download once
        logger.info("Model %s not in local cache; downloading", name)
        return model_class(name, **kwargs)


@lru_cache(maxsize=512)
def _embed_query_cached(text: str) -> tuple[float, ...]:
    model = get_embedding_model()
    vector = model.encode(
        [config.BGE_QUERY_INSTRUCTION + text],
        normalize_embeddings=True,   # same normalisation as the stored vectors
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    return tuple(float(x) for x in vector)


def embed_query(text: str) -> list[float]:
    """Embed a search query with the BGE query instruction.

    Documents were embedded WITHOUT the instruction; queries get it. This
    asymmetric setup is what BGE was trained for (short query vs. passage).
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed an empty query")
    return list(_embed_query_cached(text.strip()))
