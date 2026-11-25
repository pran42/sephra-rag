"""Read-only access to the existing persistent ChromaDB collections.

This module never creates, deletes or modifies collections. Building the
database is the job of src/build_vector_db.py, which is guarded against
accidental re-runs.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sephra import config

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when the vector database is missing or incomplete."""


@lru_cache(maxsize=1)
def get_chroma_client():
    import chromadb  # lazy import

    sqlite_file = config.CHROMA_DIR / "chroma.sqlite3"
    if not sqlite_file.exists():
        # Checked BEFORE creating the client, because PersistentClient would
        # silently create a new empty database.
        raise VectorStoreError(
            f"No ChromaDB found at {config.CHROMA_DIR}. "
            "Build it once with: python src/build_vector_db.py --rebuild"
        )
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


@lru_cache(maxsize=4)
def get_collection(name: str):
    client = get_chroma_client()
    existing = {c.name for c in client.list_collections()}
    if name not in existing:
        raise VectorStoreError(
            f"Collection '{name}' not found in {config.CHROMA_DIR}. "
            f"Available: {sorted(existing)}"
        )
    collection = client.get_collection(name)
    if collection.count() == 0:
        raise VectorStoreError(f"Collection '{name}' is empty")
    return collection


def get_product_collection():
    return get_collection(config.PRODUCT_COLLECTION)


def get_review_collection():
    return get_collection(config.REVIEW_COLLECTION)
