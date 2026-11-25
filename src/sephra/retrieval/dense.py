"""Dense (semantic) retrieval over the existing ChromaDB collections.

Uses the stored bge-small-en-v1.5 vectors; nothing is re-embedded except
the user's query.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sephra.vector_store import get_product_collection, get_review_collection

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DenseHit:
    """One unique product (or review) found by vector search."""

    item_id: str          # product_id for products, parent review id for reviews
    parent_id: str        # parent document id
    similarity: float     # cosine similarity = 1 - cosine distance
    best_chunk_id: str    # the chunk that matched best


def _dedupe_hits(ids: list[str], metadatas: list[dict], distances: list[float],
                 key_field: str) -> list[DenseHit]:
    """Keep the best-scoring chunk per item (Chroma returns best-first)."""
    seen: dict[str, DenseHit] = {}
    for chunk_id, metadata, distance in zip(ids, metadatas, distances):
        metadata = metadata or {}
        key = metadata.get(key_field)
        parent_id = metadata.get("parent_id")
        if not key or not parent_id:
            logger.debug("Chunk %s lacks %s/parent_id; skipped", chunk_id, key_field)
            continue
        if key not in seen:
            seen[key] = DenseHit(item_id=key, parent_id=parent_id,
                                 similarity=1.0 - float(distance), best_chunk_id=chunk_id)
    return list(seen.values())


def _query(collection, embedding: list[float], n_results: int,
           where: dict[str, Any] | None) -> dict:
    n_results = max(1, min(n_results, collection.count()))
    kwargs: dict[str, Any] = {
        "query_embeddings": [embedding],
        "n_results": n_results,
        "include": ["metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)


def dense_product_search(embedding: list[float], n_chunks: int,
                         where: dict[str, Any] | None = None) -> list[DenseHit]:
    """Search product chunks, then collapse to unique products (best chunk wins)."""
    result = _query(get_product_collection(), embedding, n_chunks, where)
    if not result["ids"] or not result["ids"][0]:
        return []
    return _dedupe_hits(result["ids"][0], result["metadatas"][0],
                        result["distances"][0], key_field="product_id")


def dense_review_search(embedding: list[float], product_id: str,
                        n_chunks: int) -> list[DenseHit]:
    """Search ONLY the reviews of one product (mandatory product_id filter).

    Returns unique parent reviews. This is the only way reviews are ever
    retrieved: reviews are never searched across products.
    """
    if not product_id:
        raise ValueError("product_id is required for review search")
    result = _query(get_review_collection(), embedding, n_chunks,
                    where={"product_id": product_id})
    if not result["ids"] or not result["ids"][0]:
        return []
    return _dedupe_hits(result["ids"][0], result["metadatas"][0],
                        result["distances"][0], key_field="parent_id")
