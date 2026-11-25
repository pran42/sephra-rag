"""Optional cross-encoder reranking of the top fused candidates.

A bi-encoder (our BGE embeddings) encodes query and product separately and
compares vectors: fast, but it never "reads" them together. A cross-encoder
reads the (query, product) pair jointly and outputs a relevance score:
more accurate, but far slower, so it only reranks a short candidate list
(``rerank_candidates``, default 25).

Benchmarked models (see docs/evaluation_report.md for the results):
  minilm   cross-encoder/ms-marco-MiniLM-L-6-v2   ~22M params
  bge-base BAAI/bge-reranker-base                 ~278M params
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sephra.catalog import Product

logger = logging.getLogger(__name__)

RERANKER_MODELS: dict[str, str] = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-base": "BAAI/bge-reranker-base",
}


@lru_cache(maxsize=2)
def get_reranker(name: str):
    if name not in RERANKER_MODELS:
        raise ValueError(f"Unknown reranker {name!r}; choose from {sorted(RERANKER_MODELS)}")
    from sentence_transformers import CrossEncoder  # lazy import

    from sephra.embeddings import load_hf_model

    logger.info("Loading reranker %s", RERANKER_MODELS[name])
    return load_hf_model(CrossEncoder, RERANKER_MODELS[name], max_length=512)


def rerank_text(product: Product) -> str:
    """Product description for the cross-encoder. Ingredients go last, so
    they are the part cut off if the text exceeds 512 tokens."""
    parts = [f"{product.name} by {product.brand}.", f"Category: {product.category_path}."]
    if product.highlights:
        parts.append("Highlights: " + ", ".join(product.highlights) + ".")
    if product.ingredients:
        parts.append("Ingredients: " + product.ingredients[:1500])
    return " ".join(parts)


def rerank_scores(name: str, query: str, products: list[Product]) -> list[float]:
    """Relevance score for each product (higher = more relevant)."""
    if not products:
        return []
    model = get_reranker(name)
    pairs = [(query, rerank_text(product)) for product in products]
    scores = model.predict(pairs, show_progress_bar=False, batch_size=16)
    return [float(score) for score in scores]
