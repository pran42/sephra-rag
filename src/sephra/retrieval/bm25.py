"""BM25 keyword retrieval over complete product records.

Why BM25 next to dense retrieval?
- Dense vectors capture meaning ("hydrating" ~ "moisture boost") but can
  blur exact terms: brand names, product types, tags like "Oil Free".
- BM25 rewards exact term matches, weighted by how rare the term is.

What is indexed: product name (twice, as a cheap field boost), brand,
category path and highlight tags. Ingredient lists are deliberately NOT
indexed: they are long INCI lists that would dominate BM25's length
normalisation, and explicit ingredient requests are handled by the hard
ingredient filter instead.

The index is built in memory from the catalogue in about a second.
No embeddings are touched.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np

from sephra.catalog import Catalog, Product, get_catalog
from sephra.query.rule_parser import fold

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "of", "with", "to", "in", "on", "my", "me",
    "i", "is", "it", "that", "this", "be", "are", "by", "from", "at", "as", "you",
    "your", "some", "something", "product", "products", "show", "find", "need", "want",
    "recommend", "suggest", "best", "good", "please", "looking", "which", "what",
}


def _stem(word: str) -> str:
    """Very light plural stripping: 'moisturizers' -> 'moisturizer'."""
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", fold(text)) if w not in STOPWORDS]


def bm25_document(product: Product) -> str:
    return " ".join([product.name, product.name, product.brand, product.category_path,
                     " ".join(product.highlights)])


class BM25ProductIndex:
    def __init__(self, catalog: Catalog) -> None:
        from rank_bm25 import BM25Okapi

        self.product_ids = list(catalog.products)
        corpus = [tokenize(bm25_document(catalog.products[pid])) for pid in self.product_ids]
        self._bm25 = BM25Okapi(corpus)
        self._position = {pid: i for i, pid in enumerate(self.product_ids)}

    def search(self, query: str, eligible_ids: set[str] | None = None,
               top_n: int = 50) -> list[tuple[str, float]]:
        """Return (product_id, score) for the top products with score > 0."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        if eligible_ids is not None:
            mask = np.zeros(len(scores), dtype=bool)
            for pid in eligible_ids:
                position = self._position.get(pid)
                if position is not None:
                    mask[position] = True
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:top_n]
        return [(self.product_ids[i], float(scores[i])) for i in order if scores[i] > 0]


@lru_cache(maxsize=1)
def get_bm25_index() -> BM25ProductIndex:
    logger.info("Building BM25 index over product records")
    return BM25ProductIndex(get_catalog())
