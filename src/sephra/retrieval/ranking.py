"""Candidate generation and ranking for products.

Pipeline (inside the eligible set produced by the hard filters):

    dense (Chroma, filtered)  --+
                               +--> RRF fusion --> [optional cross-encoder rerank]
    BM25 (eligible ids only) --+         |
                                         v
            relevance (0-1, min-max normalised within the candidate pool)
          + skin-type tag boost (<= 0.10, product highlights match the user's skin type)
          + rating boost        (<= 0.05, small on purpose)
                                         |
                                         v
            variant diversity: one slot per product family

Design choices worth defending in an interview:
* Relevance dominates. Boosts can reorder close candidates but cannot pull
  an irrelevant product to the top.
* Review COUNT is not a ranking signal: more reviews in one dataset
  partition says nothing about suitability, and zero-review products must
  stay recommendable.
* Ratings are a small tie-breaker, never a filter (unless the user asked for one).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from sephra.catalog import Catalog, Product
from sephra.config import RetrievalSettings
from sephra.query.filters import FilterResult
from sephra.query.schema import StructuredQuery
from sephra.query.vocabulary import SKIN_TYPE_TAGS
from sephra.retrieval.bm25 import get_bm25_index
from sephra.retrieval.dense import dense_product_search
from sephra.retrieval.fusion import reciprocal_rank_fusion
from sephra.retrieval.reranker import rerank_scores

logger = logging.getLogger(__name__)

RetrievalMode = Literal["dense", "hybrid", "hybrid_rerank"]


@dataclass
class ProductCandidate:
    product_id: str
    dense_rank: int | None = None
    dense_similarity: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    relevance: float = 0.0               # normalised 0-1
    skin_boost: float = 0.0
    rating_boost: float = 0.0
    matched_skin_tags: list[str] = field(default_factory=list)
    variant_ids: list[str] = field(default_factory=list)   # same family, not shown separately

    @property
    def final_score(self) -> float:
        return self.relevance + self.skin_boost + self.rating_boost


def _min_max(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [1.0 for _ in values]
    return [(v - low) / (high - low) for v in values]


def skin_tag_matches(product: Product, query: StructuredQuery) -> list[str]:
    wanted: set[str] = set()
    for skin in query.skin_types:
        wanted |= SKIN_TYPE_TAGS.get(skin.value, set())
    return sorted(wanted & set(product.highlights))


def generate_candidates(query: StructuredQuery, filters: FilterResult, embedding: list[float],
                        catalog: Catalog, settings: RetrievalSettings,
                        mode: RetrievalMode, reranker: str | None = None
                        ) -> list[ProductCandidate]:
    """Return eligible candidates with relevance scores (not yet diversified)."""
    eligible = filters.eligible_ids
    if not eligible:
        return []

    # 1. Dense retrieval inside the eligible set.
    dense_hits = [h for h in dense_product_search(embedding, settings.dense_chunk_candidates,
                                                  where=filters.chroma_where)
                  if h.item_id in eligible]
    candidates: dict[str, ProductCandidate] = {}
    for rank, hit in enumerate(dense_hits, start=1):
        candidates[hit.item_id] = ProductCandidate(hit.item_id, dense_rank=rank,
                                                   dense_similarity=hit.similarity)

    if mode == "dense":
        ordered = list(candidates.values())
        for candidate, value in zip(ordered, _min_max([c.dense_similarity for c in ordered] or [0])):
            candidate.relevance = value
    else:
        # 2. BM25 inside the same eligible set.
        bm25_hits = get_bm25_index().search(
            query.semantic_query, None if filters.is_unfiltered else eligible,
            top_n=settings.bm25_candidates)
        for rank, (pid, score) in enumerate(bm25_hits, start=1):
            candidate = candidates.setdefault(pid, ProductCandidate(pid))
            candidate.bm25_rank, candidate.bm25_score = rank, score

        # 3. Fuse the two rankings.
        fused = reciprocal_rank_fusion({"dense": [h.item_id for h in dense_hits],
                                        "bm25": [pid for pid, _ in bm25_hits]}, k=settings.rrf_k)
        for pid, score, _ in fused:
            candidates[pid].rrf_score = score
        ordered = [candidates[pid] for pid, _, _ in fused]
        for candidate, value in zip(ordered, _min_max([c.rrf_score for c in ordered] or [0])):
            candidate.relevance = value

        # 4. Optional cross-encoder rerank of the top fused candidates.
        if mode == "hybrid_rerank":
            top = ordered[: settings.rerank_candidates]
            scores = rerank_scores(reranker or "minilm", query.semantic_query,
                                   [catalog.products[c.product_id] for c in top])
            for candidate, score in zip(top, scores):
                candidate.rerank_score = score
            normalised = _min_max(scores) if scores else []
            for candidate, value in zip(top, normalised):
                candidate.relevance = value
            # Candidates outside the reranked window stay below all reranked ones.
            for candidate in ordered[settings.rerank_candidates:]:
                candidate.relevance = -1.0
            ordered = top + ordered[settings.rerank_candidates:]

    # 5. Small, explainable boosts.
    for candidate in ordered:
        product = catalog.products[candidate.product_id]
        candidate.matched_skin_tags = skin_tag_matches(product, query)
        if candidate.matched_skin_tags:
            candidate.skin_boost = settings.skin_type_tag_boost
        if product.rating is not None:
            candidate.rating_boost = settings.rating_boost_max * max(0.0, min(1.0, (product.rating - 3.5) / 1.5))
    return sorted(ordered, key=lambda c: (-c.final_score, c.product_id))


def diversify(candidates: list[ProductCandidate], catalog: Catalog, top_k: int,
              eligible_ids: set[str], max_per_brand: int | None = 2) -> list[ProductCandidate]:
    """Build the final list with two diversity rules:

    1. One slot per product family. Other ELIGIBLE family members are
       attached as variants (each keeps its own id, price, rating, reviews).
    2. At most ``max_per_brand`` products per brand (pass None to disable,
       e.g. when the user asked for one brand). If the brand cap would
       leave empty slots, they are filled with the best skipped products.
    """
    chosen: list[ProductCandidate] = []
    skipped_for_brand: list[ProductCandidate] = []
    families: set[str] = set()
    brand_counts: dict[str, int] = {}

    def take(candidate: ProductCandidate) -> None:
        product = catalog.products[candidate.product_id]
        families.add(product.family_id)
        brand_counts[product.brand] = brand_counts.get(product.brand, 0) + 1
        candidate.variant_ids = [pid for pid in catalog.family_members(candidate.product_id)
                                 if pid != candidate.product_id and pid in eligible_ids]
        chosen.append(candidate)

    for candidate in candidates:
        if len(chosen) == top_k:
            break
        product = catalog.products[candidate.product_id]
        if product.family_id in families:
            continue
        if max_per_brand is not None and brand_counts.get(product.brand, 0) >= max_per_brand:
            skipped_for_brand.append(candidate)
            continue
        take(candidate)

    for candidate in skipped_for_brand:          # back-fill, still one per family
        if len(chosen) == top_k:
            break
        if catalog.products[candidate.product_id].family_id not in families:
            take(candidate)
    return chosen
