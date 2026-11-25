"""Product-specific review retrieval and balanced evidence selection.

Rules this module guarantees:
1. Reviews are ONLY searched with a mandatory ``product_id`` filter, and
   every returned review is re-checked against the catalogue to confirm it
   belongs to that product.
2. A product with zero loaded reviews gets an explicit "no reviews in our
   loaded dataset" note. It is never dropped and never given invented reviews.
3. Selection is balanced but not forced: the most query-relevant review
   from each sentiment bucket (positive 4-5 stars, mixed 3, critical 1-2)
   is included IF that bucket exists for the product; remaining slots go
   to the most relevant reviews regardless of sentiment.
4. Selected excerpts are always shown next to the rating distribution
   of ALL the product's loaded reviews, so a handful of excerpts is never
   presented as representative.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sephra.catalog import Catalog, Review
from sephra.config import RetrievalSettings
from sephra.query.schema import SkinType
from sephra.retrieval.dense import DenseHit, _dedupe_hits, _query
from sephra.vector_store import get_review_collection

logger = logging.getLogger(__name__)

SENTIMENTS = ("positive", "mixed", "critical")
SKIN_MATCH_BONUS = 0.02   # tiny tie-breaker for reviewers with the requested skin type
_RATING_FILTERS = {"positive": {"$gte": 4}, "mixed": {"$eq": 3}, "critical": {"$lte": 2}}


@dataclass
class SelectedReview:
    review: Review
    similarity: float
    reason: str                    # why it was selected
    reviewer_skin_match: bool      # reviewer's skin type matches the user's


@dataclass
class ReviewStats:
    """Statistics over ALL loaded reviews of one product (not just selected ones)."""

    total: int
    positive: int
    mixed: int
    critical: int
    recommend_yes: int
    recommend_known: int
    skin_types: dict[str, int]

    @property
    def recommend_rate(self) -> float | None:
        return self.recommend_yes / self.recommend_known if self.recommend_known else None


@dataclass
class ReviewEvidence:
    product_id: str
    loaded_review_count: int
    sephora_review_count: int | None
    stats: ReviewStats | None
    selected: list[SelectedReview] = field(default_factory=list)
    candidates_considered: int = 0
    notes: list[str] = field(default_factory=list)


def compute_review_stats(product_id: str, catalog: Catalog) -> ReviewStats | None:
    review_ids = catalog.reviews_by_product.get(product_id, [])
    if not review_ids:
        return None
    reviews = [catalog.reviews[rid] for rid in review_ids]
    skin_counts: dict[str, int] = {}
    for review in reviews:
        if review.skin_type:
            skin_counts[review.skin_type] = skin_counts.get(review.skin_type, 0) + 1
    known = [r for r in reviews if r.is_recommended is not None]
    return ReviewStats(
        total=len(reviews),
        positive=sum(r.sentiment == "positive" for r in reviews),
        mixed=sum(r.sentiment == "mixed" for r in reviews),
        critical=sum(r.sentiment == "critical" for r in reviews),
        recommend_yes=sum(1 for r in known if r.is_recommended),
        recommend_known=len(known),
        skin_types=dict(sorted(skin_counts.items(), key=lambda kv: -kv[1])),
    )


def _search_reviews(embedding: list[float], product_id: str, n_chunks: int,
                    sentiment: str | None = None) -> list[DenseHit]:
    where: dict = {"product_id": product_id}
    if sentiment:
        where = {"$and": [where, {"rating": _RATING_FILTERS[sentiment]}]}
    result = _query(get_review_collection(), embedding, n_chunks, where)
    if not result["ids"] or not result["ids"][0]:
        return []
    return _dedupe_hits(result["ids"][0], result["metadatas"][0], result["distances"][0],
                        key_field="parent_id")


def get_review_evidence(product_id: str, embedding: list[float], catalog: Catalog,
                        settings: RetrievalSettings, skin_types: list[SkinType] | None = None,
                        max_reviews: int | None = None) -> ReviewEvidence:
    product = catalog.get_product(product_id)
    if product is None:
        raise KeyError(f"Unknown product_id {product_id}")
    max_reviews = max_reviews or settings.reviews_per_product
    evidence = ReviewEvidence(product_id=product_id,
                              loaded_review_count=product.loaded_review_count,
                              sephora_review_count=product.sephora_review_count,
                              stats=compute_review_stats(product_id, catalog))

    if product.loaded_review_count == 0:
        note = "No customer reviews for this product in our loaded dataset partition."
        if product.sephora_review_count:
            note += (f" (The dataset's product record lists {product.sephora_review_count} reviews "
                     "on Sephora when it was collected; those are in review partitions we did not load.)")
        evidence.notes.append(note)
        return evidence

    wanted_skin = {s.value for s in (skin_types or [])}

    def to_candidates(hits: list[DenseHit]) -> dict[str, tuple[Review, float, bool]]:
        found: dict[str, tuple[Review, float, bool]] = {}
        for hit in hits:
            review = catalog.reviews.get(hit.item_id)
            if review is None:
                logger.warning("Review %s missing from catalogue; skipped", hit.item_id)
                continue
            if review.product_id != product_id:   # defence in depth: never mis-attribute
                logger.error("Review %s belongs to %s, not %s; dropped",
                             hit.item_id, review.product_id, product_id)
                continue
            skin_match = bool(wanted_skin and review.skin_type in wanted_skin)
            found[review.review_id] = (review, hit.similarity, skin_match)
        return found

    candidates = to_candidates(_search_reviews(embedding, product_id,
                                               settings.review_chunk_candidates))

    # If a sentiment bucket exists for this product but no review from it was
    # among the most relevant candidates, run one targeted search for it.
    stats = evidence.stats
    for sentiment in SENTIMENTS:
        exists = stats is not None and getattr(stats, sentiment) > 0
        present = any(c[0].sentiment == sentiment for c in candidates.values())
        if exists and not present:
            candidates.update(to_candidates(_search_reviews(embedding, product_id, 5, sentiment)))

    evidence.candidates_considered = len(candidates)

    def score(item: tuple[Review, float, bool]) -> float:
        return item[1] + (SKIN_MATCH_BONUS if item[2] else 0.0)

    ranked = sorted(candidates.values(), key=score, reverse=True)
    chosen: dict[str, SelectedReview] = {}
    for sentiment in SENTIMENTS:
        for review, similarity, skin_match in ranked:
            if review.sentiment == sentiment:
                chosen[review.review_id] = SelectedReview(
                    review, similarity, f"most relevant {sentiment} review", skin_match)
                break
        if len(chosen) >= max_reviews:
            break
    for review, similarity, skin_match in ranked:
        if len(chosen) >= max_reviews:
            break
        if review.review_id not in chosen:
            chosen[review.review_id] = SelectedReview(review, similarity,
                                                      "highly relevant to the query", skin_match)

    evidence.selected = sorted(chosen.values(), key=lambda s: -s.similarity)
    missing = [s for s in SENTIMENTS if stats is not None and getattr(stats, s) == 0]
    if missing:
        evidence.notes.append("This product has no " + " or ".join(missing)
                              + " reviews in our loaded dataset.")
    if any(s.review.shared_product_ids for s in evidence.selected):
        evidence.notes.append("Some reviews are attached to more than one product ID in the "
                              "dataset (typically size variants of the same product).")
    return evidence
