"""The end-to-end retrieval pipeline (no LLM required).

    query text
      -> parse (rule-based today, OpenRouter LLM later)  -> StructuredQuery
      -> validate against real catalogue values
      -> route by intent:
           recommendation: hard filters -> dense + BM25 -> RRF -> [rerank]
                           -> boosts -> variant diversity -> final constraint check
           comparison / review question: resolve named products
      -> product-specific review evidence for each product
      -> SearchResponse (structured, ready for the UI or an LLM prompt)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from sephra.catalog import Catalog, Product, get_catalog
from sephra.config import DEFAULT_SETTINGS, RetrievalSettings
from sephra.embeddings import embed_query
from sephra.query.filters import FilterResult, apply_hard_filters, product_violations, validate_query
from sephra.query.rule_parser import parse_query
from sephra.query.schema import Intent, StructuredQuery
from sephra.retrieval.ranking import (ProductCandidate, RetrievalMode, diversify,
                                      generate_candidates)
from sephra.retrieval.resolver import ResolvedName, resolve_names
from sephra.retrieval.reviews import ReviewEvidence, get_review_evidence

logger = logging.getLogger(__name__)

Status = Literal["ok", "no_results", "invalid_query", "product_not_found"]
QueryParser = Callable[[str, set[str]], StructuredQuery]

DATASET_DISCLAIMER = (
    "Prices, ratings, availability and reviews come from a static public dataset snapshot "
    "(Kaggle: Sephora Products and Skincare Reviews) and may not reflect current information. "
    "Sephra is not affiliated with Sephora."
)


@dataclass
class ProductResult:
    rank: int
    product: Product
    candidate: ProductCandidate | None
    variants: list[Product]
    evidence: ReviewEvidence


@dataclass
class SearchResponse:
    query: StructuredQuery
    status: Status
    message: str
    mode: str
    products: list[ProductResult] = field(default_factory=list)
    filter_summary: str = ""
    resolved_names: list[ResolvedName] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    disclaimer: str = DATASET_DISCLAIMER


def _default_parser(text: str, brands: set[str]) -> StructuredQuery:
    return parse_query(text, known_brands=brands)


class SephraRetriever:
    """Holds the shared catalogue and runs searches. Create once, reuse."""

    def __init__(self, catalog: Catalog | None = None,
                 settings: RetrievalSettings = DEFAULT_SETTINGS,
                 parser: QueryParser | None = None) -> None:
        self.catalog = catalog or get_catalog()
        self.settings = settings
        self.parser = parser or _default_parser
        self.known_brands = {p.brand for p in self.catalog.products.values()}

    # ------------------------------------------------------------------ API

    def search(self, text: str, mode: RetrievalMode = "hybrid", top_k: int | None = None,
               reranker: str | None = None) -> SearchResponse:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        query = self.parser(text, self.known_brands)
        timings["parse"] = time.perf_counter() - start
        response = self.search_structured(query, mode=mode, top_k=top_k, reranker=reranker)
        response.timings = {**timings, **response.timings}
        return response

    def search_structured(self, query: StructuredQuery, mode: RetrievalMode = "hybrid",
                          top_k: int | None = None, reranker: str | None = None) -> SearchResponse:
        top_k = top_k or self.settings.top_k
        mode_label = f"{mode}({reranker or 'minilm'})" if mode == "hybrid_rerank" else mode
        errors = validate_query(query, self.catalog)
        if errors:
            return SearchResponse(query, "invalid_query",
                                  "The query contains values that do not exist in the dataset: "
                                  + "; ".join(errors), mode_label)
        if query.intent in (Intent.COMPARISON, Intent.REVIEW_QUESTION):
            return self._named_products(query, mode_label)
        return self._recommend(query, mode, mode_label, top_k, reranker)

    def review_evidence_for(self, product_id: str, question: str = "",
                            max_reviews: int = 6) -> ReviewEvidence:
        """Evidence for one known product (used by UI follow-ups)."""
        text = question or "overall experience, results, texture, scent, pros and cons"
        return get_review_evidence(product_id, embed_query(text), self.catalog,
                                   self.settings, max_reviews=max_reviews)

    # ------------------------------------------------------- recommendations

    def _recommend(self, query: StructuredQuery, mode: RetrievalMode, mode_label: str,
                   top_k: int, reranker: str | None) -> SearchResponse:
        timings: dict[str, float] = {}
        t = time.perf_counter()
        filters: FilterResult = apply_hard_filters(query, self.catalog)
        timings["filters"] = time.perf_counter() - t

        if filters.is_empty:
            failing = next((label for label, remaining in filters.steps if remaining == 0), "")
            message = ("No products in the dataset satisfy all of your requirements "
                       f"({'; '.join(query.describe_constraints())}). "
                       f"The constraint that removed the last products was: {failing}. "
                       "Sephra does not relax your requirements automatically; "
                       "try changing one of them.")
            return SearchResponse(query, "no_results", message, mode_label,
                                  filter_summary=filters.explain(), timings=timings)

        t = time.perf_counter()
        embedding = embed_query(query.semantic_query)
        timings["embed"] = time.perf_counter() - t

        t = time.perf_counter()
        candidates = generate_candidates(query, filters, embedding, self.catalog,
                                         self.settings, mode, reranker)
        timings["retrieve_rank"] = time.perf_counter() - t

        # Final safety net: re-check every candidate against the hard constraints.
        safe: list[ProductCandidate] = []
        warnings: list[str] = []
        for candidate in candidates:
            violations = product_violations(self.catalog.products[candidate.product_id], query)
            if violations:
                logger.error("Dropped %s: violates %s", candidate.product_id, violations)
                warnings.append(f"Internal check removed {candidate.product_id} ({violations}).")
            else:
                safe.append(candidate)

        # A brand cap makes no sense when the user asked for specific brands.
        brand_cap = None if query.brands else self.settings.max_per_brand
        chosen = diversify(safe, self.catalog, top_k, filters.eligible_ids, brand_cap)

        t = time.perf_counter()
        results = [self._build_result(rank, c.product_id, c, embedding, query)
                   for rank, c in enumerate(chosen, start=1)]
        timings["reviews"] = time.perf_counter() - t

        status: Status = "ok" if results else "no_results"
        message = (f"Found {len(results)} product(s) from {len(filters.eligible_ids)} "
                   "eligible products in the dataset." if results
                   else "Eligible products exist but none could be retrieved.")
        return SearchResponse(query, status, message, mode_label, results,
                              filter_summary=filters.explain(), timings=timings,
                              warnings=warnings)

    # -------------------------------------------- comparison / review question

    def _named_products(self, query: StructuredQuery, mode_label: str) -> SearchResponse:
        timings: dict[str, float] = {}
        t = time.perf_counter()
        resolved = resolve_names(query.product_names, self.catalog) if query.product_names else []
        timings["resolve"] = time.perf_counter() - t
        found = [r for r in resolved if r.product_id]
        missing = [r for r in resolved if not r.product_id]
        if not found or missing:
            names = ", ".join(f"'{r.text}'" for r in (missing or resolved)) or "(no product name given)"
            return SearchResponse(query, "product_not_found",
                                  f"Could not confidently identify these product(s) in the dataset: "
                                  f"{names}. Please use the product's full name.",
                                  mode_label, resolved_names=resolved, timings=timings)

        question = query.semantic_query
        embedding = embed_query(question)
        max_reviews = 6 if query.intent == Intent.REVIEW_QUESTION else self.settings.reviews_per_product
        t = time.perf_counter()
        results = []
        for rank, item in enumerate(found, start=1):
            product = self.catalog.products[item.product_id]
            evidence = get_review_evidence(product.product_id, embedding, self.catalog,
                                           self.settings, query.skin_types, max_reviews)
            variants = [self.catalog.products[p] for p in self.catalog.family_members(product.product_id)
                        if p != product.product_id]
            results.append(ProductResult(rank, product, None, variants, evidence))
        timings["reviews"] = time.perf_counter() - t
        what = "Comparison of" if query.intent == Intent.COMPARISON else "Customer reviews for"
        return SearchResponse(query, "ok", f"{what}: " + "; ".join(p.product.name for p in results),
                              mode_label, results, resolved_names=resolved, timings=timings)

    # ---------------------------------------------------------------- helpers

    def _build_result(self, rank: int, product_id: str, candidate: ProductCandidate | None,
                      embedding: list[float], query: StructuredQuery) -> ProductResult:
        product = self.catalog.products[product_id]
        evidence = get_review_evidence(product_id, embedding, self.catalog, self.settings,
                                       query.skin_types)
        variant_ids = candidate.variant_ids if candidate else []
        return ProductResult(rank, product, candidate,
                             [self.catalog.products[v] for v in variant_ids], evidence)
