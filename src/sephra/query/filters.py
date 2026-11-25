"""Deterministic hard-constraint filtering.

Hard constraints are enforced in Python code, never by asking a model to
"respect the budget". There is ONE definition of "satisfies the query"
(``product_violations``), used for:
  1. computing the eligible product set before search,
  2. the final safety check on every returned product,
  3. the automatic hard-filter checks in evaluation.

Two layers are applied to ChromaDB:
  * native metadata filters (``build_metadata_where``) for fields stored in
    Chroma (price, rating, primary/secondary category, brand, stock);
  * an explicit ``product_id $in [...]`` list when a constraint needs data
    Chroma does not store (tertiary category, highlight tags, ingredients).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from sephra.catalog import Catalog, Product
from sephra.query.schema import CategoryFilter, StructuredQuery
from sephra.query.vocabulary import ATTRIBUTE_PATTERNS, INGREDIENT_SYNONYMS

logger = logging.getLogger(__name__)

PRICE_EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Single-constraint checks
# ---------------------------------------------------------------------------

def matches_category(product: Product, category: CategoryFilter) -> bool:
    return ((category.primary is None or product.primary_category == category.primary)
            and (category.secondary is None or product.secondary_category == category.secondary)
            and (category.tertiary is None or product.tertiary_category == category.tertiary))


def has_attribute(product: Product, attribute: str) -> bool:
    """Tag present in highlights, OR the product name states it."""
    if attribute in product.highlights:
        return True
    pattern = ATTRIBUTE_PATTERNS.get(attribute)
    return bool(pattern and re.search(rf"\b(?:{pattern})\b", product.name.lower()))


def has_ingredient(product: Product, ingredient: str) -> bool:
    """Ingredient list, highlight tags or product name mention a known spelling.

    Products with no ingredient list cannot be verified and do not match.
    """
    spellings = INGREDIENT_SYNONYMS.get(ingredient.lower(), [ingredient.lower()])
    haystack = " ".join([(product.ingredients or ""), " ".join(product.highlights),
                         product.name]).lower()
    return any(spelling in haystack for spelling in spellings)


def _constraint_checks(query: StructuredQuery) -> list[tuple[str, Callable[[Product], bool]]]:
    """The hard constraints of a query as (label, check) pairs, in filter order."""
    checks: list[tuple[str, Callable[[Product], bool]]] = []
    if query.categories:
        label = "category: " + " OR ".join(c.label() for c in query.categories)
        checks.append((label, lambda p: any(matches_category(p, c) for c in query.categories)))
    if query.brands:
        brands = set(query.brands)
        checks.append(("brand: " + " OR ".join(query.brands), lambda p: p.brand in brands))
    if query.max_price is not None:
        checks.append((f"price <= ${query.max_price:g}",
                       lambda p: p.price_usd <= query.max_price + PRICE_EPSILON))
    if query.min_price is not None:
        checks.append((f"price >= ${query.min_price:g}",
                       lambda p: p.price_usd >= query.min_price - PRICE_EPSILON))
    if query.min_rating is not None:
        # Products with no rating cannot be shown to meet a rating threshold.
        checks.append((f"rating >= {query.min_rating:g}",
                       lambda p: p.rating is not None and p.rating >= query.min_rating))
    if query.in_stock_only:
        checks.append(("in stock", lambda p: not p.out_of_stock))
    for attribute in query.required_attributes:
        checks.append((f"attribute: {attribute}", lambda p, a=attribute: has_attribute(p, a)))
    for ingredient in query.required_ingredients:
        checks.append((f"ingredient: {ingredient}", lambda p, i=ingredient: has_ingredient(p, i)))
    return checks


def product_violations(product: Product, query: StructuredQuery) -> list[str]:
    """Return the hard constraints this product violates (empty = eligible)."""
    return [label for label, check in _constraint_checks(query) if not check(product)]


# ---------------------------------------------------------------------------
# Validation of the structured query against real catalogue values
# ---------------------------------------------------------------------------

def validate_query(query: StructuredQuery, catalog: Catalog) -> list[str]:
    """Reject values that do not exist in the dataset (e.g. an invented
    category from an LLM) instead of silently ignoring them."""
    errors: list[str] = []
    products = catalog.products.values()
    primaries = {p.primary_category for p in products}
    secondaries = {p.secondary_category for p in products if p.secondary_category}
    tertiaries = {p.tertiary_category for p in products if p.tertiary_category}
    brands = {p.brand for p in products}
    tags = {tag for p in products for tag in p.highlights}

    for category in query.categories:
        if category.primary and category.primary not in primaries:
            errors.append(f"Unknown primary category: {category.primary!r}")
        if category.secondary and category.secondary not in secondaries:
            errors.append(f"Unknown secondary category: {category.secondary!r}")
        if category.tertiary and category.tertiary not in tertiaries:
            errors.append(f"Unknown tertiary category: {category.tertiary!r}")
    for brand in query.brands:
        if brand not in brands:
            errors.append(f"Unknown brand: {brand!r}")
    for attribute in query.required_attributes:
        if attribute not in tags:
            errors.append(f"Unknown attribute tag: {attribute!r}")
    return errors


# ---------------------------------------------------------------------------
# Applying the filters
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    eligible_ids: set[str]
    steps: list[tuple[str, int]] = field(default_factory=list)  # (constraint, products remaining)
    chroma_where: dict[str, Any] | None = None
    total_products: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.eligible_ids

    @property
    def is_unfiltered(self) -> bool:
        return len(self.eligible_ids) == self.total_products

    def explain(self) -> str:
        """e.g. 'all products: 8494 -> category: ...: 403 -> price <= $30: 81'"""
        parts = [f"all products: {self.total_products}"]
        parts += [f"{label}: {remaining}" for label, remaining in self.steps]
        return " -> ".join(parts)


def apply_hard_filters(query: StructuredQuery, catalog: Catalog) -> FilterResult:
    """Compute the eligible product set, recording how many survive each step."""
    remaining = list(catalog.products.values())
    steps: list[tuple[str, int]] = []
    for label, check in _constraint_checks(query):
        remaining = [p for p in remaining if check(p)]
        steps.append((label, len(remaining)))

    eligible = {p.product_id for p in remaining}
    result = FilterResult(eligible_ids=eligible, steps=steps,
                          total_products=len(catalog.products))
    result.chroma_where = build_chroma_where(query, eligible, len(catalog.products))
    logger.info("Hard filters: %s", result.explain())
    return result


def _and(conditions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Chroma requires $and to have at least two conditions."""
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def _or(conditions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$or": conditions}


def build_metadata_where(query: StructuredQuery) -> dict[str, Any] | None:
    """Native Chroma filter for the constraints Chroma metadata can express.

    Example: "moisturizer under $30 rated 4+" ->
      {"$and": [{"price_usd": {"$lte": 30}}, {"rating": {"$gte": 4}}, ...]}
    """
    conditions: list[dict[str, Any]] = []
    if query.max_price is not None:
        conditions.append({"price_usd": {"$lte": query.max_price}})
    if query.min_price is not None:
        conditions.append({"price_usd": {"$gte": query.min_price}})
    if query.min_rating is not None:
        conditions.append({"rating": {"$gte": query.min_rating}})
    if query.brands:
        conditions.append({"brand_name": {"$in": list(query.brands)}})
    if query.in_stock_only:
        conditions.append({"out_of_stock": 0})
    if query.categories:
        category_options = []
        for category in query.categories:
            parts = []
            if category.primary:
                parts.append({"primary_category": category.primary})
            if category.secondary:
                parts.append({"secondary_category": category.secondary})
            category_options.append(_and(parts))
        if all(category_options):
            conditions.append(_or(category_options))
    return _and(conditions)


def needs_id_filter(query: StructuredQuery) -> bool:
    """True if some constraint uses data that is not in Chroma metadata."""
    return bool(query.required_attributes or query.required_ingredients
                or any(c.tertiary for c in query.categories))


def build_chroma_where(query: StructuredQuery, eligible_ids: set[str],
                       total_products: int) -> dict[str, Any] | None:
    if len(eligible_ids) == total_products:
        return None                                   # nothing to filter
    if needs_id_filter(query):
        return {"product_id": {"$in": sorted(eligible_ids)}}
    return build_metadata_where(query)
