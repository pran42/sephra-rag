"""Product-family (variant) grouping.

The dataset contains size variants and close relatives with DIFFERENT
product IDs, prices, ratings and review sets, e.g.

    P411403  Confidence in a Cream Anti-Aging Hydrating Moisturizer   $20  4.62   0 reviews
    P427641  Mini Confidence in a Cream Hydrating Moisturizer         $20  3.68  47 reviews

We never merge their data. We only assign a shared ``family_id`` so the
ranking can avoid filling the top-5 with near-duplicates. Each member
keeps its own id, price, rating and reviews.

Grouping rule (within the SAME brand only):
1. Identical review texts: the dataset attaches the same reviews to size
   variants (all 156 Algae + Moringa reviews appear under two IDs). If at
   least half of the smaller product's reviews are shared, group them.
2. Same category AND similar names (Jaccard overlap of name words,
   ignoring size words):
   - size variants ("mini", "travel", ...): the shorter name's words must
     all appear in the longer name and overlap >= 0.60
   - otherwise overlap >= 0.85 (strict, so different scents such as
     "Good Girl" vs "Good Girl Blush" are not grouped).
   The category check stops chains such as
   "Mini Invisible Oil" -> "Invisible Oil Shampoo".

This is a heuristic. It is deliberately conservative, and the UI labels
grouped products as "related variants" rather than "the same product".
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sephra.catalog import Catalog

SIZE_MARKERS = {"mini", "travel", "jumbo", "refill", "deluxe", "supersize", "value"}
IGNORED_WORDS = SIZE_MARKERS | {"size", "oz", "ml", "spray", "a", "an", "the", "and",
                                "with", "for", "of", "in", "to"}

NAME_THRESHOLD_WITH_SIZE_MARKER = 0.60
NAME_THRESHOLD_DEFAULT = 0.85
SHARED_REVIEW_THRESHOLD = 0.5


def _fold(text: str) -> str:
    """Lowercase and strip accents (e.g. 'Lancôme' -> 'lancome')."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def name_tokens(name: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", _fold(name)) if w not in IGNORED_WORDS}


def has_size_marker(name: str) -> bool:
    return bool(set(re.findall(r"[a-z]+", _fold(name))) & SIZE_MARKERS)


def name_similarity(name_a: str, name_b: str) -> float:
    a, b = name_tokens(name_a), name_tokens(name_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def are_name_variants(name_a: str, name_b: str) -> bool:
    """Name-only test (category compatibility is checked separately)."""
    if has_size_marker(name_a) or has_size_marker(name_b):
        # Size variant: the shorter name's words must all appear in the
        # longer name, e.g. "Mini Confidence in a Cream Hydrating Moisturizer"
        # vs "Confidence in a Cream Anti-Aging Hydrating Moisturizer".
        a, b = name_tokens(name_a), name_tokens(name_b)
        shorter, longer = sorted((a, b), key=len)
        return shorter <= longer and name_similarity(name_a, name_b) >= NAME_THRESHOLD_WITH_SIZE_MARKER
    return name_similarity(name_a, name_b) >= NAME_THRESHOLD_DEFAULT


def categories_compatible(a, b) -> bool:
    """Variants must be the same kind of product.

    Mini products sometimes live under primary category "Mini Size" with
    the real category as their secondary (e.g. Mini Size > Hair).
    """
    if a.primary_category == "Mini Size" and b.primary_category == "Mini Size":
        return a.secondary_category == b.secondary_category
    if a.primary_category == "Mini Size":
        return a.secondary_category == b.primary_category
    if b.primary_category == "Mini Size":
        return b.secondary_category == a.primary_category
    return (a.primary_category, a.secondary_category, a.tertiary_category) == \
           (b.primary_category, b.secondary_category, b.tertiary_category)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # keep the alphabetically smaller id as root -> deterministic ids
            self.parent[max(ra, rb)] = min(ra, rb)


def build_families(catalog: "Catalog") -> dict[str, list[str]]:
    """Assign ``family_id`` to every product and return family -> members."""
    uf = _UnionFind()
    by_brand: dict[str, list[str]] = defaultdict(list)
    for product in catalog.products.values():
        uf.find(product.product_id)
        by_brand[product.brand].append(product.product_id)

    # Rule 1: shared review texts.
    shared_counts: dict[tuple[str, str], int] = defaultdict(int)
    for review in catalog.reviews.values():
        for other in review.shared_product_ids:
            if review.product_id < other:
                shared_counts[(review.product_id, other)] += 1
    for (a, b), shared in shared_counts.items():
        pa, pb = catalog.products.get(a), catalog.products.get(b)
        if pa is None or pb is None or pa.brand != pb.brand:
            continue
        smaller = min(pa.loaded_review_count, pb.loaded_review_count) or 1
        if shared / smaller >= SHARED_REVIEW_THRESHOLD:
            uf.union(a, b)

    # Rule 2: name similarity within a brand (about 240k comparisons, fast).
    for ids in by_brand.values():
        for i, a in enumerate(ids):
            product_a = catalog.products[a]
            for b in ids[i + 1:]:
                product_b = catalog.products[b]
                if not categories_compatible(product_a, product_b):
                    continue
                if "Mini Size" in (product_a.primary_category, product_b.primary_category):
                    # "Mini Size > Hair" is too coarse to confirm the product
                    # type, so require identical names apart from size words.
                    same = name_tokens(product_a.name) == name_tokens(product_b.name)
                else:
                    same = are_name_variants(product_a.name, product_b.name)
                if same:
                    uf.union(a, b)

    families: dict[str, list[str]] = defaultdict(list)
    for product in catalog.products.values():
        root = uf.find(product.product_id)
        product.family_id = f"family_{root}"
        families[product.family_id].append(product.product_id)
    return dict(families)
