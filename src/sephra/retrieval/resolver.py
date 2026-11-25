"""Resolve product names typed by the user ("compare X and Y") to product IDs.

Matching uses word overlap (Jaccard) between the user's text and
"brand + product name". A resolution below ``MIN_MATCH_SCORE`` is reported
as not found instead of guessing a product.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sephra.catalog import Catalog
from sephra.query.rule_parser import fold

MIN_MATCH_SCORE = 0.5
_IGNORED = {"the", "a", "an", "by", "from", "s"}


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z0-9]+", fold(text)) if w not in _IGNORED)


@dataclass
class ResolvedName:
    text: str
    product_id: str | None
    score: float
    alternatives: list[tuple[str, float]] = field(default_factory=list)


_NAME_INDEX_CACHE: dict[int, list[tuple[str, frozenset[str], frozenset[str]]]] = {}


def _name_index(catalog: Catalog) -> list[tuple[str, frozenset[str], frozenset[str]]]:
    """Token sets for every product, computed once per catalogue object."""
    key = id(catalog)
    if key not in _NAME_INDEX_CACHE:
        _NAME_INDEX_CACHE[key] = [(pid, _tokens(f"{p.brand} {p.name}"), _tokens(p.name))
                                  for pid, p in catalog.products.items()]
    return _NAME_INDEX_CACHE[key]


def resolve_name(text: str, catalog: Catalog) -> ResolvedName:
    query_tokens = _tokens(text)
    if not query_tokens:
        return ResolvedName(text, None, 0.0)
    scored: list[tuple[str, float]] = []
    for pid, full_tokens, name_tokens in _name_index(catalog):
        # Users may or may not include the brand, so try both spellings.
        best = 0.0
        for candidate in (full_tokens, name_tokens):
            union = len(query_tokens | candidate)
            if union:
                best = max(best, len(query_tokens & candidate) / union)
        if best > 0:
            scored.append((pid, best))
    scored.sort(key=lambda pair: -pair[1])
    if not scored or scored[0][1] < MIN_MATCH_SCORE:
        return ResolvedName(text, None, scored[0][1] if scored else 0.0, scored[:3])
    return ResolvedName(text, scored[0][0], scored[0][1], scored[1:4])


def resolve_names(segments: list[str], catalog: Catalog) -> list[ResolvedName]:
    """Resolve each segment. A single segment like 'X and Y' (comparison
    without 'vs') is split at the 'and'/'or' that resolves both halves best."""
    if len(segments) == 1:
        text = segments[0]
        splits = [m.span() for m in re.finditer(r"\s+(?:and|or|&|with)\s+", text)]
        best_pair: list[ResolvedName] | None = None
        best_score = -1.0
        for start, end in splits:
            left, right = resolve_name(text[:start], catalog), resolve_name(text[end:], catalog)
            score = min(left.score, right.score)
            if left.product_id and right.product_id and score > best_score:
                best_pair, best_score = [left, right], score
        if best_pair:
            return best_pair
    return [resolve_name(segment, catalog) for segment in segments]
