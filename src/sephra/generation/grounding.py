"""Automatic grounding checks for generated answers.

These run on any answer text (from any LLM) and flag facts that do not
appear in the evidence. They catch the most damaging hallucinations
(invented prices, ratings, products), not every possible error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sephra.retrieval.pipeline import SearchResponse

_PRICE = re.compile(r"\$\s?(\d+(?:\.\d{1,2})?)")
_PRODUCT_ID = re.compile(r"\bP\d{5,7}\b")
_RATING = re.compile(r"\b([0-5](?:\.\d{1,4})?)\s*(?:/\s*5|out of 5|stars?)")


@dataclass
class GroundingReport:
    unknown_prices: list[str] = field(default_factory=list)
    unknown_product_ids: list[str] = field(default_factory=list)
    unknown_ratings: list[str] = field(default_factory=list)
    recommends_when_no_results: bool = False

    @property
    def is_grounded(self) -> bool:
        return not (self.unknown_prices or self.unknown_product_ids
                    or self.unknown_ratings or self.recommends_when_no_results)


def check_answer_grounding(answer: str, response: SearchResponse,
                           user_text: str = "") -> GroundingReport:
    products = [r.product for r in response.products]
    for result in response.products:
        products.extend(result.variants)

    allowed_ids = {p.product_id for p in products}
    allowed_prices = {round(p.price_usd, 2) for p in products}
    allowed_prices |= {round(p.sale_price_usd, 2) for p in products if p.sale_price_usd}
    # Prices the user typed (e.g. "under $30") may be repeated back.
    allowed_prices |= {round(float(x), 2) for x in _PRICE.findall(user_text or response.query.original_query)}
    for value in (response.query.max_price, response.query.min_price):
        if value is not None:
            allowed_prices.add(round(value, 2))

    # Whole stars 1-5 describe review buckets ("4-5 stars" = positive), which
    # the context defines, so they are always allowed. Decimal ratings must
    # match a product's dataset rating.
    allowed_ratings = {1.0, 2.0, 3.0, 4.0, 5.0}
    for p in products:
        if p.rating is not None:
            allowed_ratings |= {round(p.rating, 1), round(p.rating, 2), round(p.rating, 4)}
    for result in response.products:
        allowed_ratings |= {float(s.review.rating) for s in result.evidence.selected
                            if s.review.rating is not None}
    if response.query.min_rating is not None:
        allowed_ratings.add(round(response.query.min_rating, 2))

    report = GroundingReport()
    for raw in _PRICE.findall(answer):
        if round(float(raw), 2) not in allowed_prices:
            report.unknown_prices.append(f"${raw}")
    for pid in _PRODUCT_ID.findall(answer):
        if pid not in allowed_ids:
            report.unknown_product_ids.append(pid)
    for raw in _RATING.findall(answer):
        value = float(raw)
        if not any(abs(value - allowed) < 0.005 for allowed in allowed_ratings):
            report.unknown_ratings.append(raw)
    if response.status != "ok" and _PRODUCT_ID.search(answer):
        report.recommends_when_no_results = True
    return report
