"""Evidence assembly: turn a SearchResponse into grounded context text.

This text is everything a future LLM is allowed to know when answering.
Each product block carries its product ID, and each review carries its
own review ID and the product ID it belongs to, so every claim in an
answer can be traced back to its source.
"""

from __future__ import annotations

from sephra.retrieval.pipeline import ProductResult, SearchResponse

MAX_INGREDIENT_CHARS = 400


def _fmt_rating(value: float | None) -> str:
    return f"{value:.2f}/5" if value is not None else "not available in dataset"


def product_block(result: ProductResult) -> str:
    p, e = result.product, result.evidence
    lines = [
        f"### PRODUCT {result.rank} [product_id={p.product_id}]",
        f"Name: {p.name}",
        f"Brand: {p.brand}",
        f"Category: {p.category_path}",
        f"Price (dataset): ${p.price_usd:.2f}"
        + (f" (sale price in dataset: ${p.sale_price_usd:.2f})" if p.sale_price_usd else ""),
        f"Rating (dataset): {_fmt_rating(p.rating)}",
        f"Availability (dataset snapshot): {'out of stock' if p.out_of_stock else 'in stock'}",
    ]
    if p.size:
        lines.append(f"Size: {p.size}")
    if p.highlights:
        lines.append("Listing highlights: " + ", ".join(p.highlights))
    if p.ingredients:
        text = p.ingredients[:MAX_INGREDIENT_CHARS]
        lines.append("Ingredients (truncated): " + text + ("..." if len(p.ingredients) > MAX_INGREDIENT_CHARS else ""))
    if result.candidate and result.candidate.matched_skin_tags:
        lines.append("Skin-type tags matching the request: " + ", ".join(result.candidate.matched_skin_tags))
    for variant in result.variants:
        lines.append(f"Related variant [product_id={variant.product_id}]: {variant.name}, "
                     f"${variant.price_usd:.2f}, rating {_fmt_rating(variant.rating)}, "
                     f"{variant.loaded_review_count} loaded reviews")

    lines.append(f"Reviews for this product in our loaded dataset: {e.loaded_review_count}")
    if e.stats:
        s = e.stats
        rate = f"{s.recommend_rate:.0%}" if s.recommend_rate is not None else "unknown"
        lines.append(f"Rating distribution of ALL {s.total} loaded reviews: {s.positive} positive (4-5 stars), "
                     f"{s.mixed} mixed (3), {s.critical} critical (1-2); recommended by {rate} "
                     f"of reviewers who answered.")
    if e.selected:
        lines.append(f"Selected review excerpts ({len(e.selected)} of {e.loaded_review_count}; "
                     "selected for relevance and balance, NOT a representative sample):")
        for s in e.selected:
            r = s.review
            skin = f", reviewer skin type: {r.skin_type}" if r.skin_type else ""
            lines.append(f"- [review_id={r.review_id}, product_id={r.product_id}, {r.rating} stars{skin}] "
                         f"{r.text}")
    for note in e.notes:
        lines.append(f"Note: {note}")
    return "\n".join(lines)


def build_context(response: SearchResponse) -> str:
    query = response.query
    header = [
        f"USER QUESTION: {query.original_query}",
        f"INTENT: {query.intent.value}",
        "HARD CONSTRAINTS (already enforced by code): "
        + ("; ".join(query.describe_constraints()) or "none"),
        "SOFT PREFERENCES: "
        + (", ".join(s.value + " skin" for s in query.skin_types) or "none")
        + f"; semantic request: {query.semantic_query}",
        f"RETRIEVAL STATUS: {response.status}. {response.message}",
        f"DATASET NOTE: {response.disclaimer}",
    ]
    blocks = [product_block(result) for result in response.products]
    return "\n".join(header) + "\n\n" + ("\n\n".join(blocks) if blocks else "(no products retrieved)")
