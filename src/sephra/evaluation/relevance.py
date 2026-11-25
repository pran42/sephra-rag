"""Transparent, attribute-based relevance labels for the evaluation queries.

HOW THESE LABELS WERE CREATED (read before trusting the metrics)
-----------------------------------------------------------------
* No labels were produced by any retrieval system or LLM. Each query has
  a hand-written rubric below that grades EVERY product in the catalogue
  from verifiable dataset fields: category, price, rating, highlight tags,
  product name and ingredient list.
* Grades: 0 = not relevant, 1 = relevant, 2 = highly relevant.
* Because rubrics cover the whole catalogue, recall can be computed
  exactly. No pooling is needed.

KNOWN LIMITATIONS
* Claims like "not greasy", "everyday use" or "gentle" cannot be verified
  from structured data. Rubrics use the closest written signals (tags,
  words in the product name), so they measure "the listing claims X",
  not "the product actually is X".
* Rubrics read highlight tags and names, which BM25 also indexes, so they
  may slightly favour keyword retrieval. This is disclosed in the report.
* The rubrics were written by the developer (with AI assistance) and
  should be reviewed by a human. They are a transparent proxy, not
  ground truth from shoppers.
"""

from __future__ import annotations

from collections.abc import Callable

from sephra.catalog import Catalog, Product
from sephra.query.filters import has_attribute, has_ingredient
from sephra.query.vocabulary import SKIN_TYPE_TAGS

Rubric = Callable[[Product], int]


def _name(p: Product) -> str:
    return p.name.lower()


def _any(text: str, words: tuple[str, ...]) -> bool:
    return any(w in text for w in words)


def _is_face_moisturizer(p: Product) -> bool:
    return p.primary_category == "Skincare" and p.tertiary_category in {"Moisturizers", "Night Creams"}


def _skin_tag(p: Product, skin: str) -> bool:
    return bool(SKIN_TYPE_TAGS[skin] & set(p.highlights))


def q01(p: Product) -> int:  # Moisturizer for dry skin under $30, hydrating, not greasy
    if not (_is_face_moisturizer(p) and p.price_usd <= 30):
        return 0
    dry = _skin_tag(p, "dry") or "dry" in _name(p)
    hydrating = "Hydrating" in p.highlights or "Hyaluronic Acid" in p.highlights or "hydrat" in _name(p)
    return 2 if dry and hydrating else 1


def q02(p: Product) -> int:  # Fragrance-free skincare products
    return int(p.primary_category == "Skincare" and has_attribute(p, "Fragrance Free"))


def q03(p: Product) -> int:  # Lightweight moisturizer for oily skin
    if not _is_face_moisturizer(p):
        return 0
    oily = _skin_tag(p, "oily") or "oily" in _name(p)
    light = _any(_name(p), ("gel", "light", "water", "oil-free", "oil free", "matte", "fluid")) \
        or "Oil Free" in p.highlights
    return 2 if oily and light else int(oily or light)


def q04(p: Product) -> int:  # Sunscreen under $25
    return int(p.secondary_category == "Sunscreen" or p.tertiary_category == "Sunscreen") \
        if p.price_usd <= 25 else 0


def q05(p: Product) -> int:  # Highly rated shampoo for damaged hair
    if p.tertiary_category != "Shampoo" or p.rating is None or p.rating < 4.0:
        return 0
    damage = "Good for: Damage" in p.highlights or _any(_name(p), ("damage", "repair", "bond", "strength"))
    return 2 if damage else 1


def q06(p: Product) -> int:  # Perfume for everyday use
    return int(p.primary_category == "Fragrance"
               and p.tertiary_category in {"Perfume", "Rollerballs & Travel Size", "Cologne"})


def q07(p: Product) -> int:  # Skincare products from Drunk Elephant
    return int(p.brand == "Drunk Elephant" and p.primary_category == "Skincare")


def q08(p: Product) -> int:  # Hydrating oil-free gel moisturizer under $30
    if not (_is_face_moisturizer(p) and p.price_usd <= 30):
        return 0
    oil_free = has_attribute(p, "Oil Free")
    gel_or_hydrating = "gel" in _name(p) or "Hydrating" in p.highlights
    return 2 if oil_free and gel_or_hydrating else int(oil_free or gel_or_hydrating)


def q11(p: Product) -> int:  # Face serum under $15
    return int(p.tertiary_category == "Face Serums" and p.price_usd <= 15)


def q12(p: Product) -> int:  # Vitamin C serum for brightening
    if p.tertiary_category != "Face Serums" or not has_ingredient(p, "vitamin c"):
        return 0
    hero = "Vitamin C" in p.highlights or _any(_name(p), ("vitamin c", "vit c", "c-", " c "))
    return 2 if hero else 1


def q15(p: Product) -> int:  # Serum with hyaluronic acid
    if p.tertiary_category != "Face Serums" or not has_ingredient(p, "hyaluronic acid"):
        return 0
    return 2 if "Hyaluronic Acid" in p.highlights or "hyaluronic" in _name(p) else 1


def q16(p: Product) -> int:  # Vegan mascara
    return int(p.tertiary_category == "Mascara" and has_attribute(p, "Vegan"))


def q17(p: Product) -> int:  # Hair oil for frizz
    if p.tertiary_category != "Hair Oil":
        return 0
    return 2 if "Good for: Frizz" in p.highlights or "frizz" in _name(p) else 1


def q18(p: Product) -> int:  # Body lotion for very dry skin
    if p.tertiary_category != "Body Lotions & Body Oils":
        return 0
    return 2 if _skin_tag(p, "dry") or "dry" in _name(p) else 1


def q19(p: Product) -> int:  # Long-lasting matte lipstick
    if p.tertiary_category not in {"Lipstick", "Liquid Lipstick"}:
        return 0
    matte = "Matte Finish" in p.highlights or "matte" in _name(p)
    lasting = "Long-wearing" in p.highlights or _any(_name(p), ("long", "wear", "stay", "lasting", "transfer"))
    return matte + lasting


def q20(p: Product) -> int:  # Gentle cleanser for sensitive skin
    if p.tertiary_category not in {"Face Wash & Cleansers", "Face Wash"}:
        return 0
    gentle = _any(_name(p), ("gentle", "calm", "sooth", "sensitive", "mild", "barrier"))
    fragrance_free = has_attribute(p, "Fragrance Free")
    return gentle + fragrance_free


RUBRICS: dict[str, Rubric] = {
    "Q01": q01, "Q02": q02, "Q03": q03, "Q04": q04, "Q05": q05, "Q06": q06, "Q07": q07,
    "Q08": q08, "Q11": q11, "Q12": q12, "Q15": q15, "Q16": q16, "Q17": q17, "Q18": q18,
    "Q19": q19, "Q20": q20,
}

# Queries evaluated by a specific expected outcome instead of ranked relevance.
EXPECTED_NAMED_PRODUCTS: dict[str, list[str]] = {
    "Q09": ["P379709"],                 # Tata Harper Repairative Anti-Aging Moisturizer
    "Q14": ["P429637", "P379709"],      # La Mer Cool Gel Cream vs Tata Harper
}
EXPECTED_NO_RESULTS = {"Q10"}           # Tatcha has no moisturizer under $10
UNLABELLED = {"Q13"}                    # "I need something nice": deliberately ambiguous


def label_catalog(query_id: str, catalog: Catalog) -> dict[str, int]:
    """Grades for every product with grade > 0."""
    rubric = RUBRICS[query_id]
    return {pid: grade for pid, p in catalog.products.items() if (grade := rubric(p)) > 0}
