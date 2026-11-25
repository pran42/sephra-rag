"""Rule-based (regex) query parser: the no-API-key fallback.

IMPORTANT LIMITATION: this is NOT equivalent to LLM query understanding.
It recognises a fixed vocabulary (see vocabulary.py) and common phrasings
of prices, ratings, skin types and attributes. It does not understand
paraphrases it has no rule for ("nothing too pricey"), negation beyond the
listed attributes, or conversational context. Anything it cannot map to a
hard constraint stays in ``semantic_query`` for semantic search, and its
interpretations are recorded in ``notes`` so the UI can show them.

When an OpenRouter LLM parser is added later, it will produce the same
``StructuredQuery`` and be validated by the same filter code.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from sephra.query.schema import CategoryFilter, Intent, SkinType, StructuredQuery
from sephra.query.vocabulary import (AMBIGUOUS_BRANDS, ATTRIBUTE_PATTERNS,
                                     BROAD_CATEGORY_PATTERNS, INGREDIENT_QUERY_PATTERNS,
                                     PRODUCT_TYPE_PATTERNS)

HIGHLY_RATED_THRESHOLD = 4.0
NUMBER = r"(\d+(?:\.\d+)?)"


def fold(text: str) -> str:
    """Lowercase, strip accents, normalise curly quotes."""
    text = text.replace("’", "'").replace("‘", "'")
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _blank(text: str, start: int, end: int) -> str:
    """Replace a matched span with spaces (keeps other offsets valid)."""
    return text[:start] + " " * (end - start) + text[end:]


# ---------------------------------------------------------------------------
# Individual extractors. Each returns its findings and the text with the
# matched phrase blanked out, so later extractors don't re-use it.
# ---------------------------------------------------------------------------

_MAX_PRICE_PATTERNS = [
    rf"(?:under|below|less than|cheaper than|up to|max(?:imum)?(?: of)?|no more than|at most|within|<=?)\s*\$\s*{NUMBER}",
    rf"(?:under|below|less than|cheaper than|up to|no more than|at most)\s*{NUMBER}\s*(?:dollars|usd|bucks)\b",
    rf"(?:under|below|less than|up to|no more than|at most)\s*{NUMBER}\b(?!\s*(?:stars?|\+|rating|\.\d))",
    rf"\$\s*{NUMBER}\s*(?:or less|or under|and under|max(?:imum)?)",
    rf"budget(?: of| is| around)?\s*\$?\s*{NUMBER}",
]
_MIN_PRICE_PATTERNS = [
    rf"(?:over|above|more than|at least|starting at|from)\s*\$\s*{NUMBER}",
]
_RANGE_PATTERN = rf"between\s*\$?\s*{NUMBER}\s*(?:and|-|to)\s*\$?\s*{NUMBER}"


def extract_price(text: str, notes: list[str]) -> tuple[float | None, float | None, str]:
    max_price = min_price = None
    match = re.search(_RANGE_PATTERN, text)
    if match:
        low, high = sorted((float(match.group(1)), float(match.group(2))))
        notes.append(f"Price range ${low:g}-${high:g} applied as a hard filter.")
        return low, high, _blank(text, *match.span())
    for pattern in _MAX_PRICE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            max_price = float(match.group(1))
            text = _blank(text, *match.span())
            break
    for pattern in _MIN_PRICE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            min_price = float(match.group(1))
            text = _blank(text, *match.span())
            break
    return min_price, max_price, text


_RATING_PATTERNS = [
    rf"(?:rated|rating(?: of)?|ratings? (?:of |above |over )?)\s*(?:at least|above|over|>=?|of)?\s*{NUMBER}\s*(?:\+|stars?|or (?:more|higher|above|better)|and (?:up|above))?",
    rf"{NUMBER}\s*\+?\s*(?:stars?|★)(?:\s*(?:or (?:more|higher|above|better)|and (?:up|above)|rating))?",
]
_HIGHLY_RATED = r"highly[- ]rated|top[- ]rated|best[- ]rated|well[- ]rated|highest[- ]rated|good ratings?|high ratings?"


def extract_rating(text: str, notes: list[str]) -> tuple[float | None, str]:
    for pattern in _RATING_PATTERNS:
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            if 0 <= value <= 5:
                return value, _blank(text, *match.span())
            notes.append(f"Ignored rating {value:g}: ratings in the dataset are 0-5.")
    match = re.search(_HIGHLY_RATED, text)
    if match:
        notes.append(f"Interpreted '{match.group(0)}' as rating >= {HIGHLY_RATED_THRESHOLD:g} "
                     "(a rule-based interpretation).")
        return HIGHLY_RATED_THRESHOLD, _blank(text, *match.span())
    return None, text


def extract_attributes(text: str) -> tuple[list[str], str]:
    found: list[str] = []
    for tag, pattern in ATTRIBUTE_PATTERNS.items():
        for match in re.finditer(rf"\b(?:{pattern})\b", text):
            if tag not in found:
                found.append(tag)
            text = _blank(text, *match.span())
    return found, text


def extract_ingredients(text: str) -> list[str]:
    found: list[str] = []
    for pattern, canonical in INGREDIENT_QUERY_PATTERNS.items():
        if re.search(rf"\b(?:{pattern})\b", text) and canonical not in found:
            found.append(canonical)
    # "retinol" is a subset of "retinoids"; keep the more specific request.
    if "retinol" in found and "retinoids" in found:
        found.remove("retinoids")
    return found


_SKIN_WORDS = {"dry": SkinType.DRY, "oily": SkinType.OILY, "combination": SkinType.COMBINATION,
               "combo": SkinType.COMBINATION, "normal": SkinType.NORMAL,
               "sensitive": SkinType.SENSITIVE}
_SKIN_PATTERN = (r"\b((?:(?:very |extra )?(?:dry|oily|combination|combo|normal|sensitive)"
                 r"(?:\s*(?:,|/|and|or|&)\s*)?)+)\s*skin\b")


def extract_skin_types(text: str) -> list[SkinType]:
    found: list[SkinType] = []
    for match in re.finditer(_SKIN_PATTERN, text):
        for word in re.findall(r"dry|oily|combination|combo|normal|sensitive", match.group(1)):
            skin = _SKIN_WORDS[word]
            if skin not in found:
                found.append(skin)
    return found


def extract_categories(text: str, notes: list[str]) -> list[CategoryFilter]:
    found: list[CategoryFilter] = []
    working = text
    for pattern, filters in PRODUCT_TYPE_PATTERNS:
        for match in re.finditer(rf"\b(?:{pattern})\b", working):
            for category in filters:
                if category not in found:
                    found.append(category)
            working = _blank(working, *match.span())
    if found:
        return found
    for pattern, filters in BROAD_CATEGORY_PATTERNS:
        if re.search(rf"\b(?:{pattern})\b", working):
            for category in filters:
                if category not in found:
                    found.append(category)
    if not found:
        notes.append("No product type detected; searching all categories.")
    return found


def _brand_aliases(brand: str) -> list[str]:
    """Generate matchable spellings: 'Rare Beauty by Selena Gomez' -> also 'rare beauty'."""
    base = " ".join(re.findall(r"[a-z0-9]+", fold(brand)))
    aliases = {base}
    for separator in (" by ", " since "):
        if separator in base:
            aliases.add(base.split(separator)[0].strip())
    for suffix in (" skincare", " skin care", " haircare", " cosmetics"):
        if base.endswith(suffix) and len(base[: -len(suffix)].split()) >= 2:
            aliases.add(base[: -len(suffix)].strip())
    return sorted((a for a in aliases if a), key=len, reverse=True)


def extract_brands(text: str, known_brands: Iterable[str], notes: list[str]) -> tuple[list[str], str]:
    """Find catalogue brand names in the query (longest alias first)."""
    tokens_text = " " + " ".join(re.findall(r"[a-z0-9]+", text)) + " "
    candidates: list[tuple[str, str]] = []
    for brand in known_brands:
        for alias in _brand_aliases(brand):
            candidates.append((alias, brand))
    candidates.sort(key=lambda pair: len(pair[0]), reverse=True)

    found: list[str] = []
    for alias, brand in candidates:
        if brand in found or f" {alias} " not in tokens_text:
            continue
        if alias in AMBIGUOUS_BRANDS or brand.lower() in AMBIGUOUS_BRANDS:
            context = rf"(?:from|by|brand) {re.escape(alias)}\b|\b{re.escape(alias)} (?:brand|products?)\b"
            if not re.search(context, tokens_text):
                continue
        found.append(brand)
        tokens_text = tokens_text.replace(f" {alias} ", " " * (len(alias) + 2))
    if found:
        notes.append("Brand filter: " + ", ".join(found))
    return found, tokens_text


_COMPARISON = r"\bcompare\b|\bcomparison\b|\bvs\.?\b|\bversus\b|\bdifference between\b|\bwhich is better\b"
_REVIEW_QUESTION = (r"what (?:do|did|are) (?:customers|people|reviewers|users|buyers)(?: say| think| saying)?"
                    r"(?: about| of| on)?|(?:customer )?reviews? (?:of|for|about|on)|"
                    r"customer (?:feedback|opinions?|experiences?)(?: on| of| for| about| with)?|"
                    r"how do (?:customers|people) like")


def detect_intent(text: str) -> tuple[Intent, list[str]]:
    """Return the intent and, for comparison/review questions, product-name segments."""
    if re.search(_COMPARISON, text):
        body = re.sub(r"^.*?(?:compare|comparison of|comparison between|difference between|which is better[,:]?)\s*",
                      "", text).strip(" ?.!")
        body = re.sub(r"^(?:the|these|between)\s+", "", body)
        parts = re.split(r"\s+(?:vs\.?|versus)\s+", body)
        if len(parts) < 2:
            parts = [body]   # split on "and"/"or" later, by the name resolver
        return Intent.COMPARISON, [p.strip() for p in parts if p.strip()]
    match = re.search(_REVIEW_QUESTION, text)
    if match:
        name = text[match.end():].strip(" ?.!")
        name = re.sub(r"^(?:the|this|that|a)\s+", "", name)
        return Intent.REVIEW_QUESTION, [name] if name else []
    return Intent.RECOMMENDATION, []


_FILLER = (r"^(?:please |can you |could you |i(?:'m| am) looking for |i need |i want |"
           r"show me |find me |find |recommend(?: me)? |suggest(?: me)? |give me |any )+")


def _clean_semantic(text: str, fallback: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" ,.?!")
    text = re.sub(_FILLER, "", text).strip(" ,.?!")
    text = re.sub(r"\b(?:that is|that are|which is|and|with)\s*$", "", text).strip(" ,.?!")
    return text or fallback


def parse_query(query: str, known_brands: Iterable[str] = ()) -> StructuredQuery:
    """Parse a natural-language query into a validated StructuredQuery."""
    if not query or not query.strip():
        raise ValueError("Query is empty")
    original = query.strip()
    text = fold(original)
    notes: list[str] = ["Parsed by the rule-based fallback parser (no LLM)."]

    intent, product_names = detect_intent(text)

    min_price, max_price, text = extract_price(text, notes)
    min_rating, text = extract_rating(text, notes)
    in_stock = bool(re.search(r"\bin[- ]stock\b|\bcurrently available\b|\bavailable now\b", text))
    if in_stock:
        text = re.sub(r"\bin[- ]stock\b|\bcurrently available\b|\bavailable now\b", " ", text)
    # Keep attribute words ("fragrance-free") in the semantic search text,
    # but blank them for category detection so "fragrance" is not read as
    # the Fragrance category.
    semantic_source = text
    attributes, text = extract_attributes(text)
    ingredients = extract_ingredients(text)
    skin_types = extract_skin_types(text)

    if intent == Intent.RECOMMENDATION:
        brands, _ = extract_brands(text, known_brands, notes)
        categories = extract_categories(text, notes)
    else:
        # For named products, the name resolver handles brand/category.
        brands, categories = [], []

    if skin_types:
        notes.append("Skin type(s) " + ", ".join(s.value for s in skin_types)
                     + " used as a ranking preference, not a hard filter.")

    semantic = _clean_semantic(semantic_source, fallback=original)
    if intent == Intent.REVIEW_QUESTION:
        semantic = "overall experience, results, texture, scent, pros and cons"

    return StructuredQuery(
        original_query=original,
        intent=intent,
        semantic_query=semantic,
        max_price=max_price,
        min_price=min_price,
        min_rating=min_rating,
        categories=categories,
        brands=brands,
        in_stock_only=in_stock,
        required_attributes=attributes,
        required_ingredients=ingredients,
        skin_types=skin_types,
        product_names=product_names,
        parser="rule_based",
        notes=notes,
    )
