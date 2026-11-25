"""LLM-based query understanding, with deterministic guards and a fallback.

Flow for each query:
  1. Ask the LLM for JSON matching a strict schema. Allowed values for
     categories, attributes, ingredients and skin types come from the real
     catalogue and vocabulary, and are listed in the prompt.
  2. Validate the reply with Pydantic (StructuredQuery).
  3. Apply guards the LLM cannot override:
       - a price or rating number must actually appear in the user's text
         (except the documented "highly rated" -> 4.0 interpretation);
       - a brand must actually be mentioned in the user's text;
       - every category must exist and match at least one product.
     Values that fail a guard are removed, with a note. Invalid categories
     trigger one retry that tells the model what was wrong.
  4. On any failure (no key, rate limit, bad JSON, still invalid), fall back
     to the rule-based parser and say so in ``notes``.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter

from pydantic import ValidationError

from sephra.catalog import Catalog
from sephra.generation.llm_client import LLMClient, LLMError
from sephra.query.filters import matches_category, validate_query
from sephra.query.rule_parser import HIGHLY_RATED_THRESHOLD, _brand_aliases, fold, parse_query
from sephra.query.schema import Intent, SkinType, StructuredQuery
from sephra.query.vocabulary import ATTRIBUTE_PATTERNS, INGREDIENT_SYNONYMS

logger = logging.getLogger(__name__)

_NULLABLE_STR = {"type": ["string", "null"]}

# Words that must appear in the request before an LLM-proposed hard
# requirement is accepted (broad on purpose; see _guard_evidence).
ATTRIBUTE_EVIDENCE: dict[str, str] = {
    "Fragrance Free": r"fragrance|perfume|scent|smell",
    "Vegan": r"vegan",
    "Cruelty-Free": r"cruelty",
    "Without Parabens": r"paraben",
    "Without Sulfates SLS & SLES": r"sulfate|sulphate|\bsls\b",
    "Without Silicones": r"silicone",
    # "oily skin" is NOT a request for oil-free products (bug found in the
    # live evaluation: the model added Oil Free for "moisturizer for oily skin").
    "Oil Free": r"oil[- ]?free|without oils?|no oils?\b",
    "Alcohol Free": r"alcohol",
    "Gluten Free": r"gluten",
    "Clean at Sephora": r"\bclean\b",
}
INGREDIENT_EVIDENCE: dict[str, str] = {
    "hyaluronic acid": r"hyaluron|\bha\b", "niacinamide": r"niacinamide|vitamin b3",
    "retinol": r"retinol", "retinoids": r"retino|retinal", "vitamin c": r"vitamin c|vit c|ascorb",
    "vitamin e": r"vitamin e|tocopherol", "salicylic acid": r"salicylic|\bbha\b",
    "glycolic acid": r"glycolic|\baha\b", "lactic acid": r"lactic", "ceramides": r"ceramide",
    "squalane": r"squalane", "peptides": r"peptide", "bakuchiol": r"bakuchiol",
    "caffeine": r"caffeine", "centella asiatica": r"centella|cica", "zinc oxide": r"zinc|mineral",
    "shea butter": r"shea", "argan oil": r"argan", "aloe": r"aloe",
}
SKIN_EVIDENCE: dict[str, str] = {
    "dry": r"\bdry|dehydrat", "oily": r"\boil|greasy|shine", "combination": r"combination|combo",
    "normal": r"\bnormal", "sensitive": r"sensitive|irritat|redness|react",
}


def _output_schema() -> dict:
    """Strict JSON schema for the LLM's reply (all keys required, nulls allowed)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["intent", "semantic_query", "max_price", "min_price", "min_rating", "categories",
                     "brands", "in_stock_only", "required_attributes", "required_ingredients",
                     "skin_types", "product_names", "notes"],
        "properties": {
            "intent": {"type": "string", "enum": [i.value for i in Intent]},
            "semantic_query": {"type": "string", "maxLength": 200},
            "max_price": {"type": ["number", "null"]},
            "min_price": {"type": ["number", "null"]},
            "min_rating": {"type": ["number", "null"]},
            "categories": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["primary", "secondary", "tertiary"],
                "properties": {"primary": _NULLABLE_STR, "secondary": _NULLABLE_STR,
                               "tertiary": _NULLABLE_STR}}},
            "brands": {"type": "array", "items": {"type": "string"}},
            "in_stock_only": {"type": "boolean"},
            "required_attributes": {"type": "array", "items": {
                "type": "string", "enum": sorted(ATTRIBUTE_PATTERNS)}},
            "required_ingredients": {"type": "array", "items": {
                "type": "string", "enum": sorted(INGREDIENT_SYNONYMS)}},
            "skin_types": {"type": "array", "items": {
                "type": "string", "enum": [s.value for s in SkinType]}},
            "product_names": {"type": "array", "maxItems": 4, "items": {"type": "string", "maxLength": 150}},
            "notes": {"type": "array", "maxItems": 4, "items": {"type": "string", "maxLength": 200}},
        },
    }


def _category_paths(catalog: Catalog) -> list[str]:
    counts = Counter(" > ".join(x for x in (p.primary_category, p.secondary_category,
                                            p.tertiary_category) if x)
                     for p in catalog.products.values())
    return [f"{path} ({n})" for path, n in sorted(counts.items())]


SYSTEM_PROMPT = """\
You convert a shopper's beauty-product request into a JSON search specification
for a fixed product catalogue. Output JSON only, matching the schema.

Rules:
- Extract ONLY what the user explicitly asked for. Never invent a budget,
  rating, brand, category, skin type or attribute. If unsure, leave it empty or null.
- Prices are US dollars. If the user gives a price in another currency, set
  the price fields to null and explain in notes.
- "highly rated", "top rated" or "best rated" with no number -> min_rating 4.0.
  An explicit number ("4.5 stars") -> that number.
- categories: choose the MOST SPECIFIC matching path(s) from CATEGORY PATHS,
  copying names exactly. Fill only the levels you need: for "shampoo" use
  primary "Hair", secondary null, tertiary "Shampoo". For a broad request
  ("skincare products") use only the primary level. Several entries mean OR.
  Use [] when no product type is stated.
- brands: only brands the user names, spelled exactly as in BRANDS.
- required_attributes / required_ingredients: only explicit requirements
  ("fragrance-free", "with niacinamide").
- skin_types: skin types the PRODUCT should suit, e.g. "for dry skin" -> ["dry"].
- intent: product_comparison for "compare X and Y" / "X vs Y";
  review_question for "what do customers say about X"; otherwise
  product_recommendation. For those two intents, put the product names
  (with brand, if given) in product_names, and leave the filters empty.
- semantic_query: a short description of the desired product and its
  qualities (texture, finish, use, concerns), without prices or ratings.
- in_stock_only: true only if the user asks for in-stock or available items.
- notes: at most 3 very short notes on interpretations or ambiguity. Keep the JSON compact.

CATEGORY PATHS (product counts in brackets):
{categories}

BRANDS:
{brands}
"""


def merge_with_rules(llm: StructuredQuery, rules: StructuredQuery) -> StructuredQuery:
    """Combine the LLM parse with the rule-based parse.

    Why: in the live evaluation the LLM sometimes DROPPED constraints the
    user stated with plain keywords ("Fragrance-free skincare products" lost
    "skincare"; "gentle cleanser" lost "cleanser"). The rule parser's keyword
    matches are high-precision, and the LLM adds paraphrase understanding
    ("no perfume" -> Fragrance Free). Both sides have already passed their
    own checks, so combining them does not add invented constraints.

    Policy:
      * intent: an explicit comparison/review pattern found by the rules wins.
      * categories: the rule categories when the rules found any (exact
        vocabulary), otherwise the LLM's.
      * price / rating: the LLM value if present, else the rule value.
      * brands, attributes, ingredients, skin types: union.
    """
    added: list[str] = []
    update: dict = {}

    if rules.intent != Intent.RECOMMENDATION and llm.intent != rules.intent:
        update.update(intent=rules.intent, product_names=rules.product_names, categories=[], brands=[])
        added.append(f"intent {rules.intent.value}")
    intent = update.get("intent", llm.intent)

    if intent == Intent.RECOMMENDATION:
        if rules.categories and set(rules.categories) != set(llm.categories):
            update["categories"] = rules.categories
            added.append("categories " + " OR ".join(c.label() for c in rules.categories))
        new_brands = [b for b in rules.brands if b not in llm.brands]
        if new_brands:
            update["brands"] = llm.brands + new_brands
            added.append("brands " + ", ".join(new_brands))
    for field_name in ("max_price", "min_price", "min_rating"):
        if getattr(llm, field_name) is None and getattr(rules, field_name) is not None:
            update[field_name] = getattr(rules, field_name)
            added.append(f"{field_name} {getattr(rules, field_name):g}")
    for field_name in ("required_attributes", "required_ingredients", "skin_types"):
        extra = [v for v in getattr(rules, field_name) if v not in getattr(llm, field_name)]
        if extra:
            update[field_name] = getattr(llm, field_name) + extra
            added.append(field_name.replace("_", " ") + " " + ", ".join(getattr(v, "value", v) for v in extra))
    if rules.in_stock_only and not llm.in_stock_only:
        update["in_stock_only"] = True
        added.append("in stock")
    if added:
        update["notes"] = llm.notes + ["Added from explicit keywords (rule parser): " + "; ".join(added) + "."]
    return llm.model_copy(update=update) if update else llm


class LLMQueryParser:
    """Callable with the same signature as the rule-based parser."""

    def __init__(self, client: LLMClient, catalog: Catalog) -> None:
        self.client = client
        self.catalog = catalog
        self.schema = _output_schema()
        self.system_prompt = SYSTEM_PROMPT.format(
            categories="\n".join(_category_paths(catalog)),
            brands="; ".join(sorted({p.brand for p in catalog.products.values()}, key=str.lower)))
        self._cache: dict[str, StructuredQuery] = {}
        self.last_error: str | None = None

    # ------------------------------------------------------------------ API

    def __call__(self, text: str, known_brands: set[str]) -> StructuredQuery:
        key = text.strip()
        if key in self._cache:
            return self._cache[key].model_copy(deep=True)
        try:
            query = self._parse_with_llm(key)
        except (LLMError, ValueError, ValidationError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("LLM parsing failed (%s); using rule-based fallback", self.last_error)
            query = parse_query(key, known_brands)
            query.notes.insert(0, f"LLM parser unavailable ({type(exc).__name__}); "
                                  "used the rule-based fallback parser.")
            return query                                  # failures are not cached
        query = merge_with_rules(query, parse_query(key, known_brands))
        self._cache[key] = query.model_copy(deep=True)
        return query

    # -------------------------------------------------------------- internals

    def _ask(self, user: str) -> dict:
        reply = self.client.complete(self.system_prompt, user, json_schema=self.schema,
                                     schema_name="sephra_query", temperature=0.0,
                                     max_tokens=4000, reasoning_effort="low")
        text = reply.text.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)     # tolerate code fences
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM did not return valid JSON: {exc}") from exc

    def _ask_with_json_retry(self, user: str) -> dict:
        """Free preview models sometimes return truncated JSON; retry once."""
        try:
            return self._ask(user)
        except ValueError as exc:
            logger.info("Retrying after invalid JSON: %s", exc)
            return self._ask(user + "\n\nReturn compact JSON only. Keep notes and semantic_query short.")

    def _parse_with_llm(self, text: str) -> StructuredQuery:
        user = f"User request: {text}"
        data = self._ask_with_json_retry(user)
        query, problems = self._build(text, data)
        if problems:                                      # one corrective retry
            data = self._ask_with_json_retry(user + "\n\nYour previous answer was invalid: " + "; ".join(problems)
                             + ". Use only exact values from CATEGORY PATHS and BRANDS.")
            query, problems = self._build(text, data)
            if problems:
                raise ValueError("LLM output still invalid: " + "; ".join(problems))
        return query

    def _build(self, text: str, data: dict) -> tuple[StructuredQuery, list[str]]:
        notes = ["Parsed by the LLM query parser."] + [str(n) for n in data.get("notes") or []]
        data = dict(data)
        data["notes"] = notes
        data["original_query"] = text
        data["parser"] = "llm"
        if not (data.get("semantic_query") or "").strip():
            data["semantic_query"] = text
        data["categories"] = [{k: v for k, v in c.items() if v} for c in data.get("categories") or []
                              if any(c.values())]
        for field_name in ("max_price", "min_price"):     # drop non-positive prices
            if data.get(field_name) is not None and data[field_name] <= 0:
                data[field_name] = None
        self._guard_numbers(text, data, notes)
        self._guard_brands(text, data, notes)
        self._guard_evidence(text, data, notes)
        query = StructuredQuery.model_validate(data)
        if query.intent != Intent.RECOMMENDATION:
            query = query.model_copy(update={"categories": [], "brands": []})
        problems = validate_query(query, self.catalog)
        for category in query.categories:
            if not any(matches_category(p, category) for p in self.catalog.products.values()):
                problems.append(f"category path {category.label()!r} matches no product")
        return query, problems

    @staticmethod
    def _numbers_in(text: str) -> set[float]:
        return {float(x) for x in re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))}

    def _guard_numbers(self, text: str, data: dict, notes: list[str]) -> None:
        stated = self._numbers_in(text)
        for field_name in ("max_price", "min_price"):
            value = data.get(field_name)
            if value is not None and float(value) not in stated:
                notes.append(f"Removed {field_name}={value}: that number is not in the request.")
                data[field_name] = None
        rating = data.get("min_rating")
        if rating is not None:
            highly = re.search(r"highly|top|best|well|high(?:est)?[- ]rated|good ratings?", fold(text))
            if float(rating) not in stated and not (highly and float(rating) == HIGHLY_RATED_THRESHOLD):
                notes.append(f"Removed min_rating={rating}: not stated in the request.")
                data["min_rating"] = None

    def _guard_evidence(self, text: str, data: dict, notes: list[str]) -> None:
        """Hard requirements must be backed by a related word in the request.

        Found in live testing: for "moisturizer for dry skin under $30" the
        model added "Vegan" and "Cruelty-Free", which nobody asked for.
        Keywords are deliberately broad ("perfume" supports Fragrance Free),
        so this only removes requirements with NO basis in the text.
        """
        folded = fold(text)

        def keep(values: list[str], evidence: dict[str, str], label: str) -> list[str]:
            kept = []
            for value in values or []:
                pattern = evidence.get(value)
                if pattern and re.search(pattern, folded):
                    kept.append(value)
                else:
                    notes.append(f"Removed {label} {value!r}: not requested.")
            return kept

        data["required_attributes"] = keep(data.get("required_attributes"), ATTRIBUTE_EVIDENCE, "attribute")
        data["required_ingredients"] = keep(data.get("required_ingredients"), INGREDIENT_EVIDENCE, "ingredient")
        data["skin_types"] = keep(data.get("skin_types"), SKIN_EVIDENCE, "skin type")
        if data.get("in_stock_only") and not re.search(r"stock|availab", folded):
            notes.append("Removed in_stock_only: not requested.")
            data["in_stock_only"] = False

    def _guard_brands(self, text: str, data: dict, notes: list[str]) -> None:
        tokens = " " + " ".join(re.findall(r"[a-z0-9]+", fold(text))) + " "
        kept = []
        for brand in data.get("brands") or []:
            if any(f" {alias} " in tokens for alias in _brand_aliases(brand)):
                kept.append(brand)
            else:
                notes.append(f"Removed brand {brand!r}: not mentioned in the request.")
        data["brands"] = kept
