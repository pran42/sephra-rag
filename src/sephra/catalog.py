"""The product and review catalogue: the single source of truth for facts.

Why a catalogue separate from ChromaDB?
- ChromaDB stores *chunks* plus a subset of metadata (no tertiary category,
  no highlights list, no ingredients field). Hard filters such as
  "shampoo" (a tertiary category) or "fragrance-free" (a highlight tag)
  need the full structured product record.
- Parent-document retrieval needs the complete original document, not a
  normalised chunk.

Everything is loaded ONCE per process via ``get_catalog()``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from sephra import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Product:
    """One product exactly as recorded in the dataset snapshot."""

    product_id: str
    name: str
    brand: str
    primary_category: str
    secondary_category: str | None
    tertiary_category: str | None
    price_usd: float
    sale_price_usd: float | None
    rating: float | None
    sephora_review_count: int | None   # Sephora's own count when scraped
    loves_count: int | None
    ingredients: str | None
    highlights: list[str]
    size: str | None
    variation_type: str | None
    variation_value: str | None
    out_of_stock: bool
    online_only: bool
    sephora_exclusive: bool
    document_text: str                 # full parent document (as embedded)
    loaded_review_count: int = 0       # reviews present in OUR loaded partition
    family_id: str = ""                # product-family / variant group id

    @property
    def category_path(self) -> str:
        parts = [self.primary_category, self.secondary_category, self.tertiary_category]
        return " > ".join(p for p in parts if p)


@dataclass(slots=True)
class Review:
    """One complete customer review (recovered from the parent document)."""

    review_id: str
    product_id: str
    rating: int | None
    skin_type: str | None              # the REVIEWER's skin type
    is_recommended: bool | None
    text: str                          # title + body
    shared_product_ids: tuple[str, ...] = ()  # other products with identical text

    @property
    def sentiment(self) -> str:
        """Rating-based bucket. 4-5 positive, 3 mixed, 1-2 critical."""
        if self.rating is None:
            return "unknown"
        if self.rating >= 4:
            return "positive"
        if self.rating <= 2:
            return "critical"
        return "mixed"


@dataclass
class Catalog:
    products: dict[str, Product]
    reviews: dict[str, Review]
    reviews_by_product: dict[str, list[str]]

    def get_product(self, product_id: str) -> Product | None:
        return self.products.get(product_id)

    def family_members(self, product_id: str) -> list[str]:
        product = self.products.get(product_id)
        if product is None:
            return []
        return self._families.get(product.family_id, [product_id])

    # populated by build_families()
    _families: dict[str, list[str]] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Helpers for messy CSV values
# ---------------------------------------------------------------------------

def _clean_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _clean_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _clean_int(value: Any) -> int | None:
    number = _clean_float(value)
    return None if number is None else int(number)


def parse_list_field(value: Any) -> list[str]:
    """Parse a Python-list-looking string such as "['Vegan', 'Hydrating']".

    The Kaggle CSV stores highlights and ingredients this way. We use
    ast.literal_eval (safe: it only evaluates literals, never code).
    """
    text = _clean_str(value)
    if text is None:
        return []
    if text.startswith("["):
        try:
            with warnings.catch_warnings():
                # Some ingredient strings contain backslashes ("\A"), which
                # trigger a harmless SyntaxWarning while parsing.
                warnings.simplefilter("ignore", SyntaxWarning)
                parsed = ast.literal_eval(text)
            return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError):
            logger.debug("Could not parse list field: %.60s", text)
    return [text]


def review_body_from_document(text: str) -> str:
    """Review documents end with a 'Review: ...' line; return just that part."""
    marker = "\nReview: "
    index = text.find(marker)
    return text[index + len(marker):].strip() if index != -1 else text.strip()


def _bool_or_none(value: Any) -> bool | None:
    number = _clean_float(value)
    return None if number is None else bool(number)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {path}\n"
            "Run the data pipeline scripts in src/ first (see README)."
        )
    return path


def _load_jsonl(path: Path) -> list[dict]:
    with open(_require(path), "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _load_products(review_counts: dict[str, int]) -> dict[str, Product]:
    df = pd.read_csv(_require(config.PRODUCTS_CSV))
    documents = {doc["id"]: doc["text"] for doc in _load_jsonl(config.PRODUCT_DOCUMENTS_JSONL)}

    products: dict[str, Product] = {}
    for row in df.to_dict(orient="records"):
        product_id = str(row["product_id"]).strip()
        ingredients = parse_list_field(row.get("ingredients"))
        products[product_id] = Product(
            product_id=product_id,
            name=_clean_str(row.get("product_name")) or "",
            brand=_clean_str(row.get("brand_name")) or "",
            primary_category=_clean_str(row.get("primary_category")) or "",
            secondary_category=_clean_str(row.get("secondary_category")),
            tertiary_category=_clean_str(row.get("tertiary_category")),
            price_usd=_clean_float(row.get("price_usd")) or 0.0,
            sale_price_usd=_clean_float(row.get("sale_price_usd")),
            rating=_clean_float(row.get("rating")),
            sephora_review_count=_clean_int(row.get("reviews")),
            loves_count=_clean_int(row.get("loves_count")),
            ingredients=" ".join(ingredients) if ingredients else None,
            highlights=parse_list_field(row.get("highlights")),
            size=_clean_str(row.get("size")),
            variation_type=_clean_str(row.get("variation_type")),
            variation_value=_clean_str(row.get("variation_value")),
            out_of_stock=bool(_clean_int(row.get("out_of_stock")) or 0),
            online_only=bool(_clean_int(row.get("online_only")) or 0),
            sephora_exclusive=bool(_clean_int(row.get("sephora_exclusive")) or 0),
            document_text=documents.get(f"product_{product_id}", ""),
            loaded_review_count=review_counts.get(product_id, 0),
        )

    missing_docs = sum(1 for p in products.values() if not p.document_text)
    if missing_docs:
        logger.warning("%d products have no parent document text", missing_docs)

    return products


def _load_reviews() -> tuple[dict[str, Review], dict[str, list[str]]]:
    reviews: dict[str, Review] = {}
    by_product: dict[str, list[str]] = defaultdict(list)
    products_by_text: dict[str, set[str]] = defaultdict(set)
    text_key_by_review: dict[str, str] = {}

    for doc in _load_jsonl(config.REVIEW_DOCUMENTS_JSONL):
        meta = doc.get("metadata", {})
        product_id = meta.get("product_id")
        if not product_id:
            logger.warning("Review %s has no product_id; skipped", doc.get("id"))
            continue
        body = review_body_from_document(doc["text"])
        review = Review(
            review_id=doc["id"],
            product_id=product_id,
            rating=_clean_int(meta.get("rating")),
            skin_type=_clean_str(meta.get("skin_type")),
            is_recommended=_bool_or_none(meta.get("is_recommended")),
            text=body,
        )
        reviews[review.review_id] = review
        by_product[product_id].append(review.review_id)

        key = hashlib.md5(body.lower().encode("utf-8")).hexdigest()
        products_by_text[key].add(product_id)
        text_key_by_review[review.review_id] = key

    # Flag reviews whose exact text also appears under another product id
    # (the dataset attaches the same review to size variants).
    for review_id, key in text_key_by_review.items():
        owners = products_by_text[key]
        if len(owners) > 1:
            review = reviews[review_id]
            review.shared_product_ids = tuple(sorted(owners - {review.product_id}))

    return reviews, dict(by_product)


def _load_review_counts() -> dict[str, int]:
    df = pd.read_csv(_require(config.REVIEW_COUNTS_CSV), usecols=["product_id", "review_count"])
    return dict(zip(df["product_id"].astype(str), df["review_count"].astype(int)))


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    """Load (once) and return the full catalogue."""
    from sephra.retrieval.variants import build_families  # avoid import cycle

    logger.info("Loading catalogue from %s", config.PROCESSED_DIR)
    review_counts = _load_review_counts()
    products = _load_products(review_counts)
    reviews, reviews_by_product = _load_reviews()

    # Consistency check: counts CSV vs. actual review documents.
    mismatches = [
        pid for pid, product in products.items()
        if product.loaded_review_count != len(reviews_by_product.get(pid, []))
    ]
    if mismatches:
        logger.warning("%d products have review-count mismatches (e.g. %s)",
                       len(mismatches), mismatches[:3])

    catalog = Catalog(products=products, reviews=reviews,
                      reviews_by_product=reviews_by_product)
    catalog._families = build_families(catalog)
    logger.info("Catalogue ready: %d products, %d reviews, %d multi-product families",
                len(products), len(reviews),
                sum(1 for members in catalog._families.values() if len(members) > 1))
    return catalog
