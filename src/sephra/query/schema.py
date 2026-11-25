"""The structured representation of a user query (validated with Pydantic).

Every parser (the rule-based fallback today, an OpenRouter LLM later) must
produce a ``StructuredQuery``. The retrieval pipeline only ever sees this
object, so swapping the parser never requires changing retrieval code.

Hard constraints vs. soft preferences
-------------------------------------
HARD (enforced in code, never relaxed): price, rating, categories, brands,
in-stock, required attributes (e.g. "Fragrance Free"), required ingredients.

SOFT (used for ranking only): skin types, and everything in
``semantic_query`` ("lightweight", "not greasy", "everyday use"...).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Intent(str, Enum):
    RECOMMENDATION = "product_recommendation"
    COMPARISON = "product_comparison"
    REVIEW_QUESTION = "review_question"


class SkinType(str, Enum):
    """Skin types the user wants the PRODUCT to suit (not reviewer skin type)."""

    DRY = "dry"
    OILY = "oily"
    COMBINATION = "combination"
    NORMAL = "normal"
    SENSITIVE = "sensitive"


class CategoryFilter(BaseModel):
    """A category path. Unset levels match anything.

    Example: CategoryFilter(primary="Hair", tertiary="Shampoo").
    Several CategoryFilters in one query are combined with OR.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary: str | None = None
    secondary: str | None = None
    tertiary: str | None = None

    @model_validator(mode="after")
    def _at_least_one_level(self) -> "CategoryFilter":
        if not (self.primary or self.secondary or self.tertiary):
            raise ValueError("CategoryFilter needs at least one category level")
        return self

    def label(self) -> str:
        return " > ".join(x for x in (self.primary, self.secondary, self.tertiary) if x)


class StructuredQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_query: str = Field(min_length=1)
    intent: Intent = Intent.RECOMMENDATION
    semantic_query: str = Field(min_length=1, description="Text used for dense + BM25 search")

    # ---- HARD constraints ----
    max_price: float | None = Field(default=None, gt=0)
    min_price: float | None = Field(default=None, gt=0)
    min_rating: float | None = Field(default=None, ge=0, le=5)
    categories: list[CategoryFilter] = Field(default_factory=list)   # OR
    brands: list[str] = Field(default_factory=list)                  # OR
    in_stock_only: bool = False
    required_attributes: list[str] = Field(default_factory=list)     # AND, highlight tags
    required_ingredients: list[str] = Field(default_factory=list)    # AND

    # ---- SOFT preferences ----
    skin_types: list[SkinType] = Field(default_factory=list)

    # ---- Named products (comparison / review questions) ----
    product_names: list[str] = Field(default_factory=list)

    # ---- Provenance ----
    parser: Literal["rule_based", "llm"] = "rule_based"
    notes: list[str] = Field(default_factory=list, description="Interpretations and ambiguities")

    @field_validator("brands", "required_attributes", "required_ingredients", "product_names")
    @classmethod
    def _strip_and_dedupe(cls, values: list[str]) -> list[str]:
        seen: list[str] = []
        for value in values:
            value = value.strip()
            if value and value not in seen:
                seen.append(value)
        return seen

    @model_validator(mode="after")
    def _price_range_is_valid(self) -> "StructuredQuery":
        if self.min_price is not None and self.max_price is not None \
                and self.min_price > self.max_price:
            raise ValueError(f"min_price ({self.min_price}) is greater than max_price ({self.max_price})")
        return self

    def has_hard_constraints(self) -> bool:
        return any([self.max_price, self.min_price, self.min_rating, self.categories,
                    self.brands, self.in_stock_only, self.required_attributes,
                    self.required_ingredients])

    def describe_constraints(self) -> list[str]:
        """Human-readable list of the hard constraints (for UI / prompts)."""
        parts: list[str] = []
        if self.max_price is not None:
            parts.append(f"price <= ${self.max_price:g}")
        if self.min_price is not None:
            parts.append(f"price >= ${self.min_price:g}")
        if self.min_rating is not None:
            parts.append(f"rating >= {self.min_rating:g}")
        if self.categories:
            parts.append("category: " + " OR ".join(c.label() for c in self.categories))
        if self.brands:
            parts.append("brand: " + " OR ".join(self.brands))
        if self.in_stock_only:
            parts.append("in stock (in the dataset snapshot)")
        for attribute in self.required_attributes:
            parts.append(f"attribute: {attribute}")
        for ingredient in self.required_ingredients:
            parts.append(f"ingredient: {ingredient}")
        return parts
