"""Query schema validation and the rule-based parser."""

import pytest
from pydantic import ValidationError

from sephra.query.rule_parser import parse_query
from sephra.query.schema import CategoryFilter, Intent, SkinType, StructuredQuery
from sephra.query.vocabulary import PRODUCT_TYPE_PATTERNS, BROAD_CATEGORY_PATTERNS

BRANDS = ["Tatcha", "Drunk Elephant", "fresh", "Rare Beauty by Selena Gomez",
          "Kiehl's Since 1851", "La Mer", "Tata Harper", "The Ordinary"]


def parse(text: str) -> StructuredQuery:
    return parse_query(text, known_brands=BRANDS)


# ------------------------------ schema ------------------------------------

def test_schema_rejects_inverted_price_range():
    with pytest.raises(ValidationError):
        StructuredQuery(original_query="x", semantic_query="x", min_price=50, max_price=10)


def test_schema_rejects_out_of_range_rating():
    with pytest.raises(ValidationError):
        StructuredQuery(original_query="x", semantic_query="x", min_rating=7)


def test_schema_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        StructuredQuery(original_query="x", semantic_query="x", colour="red")


def test_category_filter_needs_a_level():
    with pytest.raises(ValidationError):
        CategoryFilter()


# ------------------------------- price ------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("moisturizer under $30", 30), ("moisturizer below 25 dollars", 25),
    ("serum less than $15.50", 15.5), ("cleanser up to $20", 20),
    ("sunscreen $25 or less", 25), ("budget of $40 for a moisturizer", 40),
    ("shampoo under 18", 18),
])
def test_max_price_extraction(text, expected):
    assert parse(text).max_price == expected


def test_price_range_extraction():
    query = parse("perfume between $50 and $100")
    assert (query.min_price, query.max_price) == (50, 100)


def test_no_price_is_invented():
    query = parse("Vitamin C serum for brightening")
    assert query.max_price is None and query.min_price is None


def test_rating_number_is_not_mistaken_for_price():
    query = parse("moisturizer rated at least 4 stars")
    assert query.min_rating == 4 and query.max_price is None and query.min_price is None


# ------------------------------- rating -----------------------------------

@pytest.mark.parametrize("text,expected", [
    ("products rated at least 4", 4.0), ("mascara 4.5+ stars", 4.5),
    ("shampoo with rating above 4.2", 4.2), ("highly rated shampoo", 4.0),
    ("top-rated serum", 4.0),
])
def test_rating_extraction(text, expected):
    assert parse(text).min_rating == expected


def test_highly_rated_interpretation_is_disclosed():
    assert any("highly rated" in note for note in parse("highly rated shampoo").notes)


# ------------------------------ categories --------------------------------

def test_moisturizer_maps_to_skincare_moisturizers():
    labels = [c.label() for c in parse("moisturizer for dry skin").categories]
    assert "Skincare > Moisturizers" in labels


def test_specific_phrase_wins_over_general_one():
    assert [c.tertiary for c in parse("dry shampoo").categories] == ["Dry Shampoo"]
    assert [c.tertiary for c in parse("hair oil for frizz").categories] == ["Hair Oil"]
    assert [c.tertiary for c in parse("body mist").categories] == ["Body Mist & Hair Mist"]


def test_broad_category_only_when_no_specific_type():
    assert [c.primary for c in parse("fragrance-free skincare products").categories] == ["Skincare"]


def test_fragrance_free_is_not_read_as_fragrance_category():
    query = parse("fragrance-free moisturizer")
    assert all(c.primary != "Fragrance" for c in query.categories)
    assert query.required_attributes == ["Fragrance Free"]


def test_no_category_invented_for_ambiguous_query():
    query = parse("I need something nice")
    assert query.categories == [] and not query.has_hard_constraints()
    assert any("No product type" in note for note in query.notes)


# -------------------------------- brands ----------------------------------

def test_brand_extraction():
    assert parse("skincare products from Drunk Elephant").brands == ["Drunk Elephant"]


def test_brand_alias_without_by_suffix():
    assert parse("Rare Beauty blush").brands == ["Rare Beauty by Selena Gomez"]


def test_ambiguous_brand_needs_context():
    assert parse("a fresh scent perfume").brands == []
    assert parse("moisturizer from fresh").brands == ["fresh"]


# ---------------------------- attributes etc. -----------------------------

def test_skin_types_are_soft_preferences():
    query = parse("serum for dry and sensitive skin")
    assert query.skin_types == [SkinType.DRY, SkinType.SENSITIVE]
    assert not any("skin" in c for c in query.describe_constraints())


def test_dry_shampoo_is_not_a_skin_type():
    assert parse("dry shampoo").skin_types == []


def test_ingredient_extraction():
    assert parse("serum with hyaluronic acid").required_ingredients == ["hyaluronic acid"]
    assert parse("vitamin c serum").required_ingredients == ["vitamin c"]


def test_in_stock_request():
    assert parse("in stock moisturizer").in_stock_only
    assert not parse("moisturizer").in_stock_only


# -------------------------------- intents ---------------------------------

def test_comparison_intent_and_names():
    query = parse("Compare La Mer The Moisturizing Cool Gel Cream vs Tata Harper Repairative Moisturizer")
    assert query.intent == Intent.COMPARISON
    assert len(query.product_names) == 2


def test_review_question_intent():
    query = parse("What do customers say about the Tata Harper Repairative Anti-Aging Moisturizer?")
    assert query.intent == Intent.REVIEW_QUESTION
    assert query.product_names == ["tata harper repairative anti-aging moisturizer"]


def test_empty_query_rejected():
    with pytest.raises(ValueError):
        parse_query("   ")


# ------------------- vocabulary matches real dataset values ---------------

@pytest.mark.integration
def test_every_vocabulary_category_exists_in_dataset(catalog):
    from sephra.query.filters import validate_query
    for _, filters in PRODUCT_TYPE_PATTERNS + BROAD_CATEGORY_PATTERNS:
        query = StructuredQuery(original_query="x", semantic_query="x", categories=filters)
        assert validate_query(query, catalog) == [], filters
