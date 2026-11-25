"""Hard filters: price, rating, category, brand, availability, attributes."""

import pytest

from sephra.catalog import Product
from sephra.query.filters import (build_metadata_where, has_attribute, has_ingredient,
                                  product_violations)
from sephra.query.schema import CategoryFilter, StructuredQuery


def make_product(**overrides) -> Product:
    base = dict(product_id="P1", name="Test Cream", brand="BrandA", primary_category="Skincare",
                secondary_category="Moisturizers", tertiary_category="Moisturizers",
                price_usd=25.0, sale_price_usd=None, rating=4.2, sephora_review_count=10,
                loves_count=0, ingredients="Water, Glycerin, Sodium Hyaluronate",
                highlights=["Vegan"], size=None, variation_type=None, variation_value=None,
                out_of_stock=False, online_only=False, sephora_exclusive=False,
                document_text="Product: Test Cream")
    base.update(overrides)
    return Product(**base)


def query(**kwargs) -> StructuredQuery:
    return StructuredQuery(original_query="q", semantic_query="q", **kwargs)


# ----------------------------- unit tests ---------------------------------

def test_max_price_is_enforced():
    assert product_violations(make_product(price_usd=70), query(max_price=30)) == ["price <= $30"]
    assert product_violations(make_product(price_usd=30), query(max_price=30)) == []


def test_min_rating_is_enforced_and_missing_rating_fails():
    assert product_violations(make_product(rating=3.9), query(min_rating=4)) == ["rating >= 4"]
    assert product_violations(make_product(rating=None), query(min_rating=4)) == ["rating >= 4"]
    assert product_violations(make_product(rating=4.0), query(min_rating=4)) == []


def test_category_filter_or_semantics():
    q = query(categories=[CategoryFilter(primary="Hair", tertiary="Shampoo"),
                          CategoryFilter(primary="Skincare", tertiary="Moisturizers")])
    assert product_violations(make_product(), q) == []
    assert product_violations(make_product(tertiary_category="Face Serums"), q) != []


def test_brand_filter():
    assert product_violations(make_product(brand="BrandB"), query(brands=["BrandA"])) == ["brand: BrandA"]


def test_in_stock_filter_only_when_requested():
    assert product_violations(make_product(out_of_stock=True), query()) == []
    assert product_violations(make_product(out_of_stock=True), query(in_stock_only=True)) == ["in stock"]


def test_attribute_from_tag_or_name():
    assert has_attribute(make_product(highlights=["Fragrance Free"]), "Fragrance Free")
    assert has_attribute(make_product(name="Hydrating Oil-Free Gel Moisturizer"), "Oil Free")
    assert not has_attribute(make_product(), "Fragrance Free")


def test_ingredient_synonyms_and_missing_ingredient_list():
    assert has_ingredient(make_product(), "hyaluronic acid")          # Sodium Hyaluronate
    assert not has_ingredient(make_product(ingredients=None), "niacinamide")


def test_no_constraints_means_no_violations():
    assert product_violations(make_product(price_usd=1900, rating=None), query()) == []


def test_native_chroma_where_syntax():
    where = build_metadata_where(query(max_price=30, min_rating=4,
                                       categories=[CategoryFilter(primary="Skincare",
                                                                  secondary="Moisturizers")]))
    assert where == {"$and": [{"price_usd": {"$lte": 30}}, {"rating": {"$gte": 4}},
                              {"$and": [{"primary_category": "Skincare"},
                                        {"secondary_category": "Moisturizers"}]}]}


def test_single_condition_is_not_wrapped_in_and():
    assert build_metadata_where(query(max_price=30)) == {"price_usd": {"$lte": 30}}
    assert build_metadata_where(query()) is None


# -------------------------- integration tests -----------------------------

@pytest.mark.integration
def test_moisturizer_under_30_excludes_the_70_dollar_tatcha(catalog):
    from sephra.query.filters import apply_hard_filters
    from sephra.query.rule_parser import parse_query
    q = parse_query("Moisturizer for dry skin under $30", known_brands={p.brand for p in catalog.products.values()})
    result = apply_hard_filters(q, catalog)
    assert "P441101" not in result.eligible_ids                 # $70 Tatcha
    assert {"P411403", "P500094", "P500120"} <= result.eligible_ids
    assert all(catalog.products[pid].price_usd <= 30 for pid in result.eligible_ids)


@pytest.mark.integration
def test_impossible_constraints_give_empty_set_not_relaxed(catalog):
    from sephra.query.filters import apply_hard_filters
    q = query(brands=["Tatcha"], max_price=10)
    result = apply_hard_filters(q, catalog)
    assert result.is_empty
    assert result.steps[-1] == ("price <= $10", 0)


@pytest.mark.integration
def test_unknown_values_are_reported_not_ignored(catalog):
    from sephra.query.filters import validate_query
    q = query(brands=["Not A Brand"], categories=[CategoryFilter(tertiary="Face Creams")])
    errors = validate_query(q, catalog)
    assert any("Not A Brand" in e for e in errors) and any("Face Creams" in e for e in errors)


@pytest.mark.integration
def test_chroma_native_filter_agrees_with_catalogue_filter(catalog):
    """Every chunk Chroma returns under the native filter must be eligible."""
    from sephra.embeddings import embed_query
    from sephra.query.filters import apply_hard_filters
    from sephra.retrieval.dense import dense_product_search
    q = query(max_price=30, min_rating=4,
              categories=[CategoryFilter(primary="Skincare", secondary="Moisturizers")])
    result = apply_hard_filters(q, catalog)
    assert "$and" in result.chroma_where                    # native syntax used
    hits = dense_product_search(embed_query("hydrating cream"), 200, where=result.chroma_where)
    assert hits and all(h.item_id in result.eligible_ids for h in hits)
