"""Catalogue loading, parent documents, review association and missing data."""

import pytest

from sephra.catalog import Review, parse_list_field, review_body_from_document


# ----------------------------- unit tests ---------------------------------

def test_parse_list_field_handles_python_list_strings():
    assert parse_list_field("['Vegan', 'Hydrating']") == ["Vegan", "Hydrating"]


def test_parse_list_field_handles_missing_values():
    assert parse_list_field(None) == []
    assert parse_list_field(float("nan")) == []
    assert parse_list_field("   ") == []


def test_parse_list_field_keeps_plain_text():
    assert parse_list_field("Water, Glycerin") == ["Water, Glycerin"]


def test_review_body_is_extracted_from_document():
    doc = "Product: X\nBrand: Y\nRating: 5\nReview: Great. I love it."
    assert review_body_from_document(doc) == "Great. I love it."


@pytest.mark.parametrize("rating,expected", [(5, "positive"), (4, "positive"),
                                             (3, "mixed"), (2, "critical"),
                                             (1, "critical"), (None, "unknown")])
def test_review_sentiment_buckets(rating, expected):
    review = Review("r", "P1", rating, None, None, "text")
    assert review.sentiment == expected


# -------------------------- integration tests -----------------------------

pytestmark_integration = pytest.mark.integration


@pytest.mark.integration
def test_catalog_counts_match_dataset(catalog):
    assert len(catalog.products) == 8494
    assert len(catalog.reviews) == 49918
    assert sum(1 for p in catalog.products.values() if p.loaded_review_count > 0) == 1104


@pytest.mark.integration
def test_every_product_has_its_parent_document(catalog):
    missing = [pid for pid, p in catalog.products.items() if not p.document_text]
    assert missing == []


@pytest.mark.integration
def test_review_counts_match_review_documents(catalog):
    for product_id, product in catalog.products.items():
        assert product.loaded_review_count == len(catalog.reviews_by_product.get(product_id, []))


@pytest.mark.integration
def test_every_review_belongs_to_a_known_product(catalog):
    for review in catalog.reviews.values():
        assert review.product_id in catalog.products


@pytest.mark.integration
def test_zero_review_product_is_still_in_catalog(catalog):
    product = catalog.get_product("P411403")  # Confidence in a Cream, 0 loaded reviews
    assert product is not None
    assert product.loaded_review_count == 0
    assert catalog.reviews_by_product.get("P411403", []) == []


@pytest.mark.integration
def test_missing_metadata_is_none_not_nan(catalog):
    no_rating = [p for p in catalog.products.values() if p.rating is None]
    assert len(no_rating) == 278          # known from the audit
    no_highlights = [p for p in catalog.products.values() if not p.highlights]
    assert len(no_highlights) == 2207


@pytest.mark.integration
def test_shared_reviews_between_size_variants_are_flagged(catalog):
    # All 156 Algae + Moringa reviews appear under both product IDs.
    shared = [r for r in catalog.reviews.values()
              if r.product_id == "P503991" and "P504026" in r.shared_product_ids]
    assert len(shared) == 156


@pytest.mark.integration
def test_known_variants_are_grouped_but_keep_their_own_data(catalog):
    assert set(catalog.family_members("P411403")) == {"P411403", "P427641"}
    full, mini = catalog.get_product("P411403"), catalog.get_product("P427641")
    assert full.rating != mini.rating
    assert full.loaded_review_count == 0 and mini.loaded_review_count == 47


@pytest.mark.integration
def test_unrelated_same_brand_products_are_not_grouped(catalog):
    # Two different Community Sixty-Six moisturizers.
    assert "P500120" not in catalog.family_members("P500094")
