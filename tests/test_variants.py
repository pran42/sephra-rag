"""Unit tests for product-family (variant) detection."""

from types import SimpleNamespace

from sephra.retrieval.variants import (are_name_variants, categories_compatible,
                                       has_size_marker, name_tokens)


def _product(primary, secondary=None, tertiary=None):
    return SimpleNamespace(primary_category=primary, secondary_category=secondary,
                           tertiary_category=tertiary)


def test_name_tokens_ignore_size_words_and_accents():
    assert name_tokens("Mini Lash Idôle Mascara") == {"lash", "idole", "mascara"}


def test_size_marker_detection():
    assert has_size_marker("Mini Maracuja Oil")
    assert has_size_marker("VANILLA MILK Eau de Parfum Travel Spray")
    assert not has_size_marker("Maracuja Oil")


def test_mini_variant_with_dropped_words_is_grouped():
    assert are_name_variants("Confidence in a Cream Anti-Aging Hydrating Moisturizer",
                             "Mini Confidence in a Cream Hydrating Moisturizer")


def test_different_products_are_not_grouped():
    assert not are_name_variants("Hydrating Oil-Free Gel Moisturizer",
                                 "Nourishing Rich Moisturizer with Hyaluronic Acid")


def test_different_scents_without_size_marker_are_not_grouped():
    assert not are_name_variants("Good Girl Eau de Parfum", "Good Girl Blush Eau de Parfum")


def test_mini_with_different_volume_number_is_not_grouped():
    assert not are_name_variants("Norvina Mini Pro Pigment Palette Vol. 2",
                                 "Norvina Pro Pigment Palette Vol. 3")


def test_category_compatibility():
    oil = _product("Hair", "Hair Styling & Treatments", "Hair Oil")
    shampoo = _product("Hair", "Shampoo & Conditioner", "Shampoo")
    mini_hair = _product("Mini Size", "Hair")
    assert not categories_compatible(oil, shampoo)
    assert categories_compatible(oil, mini_hair)
