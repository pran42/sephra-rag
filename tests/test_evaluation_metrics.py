"""Unit tests for ranking metrics and a sanity check of the rubrics."""

import pytest

from sephra.evaluation.metrics import (capped_recall_at_k, ndcg_at_k, precision_at_k,
                                       reciprocal_rank)

LABELS = {"A": 2, "B": 1, "C": 1}


def test_precision_counts_relevant_in_top_k():
    assert precision_at_k(["A", "X", "B", "Y", "Z"], LABELS, 5) == pytest.approx(0.4)


def test_precision_penalises_short_lists():
    assert precision_at_k(["A"], LABELS, 5) == pytest.approx(0.2)


def test_capped_recall():
    assert capped_recall_at_k(["A", "B"], LABELS, 2) == 1.0
    assert capped_recall_at_k(["A", "X", "Y"], LABELS, 3) == pytest.approx(1 / 3)


def test_reciprocal_rank():
    assert reciprocal_rank(["X", "Y", "B"], LABELS) == pytest.approx(1 / 3)
    assert reciprocal_rank(["X"], LABELS) == 0.0


def test_ndcg_perfect_and_imperfect():
    assert ndcg_at_k(["A", "B", "C"], LABELS, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["C", "B", "A"], LABELS, 3) < 1.0
    assert ndcg_at_k(["X"], LABELS, 3) == 0.0


@pytest.mark.integration
def test_rubrics_label_known_products_sensibly(catalog):
    from sephra.evaluation.relevance import RUBRICS
    tatcha_70 = catalog.get_product("P441101")
    assert RUBRICS["Q01"](tatcha_70) == 0          # $70 violates the $30 budget
    assert RUBRICS["Q01"](catalog.get_product("P500120")) == 2   # dry-skin tag, HA, $25
    assert RUBRICS["Q07"](catalog.get_product("P500120")) == 0   # not Drunk Elephant
