"""Grounding of answers (test 15): evidence assembly + automatic checker.

No LLM is called. The checker is tested with hand-written answers: one
grounded, and several that contain typical hallucinations.
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def response(catalog):
    from sephra.retrieval.pipeline import SephraRetriever
    return SephraRetriever(catalog).search("Moisturizer for dry skin under $30")


def test_context_contains_ids_prices_and_review_attribution(response):
    from sephra.generation.context import build_context
    context = build_context(response)
    for result in response.products:
        assert f"product_id={result.product.product_id}" in context
        assert f"${result.product.price_usd:.2f}" in context
        for selected in result.evidence.selected:
            assert f"review_id={selected.review.review_id}, product_id={result.product.product_id}" in context
    assert "NOT a representative sample" in context or all(
        not r.evidence.selected for r in response.products)
    assert "not affiliated with Sephora" in context


def test_context_states_missing_reviews(response):
    from sephra.generation.context import build_context
    zero = [r for r in response.products if r.evidence.loaded_review_count == 0]
    if zero:
        assert "No customer reviews for this product in our loaded dataset" in build_context(response)


def test_grounded_answer_passes(response):
    from sephra.generation.grounding import check_answer_grounding
    top = response.products[0].product
    answer = (f"{top.name} [{top.product_id}] costs ${top.price_usd:.2f} in the dataset, "
              f"within your $30 budget.")
    assert check_answer_grounding(answer, response).is_grounded


def test_invented_price_product_and_rating_are_flagged(response):
    from sephra.generation.grounding import check_answer_grounding
    answer = "Try the Miracle Cream [P999999] for $12.34, rated 4.37 out of 5."
    report = check_answer_grounding(answer, response)
    assert report.unknown_prices == ["$12.34"]
    assert report.unknown_product_ids == ["P999999"]
    assert report.unknown_ratings == ["4.37"]
    assert not report.is_grounded


def test_recommending_after_no_results_is_flagged(catalog):
    from sephra.generation.grounding import check_answer_grounding
    from sephra.retrieval.pipeline import SephraRetriever
    empty = SephraRetriever(catalog).search("Tatcha moisturizer under $10")
    assert check_answer_grounding("You could try [P441101].", empty).recommends_when_no_results


def test_no_provider_means_retrieval_only_mode(monkeypatch):
    from sephra.generation.llm_client import get_llm_client
    monkeypatch.delenv("SEPHRA_LLM_PROVIDER", raising=False)
    assert get_llm_client(load_env_file=False) is None
