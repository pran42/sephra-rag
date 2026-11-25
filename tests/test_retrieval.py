"""Retrieval: fusion, BM25, diversity, name resolution, reviews, full pipeline."""

from types import SimpleNamespace

import pytest

from sephra.retrieval.bm25 import tokenize
from sephra.retrieval.fusion import reciprocal_rank_fusion
from sephra.retrieval.ranking import ProductCandidate, diversify


# ------------------------------- unit tests -------------------------------

def test_rrf_rewards_agreement_between_rankers():
    fused = reciprocal_rank_fusion({"dense": ["A", "B", "C"], "bm25": ["B", "D"]}, k=60)
    order = [item for item, _, _ in fused]
    assert order[0] == "B"                      # ranked by both
    assert set(order) == {"A", "B", "C", "D"}
    assert fused[0][2] == {"dense": 2, "bm25": 1}


def test_rrf_score_formula():
    fused = dict((i, s) for i, s, _ in reciprocal_rank_fusion({"x": ["A"]}, k=60))
    assert fused["A"] == pytest.approx(1 / 61)


def test_bm25_tokenizer_stems_plurals_and_drops_stopwords():
    assert tokenize("Moisturizers for the Dry Skin") == ["moisturizer", "dry", "skin"]
    assert tokenize("glass") == ["glass"]


def _fake_catalog(products):
    by_id = {p.product_id: p for p in products}
    families = {}
    for p in products:
        families.setdefault(p.family_id, []).append(p.product_id)
    return SimpleNamespace(products=by_id,
                           family_members=lambda pid: families[by_id[pid].family_id])


def _p(pid, brand, family):
    return SimpleNamespace(product_id=pid, brand=brand, family_id=family)


def test_diversify_collapses_variants_and_attaches_eligible_ones():
    catalog = _fake_catalog([_p("P1", "A", "f1"), _p("P2", "A", "f1"), _p("P3", "B", "f3")])
    chosen = diversify([ProductCandidate("P1"), ProductCandidate("P2"), ProductCandidate("P3")],
                       catalog, top_k=5, eligible_ids={"P1", "P2", "P3"})
    assert [c.product_id for c in chosen] == ["P1", "P3"]
    assert chosen[0].variant_ids == ["P2"]


def test_diversify_does_not_attach_ineligible_variant():
    catalog = _fake_catalog([_p("P1", "A", "f1"), _p("P2", "A", "f1")])
    chosen = diversify([ProductCandidate("P1")], catalog, top_k=5, eligible_ids={"P1"})
    assert chosen[0].variant_ids == []           # e.g. the variant is over budget


def test_diversify_brand_cap_with_backfill():
    catalog = _fake_catalog([_p(f"P{i}", "A", f"f{i}") for i in range(4)] + [_p("P9", "B", "f9")])
    candidates = [ProductCandidate(f"P{i}") for i in range(4)] + [ProductCandidate("P9")]
    ids = {c.product_id for c in candidates}
    chosen = diversify(candidates, catalog, top_k=4, eligible_ids=ids, max_per_brand=2)
    assert [c.product_id for c in chosen] == ["P0", "P1", "P9", "P2"]   # P2 back-filled
    no_cap = diversify(candidates, catalog, top_k=4, eligible_ids=ids, max_per_brand=None)
    assert [c.product_id for c in no_cap] == ["P0", "P1", "P2", "P3"]


# --------------------------- integration tests ----------------------------

@pytest.fixture(scope="module")
def retriever(catalog):
    from sephra.retrieval.pipeline import SephraRetriever
    return SephraRetriever(catalog)


@pytest.mark.integration
def test_name_resolver_finds_exact_product_and_rejects_nonsense(catalog):
    from sephra.retrieval.resolver import resolve_name, resolve_names
    assert resolve_name("Tata Harper Repairative Anti-Aging Moisturizer", catalog).product_id == "P379709"
    assert resolve_name("zorblax unicorn tears", catalog).product_id is None
    pair = resolve_names(["la mer the moisturizing cool gel cream and tata harper repairative "
                          "anti-aging moisturizer"], catalog)
    assert [r.product_id for r in pair] == ["P429637", "P379709"]


@pytest.mark.integration
def test_bm25_respects_eligible_set(catalog):
    from sephra.retrieval.bm25 import get_bm25_index
    eligible = {"P500094", "P500120"}
    hits = get_bm25_index().search("moisturizer", eligible, top_n=50)
    assert {pid for pid, _ in hits} <= eligible


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_budget_query_enforces_price_and_category(retriever, mode):
    response = retriever.search("Moisturizer for dry skin under $30 that is hydrating and not greasy",
                                mode=mode)
    assert response.status == "ok" and len(response.products) == 5
    for result in response.products:
        assert result.product.price_usd <= 30
        assert result.product.secondary_category == "Moisturizers"
    assert "P441101" not in {r.product.product_id for r in response.products}   # $70 Tatcha


@pytest.mark.integration
def test_no_duplicate_products_or_families(retriever):
    response = retriever.search("Moisturizer for dry skin under $30")
    ids = [r.product.product_id for r in response.products]
    families = [r.product.family_id for r in response.products]
    assert len(ids) == len(set(ids)) and len(families) == len(set(families))


@pytest.mark.integration
def test_variants_are_attached_not_repeated(retriever):
    response = retriever.search("Confidence in a Cream hydrating moisturizer under $30")
    top_ids = [r.product.product_id for r in response.products]
    assert not {"P411403", "P427641"} <= set(top_ids)
    rep = next(r for r in response.products if r.product.product_id in {"P411403", "P427641"})
    assert {v.product_id for v in rep.variants} & {"P411403", "P427641"}


@pytest.mark.integration
def test_every_review_belongs_to_its_product(retriever):
    for text in ["Moisturizer for dry skin under $30", "Highly rated shampoo for damaged hair",
                 "Compare La Mer The Moisturizing Cool Gel Cream and Tata Harper Repairative "
                 "Anti-Aging Moisturizer"]:
        for result in retriever.search(text).products:
            for selected in result.evidence.selected:
                assert selected.review.product_id == result.product.product_id


@pytest.mark.integration
def test_zero_review_product_is_kept_with_explicit_note(retriever):
    response = retriever.search("Hydrating oil-free gel moisturizer under $30")
    zero = [r for r in response.products if r.evidence.loaded_review_count == 0]
    assert zero, "expected at least one zero-review product"
    for result in zero:
        assert result.evidence.selected == []
        assert any("loaded dataset" in note for note in result.evidence.notes)


@pytest.mark.integration
def test_balanced_evidence_includes_positive_and_critical(retriever):
    response = retriever.search("What do customers say about the Tata Harper Repairative "
                                "Anti-Aging Moisturizer?")
    assert response.status == "ok"
    evidence = response.products[0].evidence
    sentiments = {s.review.sentiment for s in evidence.selected}
    assert {"positive", "critical"} <= sentiments
    assert evidence.stats.total == 136 == evidence.loaded_review_count


@pytest.mark.integration
def test_no_results_are_reported_without_relaxing(retriever):
    response = retriever.search("Tatcha moisturizer under $10")
    assert response.status == "no_results" and response.products == []
    assert "price <= $10" in response.message


@pytest.mark.integration
def test_unknown_product_is_not_guessed(retriever):
    response = retriever.search("What do customers say about the Zorblax Unicorn Tears Elixir?")
    assert response.status == "product_not_found" and response.products == []


@pytest.mark.integration
def test_invalid_structured_query_is_rejected(retriever):
    from sephra.query.schema import StructuredQuery
    query = StructuredQuery(original_query="x", semantic_query="x", brands=["Made Up Brand"])
    assert retriever.search_structured(query).status == "invalid_query"


@pytest.mark.integration
def test_brand_query_returns_only_that_brand_and_skips_brand_cap(retriever):
    response = retriever.search("Show me skincare products from Drunk Elephant")
    assert len(response.products) == 5
    assert {r.product.brand for r in response.products} == {"Drunk Elephant"}


@pytest.mark.integration
def test_empty_eligible_set_never_queries_chroma_with_empty_in(retriever):
    response = retriever.search("vegan shampoo under $3")
    assert response.status == "no_results"
