"""Read-only checks of the existing ChromaDB, plus baseline parity.

The parity tests prove the refactored dense retriever returns exactly what
the original src/retrieve.py returned (saved in
evaluation/results/baseline_original.json) when no filters are applied.
"""

import json

import pytest

from sephra import config

pytestmark = pytest.mark.integration

BASELINE_FILE = config.EVALUATION_DIR / "results" / "baseline_original.json"


def test_collections_exist_with_expected_counts_and_metric():
    from sephra.vector_store import get_product_collection, get_review_collection
    products, reviews = get_product_collection(), get_review_collection()
    assert products.count() == 10634
    assert reviews.count() == 50013
    assert products.metadata.get("hnsw:space") == "cosine"
    assert reviews.metadata.get("hnsw:space") == "cosine"


def test_query_embedding_shape_and_normalisation():
    from sephra.embeddings import embed_query
    vector = embed_query("moisturizer for dry skin")
    assert len(vector) == 384
    assert abs(sum(x * x for x in vector) - 1.0) < 1e-4


def test_embedding_model_is_loaded_only_once():
    from sephra.embeddings import get_embedding_model
    assert get_embedding_model() is get_embedding_model()


def test_empty_query_is_rejected():
    from sephra.embeddings import embed_query
    with pytest.raises(ValueError):
        embed_query("   ")


@pytest.mark.skipif(not BASELINE_FILE.exists(), reason="baseline not captured")
def test_dense_search_reproduces_original_baseline_products_and_reviews():
    from sephra.embeddings import embed_query
    from sephra.retrieval.dense import dense_product_search, dense_review_search

    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    for record in baseline:
        embedding = embed_query(record["query"])
        hits = dense_product_search(embedding, n_chunks=100)[:5]
        assert [h.item_id for h in hits] == [p["product_id"] for p in record["products"]], record["id"]

        for product in record["products"]:
            if product["review_count"] == 0:
                continue
            review_hits = dense_review_search(embedding, product["product_id"], n_chunks=3)
            assert [h.item_id for h in review_hits] == [r["review_id"] for r in product["reviews"]]


def test_review_search_never_returns_another_products_reviews(catalog):
    from sephra.embeddings import embed_query
    from sephra.retrieval.dense import dense_review_search

    embedding = embed_query("hydrating but not greasy")
    for product_id in ["P427641", "P500120", "P503991"]:
        for hit in dense_review_search(embedding, product_id, n_chunks=30):
            assert catalog.reviews[hit.item_id].product_id == product_id


def test_review_search_with_more_requested_than_available(catalog):
    from sephra.embeddings import embed_query
    from sephra.retrieval.dense import dense_review_search

    hits = dense_review_search(embed_query("smell"), "P500120", n_chunks=50)
    assert len(hits) == catalog.get_product("P500120").loaded_review_count  # 10
