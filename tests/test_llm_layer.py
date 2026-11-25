"""LLM layer tests with fake clients: no network, no API key, no quota used."""

import json

import httpx
import pytest

from sephra.generation.llm_client import (LLMError, LLMRateLimitError, LLMReply,
                                          OpenRouterClient)


class FakeClient:
    """Returns queued replies (str) or raises queued exceptions."""

    model = "fake-model"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user, **kwargs):
        self.calls.append(user)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMReply(text=reply, model=self.model, latency_s=0.01)


def llm_json(**overrides) -> str:
    base = dict(intent="product_recommendation", semantic_query="hydrating moisturizer",
                max_price=None, min_price=None, min_rating=None, categories=[], brands=[],
                in_stock_only=False, required_attributes=[], required_ingredients=[],
                skin_types=[], product_names=[], notes=[])
    base.update(overrides)
    return json.dumps(base)


MOISTURIZER = [{"primary": "Skincare", "secondary": None, "tertiary": "Moisturizers"}]


# ----------------------------- OpenRouter client --------------------------

def _client(handler, retries=0):
    return OpenRouterClient("sk-test-key", "test/model", max_retries=retries,
                            transport=httpx.MockTransport(handler))


def test_openrouter_client_sends_auth_and_parses_reply():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"model": "test/model",
                                         "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]})

    reply = _client(handler).complete("sys", "user", json_schema={"type": "object"})
    assert reply.text == "hi" and seen["auth"] == "Bearer sk-test-key"
    assert seen["body"]["response_format"]["json_schema"]["strict"] is True


def test_openrouter_key_never_in_repr():
    assert "sk-test-key" not in repr(_client(lambda r: httpx.Response(200)))


def test_rate_limit_raises_specific_error_after_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    with pytest.raises(LLMRateLimitError):
        _client(handler, retries=2).complete("s", "u")
    assert len(calls) == 3


def test_client_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(LLMError, match="401"):
        _client(handler, retries=2).complete("s", "u")
    assert len(calls) == 1


def test_empty_content_is_an_error():
    handler = lambda r: httpx.Response(200, json={"choices": [{"message": {"content": ""},
                                                               "finish_reason": "length"}]})
    with pytest.raises(LLMError, match="no text"):
        _client(handler).complete("s", "u")


# --------------------------------- LLM parser ------------------------------

@pytest.fixture(scope="module")
def brands(catalog):
    return {p.brand for p in catalog.products.values()}


def make_parser(catalog, *replies):
    from sephra.query.llm_parser import LLMQueryParser
    client = FakeClient(*replies)
    return LLMQueryParser(client, catalog), client


@pytest.mark.integration
def test_valid_llm_output_is_used(catalog, brands):
    parser, _ = make_parser(catalog, llm_json(max_price=30, categories=MOISTURIZER, skin_types=["dry"]))
    query = parser("moisturizer for dry skin under $30", brands)
    assert query.parser == "llm" and query.max_price == 30
    assert query.categories[0].tertiary == "Moisturizers"


@pytest.mark.integration
def test_invented_price_and_rating_are_removed(catalog, brands):
    parser, _ = make_parser(catalog, llm_json(max_price=25, min_rating=4.5))
    query = parser("a nice moisturizer", brands)
    assert query.max_price is None and query.min_rating is None
    assert any("max_price" in n for n in query.notes)


@pytest.mark.integration
def test_highly_rated_maps_to_four(catalog, brands):
    parser, _ = make_parser(catalog, llm_json(min_rating=4.0))
    assert parser("highly rated shampoo", brands).min_rating == 4.0


@pytest.mark.integration
def test_invented_attributes_brands_and_skin_types_are_removed(catalog, brands):
    # Mirrors a real ERNIE output: unrequested Vegan / Cruelty-Free tags.
    parser, _ = make_parser(catalog, llm_json(required_attributes=["Vegan", "Cruelty-Free"],
                                              brands=["Tatcha"], skin_types=["oily"]))
    query = parser("moisturizer for dry skin", brands)
    assert query.required_attributes == [] and query.brands == []
    # invented "oily" removed; "dry" comes back from the rule parser (it IS in the text)
    assert [s.value for s in query.skin_types] == ["dry"]


@pytest.mark.integration
def test_paraphrased_attribute_is_kept(catalog, brands):
    parser, _ = make_parser(catalog, llm_json(required_attributes=["Fragrance Free"]))
    assert parser("no perfume please", brands).required_attributes == ["Fragrance Free"]


@pytest.mark.integration
def test_invalid_category_triggers_retry_then_success(catalog, brands):
    bad = llm_json(categories=[{"primary": "Skincare", "secondary": None, "tertiary": "Face Creams"}])
    parser, client = make_parser(catalog, bad, llm_json(categories=MOISTURIZER))
    query = parser("face cream", brands)
    assert query.parser == "llm" and len(client.calls) == 2
    assert "Face Creams" in client.calls[1]


@pytest.mark.integration
def test_impossible_category_combination_is_rejected(catalog, brands):
    bad = llm_json(categories=[{"primary": "Hair", "secondary": None, "tertiary": "Moisturizers"}])
    parser, _ = make_parser(catalog, bad, bad)
    query = parser("moisturizer", brands)
    assert query.parser == "rule_based"
    assert "fallback" in query.notes[0]


@pytest.mark.integration
@pytest.mark.parametrize("failure", [LLMRateLimitError("429"), LLMError("down")])
def test_llm_failure_falls_back_to_rules(catalog, brands, failure):
    parser, _ = make_parser(catalog, failure)
    query = parser("moisturizer under $30", brands)
    assert query.parser == "rule_based" and query.max_price == 30
    assert "fallback" in query.notes[0]


@pytest.mark.integration
def test_truncated_json_is_retried_once(catalog, brands):
    parser, client = make_parser(catalog, '{"intent": "product_rec', llm_json(categories=MOISTURIZER))
    assert parser("moisturizer", brands).parser == "llm" and len(client.calls) == 2


@pytest.mark.integration
def test_successful_parses_are_cached(catalog, brands):
    parser, client = make_parser(catalog, llm_json())
    parser("cache me", brands)
    parser("cache me", brands)
    assert len(client.calls) == 1


# ------------------------------- answer generation ------------------------

@pytest.fixture(scope="module")
def ok_response(catalog):
    from sephra.retrieval.pipeline import SephraRetriever
    return SephraRetriever(catalog).search("Moisturizer for dry skin under $30")


@pytest.mark.integration
def test_grounded_answer_is_returned(ok_response):
    from sephra.generation.answer_generator import generate_answer
    p = ok_response.products[0].product
    result = generate_answer(ok_response, FakeClient(f"Try {p.name} [{p.product_id}] at ${p.price_usd:.2f}."))
    assert result.status == "generated" and result.grounding.is_grounded


@pytest.mark.integration
def test_ungrounded_answer_is_retried_then_withheld(ok_response):
    from sephra.generation.answer_generator import generate_answer
    client = FakeClient("Buy [P000001] for $1.99.", "Still [P000001] for $1.99.")
    result = generate_answer(ok_response, client)
    assert result.status == "withheld" and result.text is None
    assert "P000001" in client.calls[1]                # the retry names the problem


@pytest.mark.integration
def test_ungrounded_then_fixed_answer_is_accepted(ok_response):
    from sephra.generation.answer_generator import generate_answer
    p = ok_response.products[0].product
    result = generate_answer(ok_response, FakeClient("Costs $1.99.", f"Costs ${p.price_usd:.2f}."))
    assert result.status == "generated" and result.attempts == 2


@pytest.mark.integration
def test_no_llm_call_for_no_results(catalog):
    from sephra.generation.answer_generator import generate_answer
    from sephra.retrieval.pipeline import SephraRetriever
    empty = SephraRetriever(catalog).search("Tatcha moisturizer under $10")
    client = FakeClient()
    assert generate_answer(empty, client).status == "skipped" and client.calls == []


@pytest.mark.integration
def test_rate_limit_during_answer_is_reported(ok_response):
    from sephra.generation.answer_generator import generate_answer
    assert generate_answer(ok_response, FakeClient(LLMRateLimitError("429"))).status == "rate_limited"


# ------------------------------ merge with rules --------------------------

@pytest.mark.integration
def test_oily_skin_is_not_evidence_for_oil_free(catalog, brands):
    # Real ERNIE behaviour on "Lightweight moisturizer for oily skin".
    parser, _ = make_parser(catalog, llm_json(required_attributes=["Oil Free"], skin_types=["oily"]))
    query = parser("Lightweight moisturizer for oily skin", brands)
    assert query.required_attributes == [] and [s.value for s in query.skin_types] == ["oily"]


@pytest.mark.integration
def test_merge_restores_category_the_llm_dropped(catalog, brands):
    # Real ERNIE behaviour on "Fragrance-free skincare products": no category.
    parser, _ = make_parser(catalog, llm_json(required_attributes=["Fragrance Free"]))
    query = parser("Fragrance-free skincare products", brands)
    assert [c.primary for c in query.categories] == ["Skincare"]
    assert query.parser == "llm" and any("rule parser" in n for n in query.notes)


@pytest.mark.integration
def test_merge_keeps_llm_paraphrase_understanding(catalog, brands):
    parser, _ = make_parser(catalog, llm_json(required_attributes=["Fragrance Free"],
                                              skin_types=["oily"]))
    query = parser("something light for my oily face, no perfume please", brands)
    assert query.required_attributes == ["Fragrance Free"]     # the rule parser alone misses this


def test_merge_explicit_comparison_pattern_wins():
    from sephra.query.llm_parser import merge_with_rules
    from sephra.query.rule_parser import parse_query
    from sephra.query.schema import StructuredQuery
    text = "Compare Product One vs Product Two"
    llm = StructuredQuery(original_query=text, semantic_query="x", parser="llm")
    merged = merge_with_rules(llm, parse_query(text))
    assert merged.intent.value == "product_comparison" and len(merged.product_names) == 2


def test_daily_limit_fails_fast_and_opens_circuit_breaker(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, json={"error": {"message": "Rate limit exceeded: free-models-per-day."}})

    client = _client(handler, retries=2)
    with pytest.raises(LLMRateLimitError, match="daily"):
        client.complete("s", "u")
    with pytest.raises(LLMRateLimitError):
        client.complete("s", "u")            # blocked: no HTTP request made
    assert len(calls) == 1
