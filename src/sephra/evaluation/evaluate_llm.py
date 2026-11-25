"""Live evaluation of the LLM layer (uses the real OpenRouter model).

Measures, on the same 20 evaluation queries:
  1. Query understanding: LLM parser vs. rule-based parser. Both feed the
     same hybrid retrieval and are scored with the same rubric labels.
     Also records how often the LLM fell back and which invented
     constraints the guards removed.
  2. Answer generation: status (generated / withheld / rate-limited),
     grounding-check results, attempts and latency.

LLM replies are cached in evaluation/results/llm_cache.json, so re-runs
use no free-tier quota. Delete the cache to re-query the model.

    .venv\\Scripts\\python.exe -m sephra.evaluation.evaluate_llm
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict

from sephra.evaluation.evaluate import (QUERIES_FILE, RESULTS_DIR, _automatic_checks,
                                        _outcome_metrics, _ranking_metrics)
from sephra.evaluation.relevance import RUBRICS, label_catalog
from sephra.generation.answer_generator import generate_answer
from sephra.generation.llm_client import LLMClient, LLMReply, get_llm_client
from sephra.logging_config import setup_logging
from sephra.query.llm_parser import LLMQueryParser
from sephra.query.rule_parser import parse_query
from sephra.retrieval.pipeline import SephraRetriever

CACHE_FILE = RESULTS_DIR / "llm_cache.json"
PAUSE_BETWEEN_CALLS_S = 3.5      # stay under free-tier per-minute limits


class CachingClient:
    """Wraps a real client; identical requests are answered from disk."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.model = inner.model
        self.cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}
        self.live_calls = 0

    def complete(self, system, user, **kwargs) -> LLMReply:
        key = hashlib.sha256(json.dumps([self.model, system, user, kwargs.get("json_schema")],
                                        sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            hit = self.cache[key]
            return LLMReply(hit["text"], hit["model"], hit["latency_s"], {"cached": True})
        time.sleep(PAUSE_BETWEEN_CALLS_S)
        self.live_calls += 1
        reply = self.inner.complete(system, user, **kwargs)
        self.cache[key] = {"text": reply.text, "model": reply.model, "latency_s": reply.latency_s}
        CACHE_FILE.write_text(json.dumps(self.cache, indent=1), encoding="utf-8")
        return reply


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    setup_logging(logging.WARNING)
    client = get_llm_client()
    if client is None:
        raise SystemExit("No LLM configured in .env")
    client = CachingClient(client)
    retriever = SephraRetriever()
    catalog = retriever.catalog
    brands = retriever.known_brands
    llm_parser = LLMQueryParser(client, catalog)
    labels = {qid: label_catalog(qid, catalog) for qid in RUBRICS}
    queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))

    rows = []
    for item in queries:
        print(f"{item['id']}: {item['query']}")
        row = {"id": item["id"], "query": item["query"]}
        for name, parser in (("rule", lambda t: parse_query(t, brands)),
                             ("llm", lambda t: llm_parser(t, brands))):
            start = time.perf_counter()
            structured = parser(item["query"])
            parse_s = time.perf_counter() - start
            response = retriever.search_structured(structured, mode="hybrid")
            ids = [r.product.product_id for r in response.products]
            metrics = (_ranking_metrics(item["id"], ids, None, labels[item["id"]])
                       if item["id"] in labels else _outcome_metrics(item["id"], response.status, ids))
            row[name] = {
                "parser_used": structured.parser,
                "constraints": structured.describe_constraints(),
                "skin_types": [s.value for s in structured.skin_types],
                "semantic_query": structured.semantic_query,
                "guard_removals": [n for n in structured.notes if n.startswith("Removed")],
                "status": response.status, "returned_ids": ids, "metrics": metrics,
                "checks": _automatic_checks(item["query"], ids, catalog, response),
                "parse_latency_s": round(parse_s, 2),
            }
            if name == "llm":
                answer = generate_answer(response, client)
                row["answer"] = {"status": answer.status, "attempts": answer.attempts,
                                 "latency_s": round(answer.latency_s, 2), "message": answer.message,
                                 "grounding": asdict(answer.grounding) if answer.grounding else None,
                                 "text": answer.text}
        rows.append(row)

    summary = {}
    for name in ("rule", "llm"):
        ranked = [r[name]["metrics"] for r in rows if r["id"] in RUBRICS]
        special = [r[name]["metrics"]["correct_outcome"] for r in rows
                   if "correct_outcome" in r[name]["metrics"]]
        summary[name] = {
            "P@5": _mean([m["P@5"] for m in ranked]),
            "strictP@5": _mean([m["strictP@5"] for m in ranked]),
            "nDCG@5": _mean([m["nDCG@5"] for m in ranked]),
            "special_cases": f"{sum(special):.0f}/{len(special)}",
            "hard_filter_violations": sum(r[name]["checks"]["hard_filter_violations"] for r in rows),
            "llm_fallbacks": sum(r[name]["parser_used"] == "rule_based" for r in rows) if name == "llm" else None,
            "guard_removals": sum(len(r[name]["guard_removals"]) for r in rows),
            "mean_parse_latency_s": _mean([r[name]["parse_latency_s"] for r in rows]),
        }
    answers = [r["answer"] for r in rows]
    statuses = {}
    for a in answers:
        statuses[a["status"]] = statuses.get(a["status"], 0) + 1
    summary["answers"] = {"status_counts": statuses,
                          "first_attempt_grounded": sum(a["status"] == "generated" and a["attempts"] == 1
                                                        for a in answers),
                          "mean_latency_s_generated": _mean([a["latency_s"] for a in answers
                                                             if a["status"] == "generated"])}
    summary["live_llm_calls_this_run"] = client.live_calls

    (RESULTS_DIR / "llm_evaluation.json").write_text(
        json.dumps({"summary": summary, "per_query": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
