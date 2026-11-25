"""Benchmark runner: compares retrieval configurations on the evaluation set.

Systems compared
  baseline_original  the untouched original retrieve.py (results captured
                     before the refactor in results/baseline_original.json)
  dense              hard filters + dense retrieval
  hybrid             hard filters + dense + BM25 fused with RRF
  hybrid_rerank_*    hybrid + cross-encoder reranking (minilm / bge-base)

Run (from any folder):
    .venv\\Scripts\\python.exe -m sephra.evaluation.evaluate
    .venv\\Scripts\\python.exe -m sephra.evaluation.evaluate --systems dense hybrid

Outputs evaluation/results/evaluation_results.json and
evaluation/results/evaluation_summary.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from sephra import config
from sephra.catalog import Catalog
from sephra.embeddings import embed_query
from sephra.evaluation.metrics import (capped_recall_at_k, ndcg_at_k, precision_at_k,
                                       reciprocal_rank, strict_precision_at_k)
from sephra.evaluation.relevance import (EXPECTED_NAMED_PRODUCTS, EXPECTED_NO_RESULTS, RUBRICS,
                                         UNLABELLED, label_catalog)
from sephra.logging_config import setup_logging
from sephra.query.filters import apply_hard_filters, product_violations
from sephra.query.rule_parser import parse_query
from sephra.retrieval.pipeline import SearchResponse, SephraRetriever
from sephra.retrieval.ranking import generate_candidates

logger = logging.getLogger(__name__)

RESULTS_DIR = config.EVALUATION_DIR / "results"
RUBRIC_GRADES: dict[str, dict[str, int]] = {}   # filled in main()
QUERIES_FILE = config.EVALUATION_DIR / "queries.json"
BASELINE_FILE = RESULTS_DIR / "baseline_original.json"
K = 5
RECALL_K = 50

SYSTEMS = {
    "dense": ("dense", None),
    "hybrid": ("hybrid", None),
    "hybrid_rerank_minilm": ("hybrid_rerank", "minilm"),
    "hybrid_rerank_bge-base": ("hybrid_rerank", "bge-base"),
}


@dataclass
class QueryResult:
    query_id: str
    query_type: str
    status: str
    returned_ids: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    checks: dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0


def _automatic_checks(query_text: str, ids: list[str], catalog: Catalog,
                      response: SearchResponse | None,
                      baseline_reviews: dict[str, list[str]] | None = None) -> dict[str, int]:
    """Label-free correctness checks. Counts of problems (0 is good)."""
    intended = parse_query(query_text, {p.brand for p in catalog.products.values()})
    checks = {"hard_filter_violations": 0, "duplicate_products": 0, "duplicate_families": 0,
              "misattributed_reviews": 0, "zero_review_without_note": 0,
              "products_with_critical_available": 0, "critical_shown": 0}
    if intended.intent.value == "product_recommendation":
        checks["hard_filter_violations"] = sum(
            1 for pid in ids if product_violations(catalog.products[pid], intended))
    checks["duplicate_products"] = len(ids) - len(set(ids))
    families = [catalog.products[pid].family_id for pid in ids]
    checks["duplicate_families"] = len(families) - len(set(families))

    if response is not None:
        for result in response.products:
            pid = result.product.product_id
            checks["misattributed_reviews"] += sum(
                1 for s in result.evidence.selected if s.review.product_id != pid)
            if result.evidence.loaded_review_count == 0 and not result.evidence.notes:
                checks["zero_review_without_note"] += 1
            stats = result.evidence.stats
            if stats and stats.critical > 0:
                checks["products_with_critical_available"] += 1
                if any(s.review.sentiment == "critical" for s in result.evidence.selected):
                    checks["critical_shown"] += 1
    elif baseline_reviews is not None:
        for pid, review_ids in baseline_reviews.items():
            checks["misattributed_reviews"] += sum(
                1 for rid in review_ids if catalog.reviews[rid].product_id != pid)
            stats_critical = sum(1 for rid in catalog.reviews_by_product.get(pid, [])
                                 if catalog.reviews[rid].sentiment == "critical")
            if stats_critical:
                checks["products_with_critical_available"] += 1
                if any(catalog.reviews[rid].sentiment == "critical" for rid in review_ids):
                    checks["critical_shown"] += 1
    return checks


def _ranking_metrics(query_id: str, ids: list[str], pool: list[str] | None,
                     labels: dict[str, int]) -> dict[str, float]:
    metrics = {"P@5": precision_at_k(ids, labels, K), "strictP@5": strict_precision_at_k(ids, labels, K),
               "MRR": reciprocal_rank(ids, labels),
               "nDCG@5": ndcg_at_k(ids, labels, K)}
    if pool is not None:
        metrics[f"cappedRecall@{RECALL_K}"] = capped_recall_at_k(pool, labels, RECALL_K)
    return metrics


def _outcome_metrics(query_id: str, status: str, ids: list[str]) -> dict[str, float]:
    if query_id in EXPECTED_NO_RESULTS:
        return {"correct_outcome": float(status == "no_results" and not ids)}
    if query_id in EXPECTED_NAMED_PRODUCTS:
        return {"correct_outcome": float(ids == EXPECTED_NAMED_PRODUCTS[query_id])}
    return {}


def evaluate_baseline(queries: list[dict], catalog: Catalog,
                      labels: dict[str, dict[str, int]]) -> list[QueryResult]:
    baseline = {r["id"]: r for r in json.loads(BASELINE_FILE.read_text(encoding="utf-8"))}
    results = []
    for item in queries:
        record = baseline[item["id"]]
        ids = [p["product_id"] for p in record["products"]]
        reviews = {p["product_id"]: [r["review_id"] for r in p["reviews"]] for p in record["products"]}
        result = QueryResult(item["id"], item["type"], "ok", ids, latency_s=record["latency_seconds"])
        if item["id"] in labels:
            result.metrics = _ranking_metrics(item["id"], ids, None, labels[item["id"]])
        else:
            # The original system had no notion of "no results" or named products.
            result.metrics = _outcome_metrics(item["id"], "ok", ids)
        result.checks = _automatic_checks(item["query"], ids, catalog, None, reviews)
        results.append(result)
    return results


def evaluate_system(name: str, retriever: SephraRetriever, queries: list[dict],
                    labels: dict[str, dict[str, int]]) -> list[QueryResult]:
    mode, reranker = SYSTEMS[name]
    catalog = retriever.catalog
    results = []
    for item in queries:
        start = time.perf_counter()
        response = retriever.search(item["query"], mode=mode, reranker=reranker)
        latency = time.perf_counter() - start
        ids = [r.product.product_id for r in response.products]
        result = QueryResult(item["id"], item["type"], response.status, ids, latency_s=latency)

        if item["id"] in labels:
            # Candidate pool BEFORE diversity truncation, for recall@50.
            query = response.query
            filters = apply_hard_filters(query, catalog)
            pool = [c.product_id for c in generate_candidates(
                query, filters, embed_query(query.semantic_query), catalog,
                retriever.settings, mode, reranker)] if not filters.is_empty else []
            result.metrics = _ranking_metrics(item["id"], ids, pool, labels[item["id"]])
        else:
            result.metrics = _outcome_metrics(item["id"], response.status, ids)
        result.checks = _automatic_checks(item["query"], ids, catalog, response)
        results.append(result)
    return results


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarise(all_results: dict[str, list[QueryResult]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name, results in all_results.items():
        ranked = [r for r in results if r.query_id in RUBRICS]
        row: dict[str, float] = {}
        for metric in ["P@5", "strictP@5", "MRR", "nDCG@5", f"cappedRecall@{RECALL_K}"]:
            values = [r.metrics[metric] for r in ranked if metric in r.metrics]
            if values:
                row[metric] = _mean(values)
        graded = [r for r in ranked if any(g == 2 for g in RUBRIC_GRADES.get(r.query_id, {}).values())]
        if graded:
            row["strictP@5_graded_only"] = _mean([r.metrics["strictP@5"] for r in graded])
        outcomes = [r.metrics["correct_outcome"] for r in results if "correct_outcome" in r.metrics]
        row["special_cases_correct"] = sum(outcomes)
        row["special_cases_total"] = len(outcomes)
        for check in ["hard_filter_violations", "duplicate_products", "duplicate_families",
                      "misattributed_reviews", "zero_review_without_note",
                      "products_with_critical_available", "critical_shown"]:
            row[check] = sum(r.checks.get(check, 0) for r in results)
        row["products_returned"] = sum(len(r.returned_ids) for r in results)
        latencies = sorted(r.latency_s for r in results)
        row["latency_mean_s"] = _mean(latencies)
        row["latency_p95_s"] = latencies[int(0.95 * (len(latencies) - 1))]
        summary[name] = row
    return summary


def _markdown(summary: dict[str, dict[str, float]], all_results: dict[str, list[QueryResult]]) -> str:
    lines = ["# Sephra retrieval evaluation (auto-generated)", "",
             f"Ranked queries with rubric labels: {len(RUBRICS)}. K = {K}.", "",
             "| System | P@5 | strict P@5* | MRR | nDCG@5 | capped R@50 | special cases | filter violations | "
             "dup. families | misattributed reviews | critical shown / available | mean latency (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, row in summary.items():
        recall = row.get(f"cappedRecall@{RECALL_K}")
        lines.append(
            f"| {name} | {row['P@5']:.3f} | {row.get('strictP@5_graded_only', float('nan')):.3f} | {row['MRR']:.3f} | {row['nDCG@5']:.3f} | "
            f"{'n/a' if recall is None else f'{recall:.3f}'} | "
            f"{row['special_cases_correct']:.0f}/{row['special_cases_total']} | "
            f"{row['hard_filter_violations']:.0f} | {row['duplicate_families']:.0f} | "
            f"{row['misattributed_reviews']:.0f} | "
            f"{row['critical_shown']:.0f}/{row['products_with_critical_available']:.0f} | "
            f"{row['latency_mean_s']:.3f} |")
    lines += ["", "*strict P@5 counts only grade-2 (highly relevant) products, averaged over the "
              "queries whose rubric has a grade-2 level.", "", "## Per-query nDCG@5", "",
              "| Query | " + " | ".join(all_results) + " |",
              "|---|" + "---|" * len(all_results)]
    query_ids = [r.query_id for r in next(iter(all_results.values()))]
    for qid in query_ids:
        cells = []
        for results in all_results.values():
            r = next(x for x in results if x.query_id == qid)
            if "nDCG@5" in r.metrics:
                cells.append(f"{r.metrics['nDCG@5']:.2f}")
            elif "correct_outcome" in r.metrics:
                cells.append("correct" if r.metrics["correct_outcome"] else "WRONG")
            else:
                cells.append(f"({r.status})")
        lines.append(f"| {qid} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()
    setup_logging(logging.WARNING)

    queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    retriever = SephraRetriever()
    labels = {qid: label_catalog(qid, retriever.catalog) for qid in RUBRICS}
    RUBRIC_GRADES.update(labels)

    # Warm-up so that one-off model loading is not counted as query latency.
    retriever.search("warm up moisturizer", mode="hybrid")
    for name in args.systems:
        if SYSTEMS[name][1]:
            retriever.search("warm up moisturizer", mode="hybrid_rerank", reranker=SYSTEMS[name][1])

    all_results: dict[str, list[QueryResult]] = {}
    if not args.no_baseline and BASELINE_FILE.exists():
        all_results["baseline_original"] = evaluate_baseline(queries, retriever.catalog, labels)
    for name in args.systems:
        print(f"Evaluating {name} ...")
        all_results[name] = evaluate_system(name, retriever, queries, labels)

    summary = summarise(all_results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "relevant_counts": {qid: {"relevant": len(l), "highly_relevant": sum(g == 2 for g in l.values())}
                            for qid, l in labels.items()},
        "per_query": {name: [r.__dict__ for r in results] for name, results in all_results.items()},
        "unlabelled": sorted(UNLABELLED),
    }
    (RESULTS_DIR / "evaluation_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = _markdown(summary, all_results)
    (RESULTS_DIR / "evaluation_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
