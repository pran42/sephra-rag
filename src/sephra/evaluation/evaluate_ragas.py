"""RAGAS evaluation of Sephra.

Part A: retrieval, no LLM needed.
  * IDBasedContextPrecision for PRODUCTS: share of the top-5 product IDs
    that are in the reference set (rubric labels, grade >= 1; and a
    "strict" variant with grade 2 only).
  * IDBasedContextRecall for NAMED-PRODUCT queries (Q09, Q14), where the
    reference set is small and exact. For broad queries with hundreds of
    relevant products, recall@5 is meaningless and is not reported.
  * IDBasedContextPrecision for REVIEWS: share of retrieved review IDs that
    really belong to the product they are shown under (review attribution).

Part B (--llm): generation, uses an LLM judge.
  * Faithfulness: share of the answer's claims supported by the contexts.
  * AnswerRelevancy: how well the answer addresses the question.
  Runs on the cached LLM answers from evaluate_llm. The judge is the same
  free ERNIE model used to write the answers (self-evaluation bias; a
  different judge would be better). Free-tier rate limits are handled.

    .venv\\Scripts\\python.exe -m sephra.evaluation.evaluate_ragas          # Part A
    .venv\\Scripts\\python.exe -m sephra.evaluation.evaluate_ragas --llm    # Part A + B
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import warnings

from sephra.evaluation.evaluate import BASELINE_FILE, QUERIES_FILE, RESULTS_DIR
from sephra.evaluation.relevance import EXPECTED_NAMED_PRODUCTS, RUBRICS, label_catalog
from sephra.logging_config import setup_logging
from sephra.retrieval.pipeline import SephraRetriever

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


async def _id_scores(retrieved: list[str], reference: list[str]) -> tuple[float, float]:
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall

    sample = SingleTurnSample(retrieved_context_ids=retrieved or ["<none>"],
                              reference_context_ids=reference or ["<none>"])
    precision = await IDBasedContextPrecision().single_turn_ascore(sample)
    recall = await IDBasedContextRecall().single_turn_ascore(sample)
    return float(precision), float(recall)


async def part_a(retriever: SephraRetriever, queries: list[dict]) -> dict:
    catalog = retriever.catalog
    labels = {qid: label_catalog(qid, catalog) for qid in RUBRICS}
    baseline = {r["id"]: [p["product_id"] for p in r["products"]]
                for r in json.loads(BASELINE_FILE.read_text(encoding="utf-8"))}

    systems: dict[str, dict[str, list[str]]] = {"baseline_original": baseline}
    review_pairs: list[tuple[list[str], list[str]]] = []
    for mode in ("dense", "hybrid"):
        systems[mode] = {}
        for item in queries:
            response = retriever.search(item["query"], mode=mode)
            systems[mode][item["id"]] = [r.product.product_id for r in response.products]
            if mode == "hybrid":
                for result in response.products:
                    retrieved = [s.review.review_id for s in result.evidence.selected]
                    if retrieved:
                        reference = catalog.reviews_by_product.get(result.product.product_id, [])
                        review_pairs.append((retrieved, reference))

    report: dict = {}
    for name, results in systems.items():
        precision, strict, named_recall, named_precision = [], [], [], []
        for qid, ids in results.items():
            if qid in labels:
                relevant = [pid for pid, grade in labels[qid].items() if grade >= 1]
                highly = [pid for pid, grade in labels[qid].items() if grade >= 2]
                precision.append((await _id_scores(ids, relevant))[0])
                if highly:
                    strict.append((await _id_scores(ids, highly))[0])
            if qid in EXPECTED_NAMED_PRODUCTS:
                p, r = await _id_scores(ids, EXPECTED_NAMED_PRODUCTS[qid])
                named_precision.append(p)
                named_recall.append(r)
        report[name] = {
            "product_context_precision": _mean(precision),
            "product_context_precision_strict": _mean(strict),
            "named_product_context_precision": _mean(named_precision),
            "named_product_context_recall": _mean(named_recall),
        }
    review_precision = [(await _id_scores(r, ref))[0] for r, ref in review_pairs]
    report["hybrid"]["review_attribution_precision"] = _mean(review_precision)
    report["hybrid"]["reviews_checked"] = sum(len(r) for r, _ in review_pairs)
    return report


async def part_b(max_samples: int) -> dict:
    """LLM-judged Faithfulness + AnswerRelevancy on cached generated answers."""
    from dotenv import load_dotenv
    from openai import AsyncOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    from sephra import config
    from sephra.evaluation.evaluate_llm import CachingClient
    from sephra.generation.context import product_block
    from sephra.generation.llm_client import get_llm_client
    from sephra.query.llm_parser import LLMQueryParser

    load_dotenv(config.PROJECT_ROOT / ".env")
    judge_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                               api_key=os.environ["OPENROUTER_API_KEY"], max_retries=1)
    # Reasoning tokens count toward the output limit; the default was too small
    # (faithfulness returned IncompleteOutputException in the first test).
    llm = llm_factory(os.environ["OPENROUTER_MODEL"], provider="openai", client=judge_client,
                      max_tokens=8000)
    embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")
    faithfulness, relevancy = Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=embeddings)

    llm_eval = json.loads((RESULTS_DIR / "llm_evaluation.json").read_text(encoding="utf-8"))
    generated = [r for r in llm_eval["per_query"]
                 if r["answer"]["status"] == "generated"][:max_samples]

    # Rebuild exactly the contexts the answers were written from (parser replies are cached).
    retriever = SephraRetriever()
    parser = LLMQueryParser(CachingClient(get_llm_client()), retriever.catalog)
    rows = []
    for row in generated:
        query = parser(row["query"], retriever.known_brands)
        response = retriever.search_structured(query, mode="hybrid")
        contexts = [product_block(r) for r in response.products]
        result = {"id": row["id"]}
        for name, metric, kwargs in (
                ("faithfulness", faithfulness, dict(user_input=row["query"], response=row["answer"]["text"],
                                                    retrieved_contexts=contexts)),
                ("answer_relevancy", relevancy, dict(user_input=row["query"], response=row["answer"]["text"]))):
            try:
                result[name] = float((await metric.ascore(**kwargs)).value)
            except Exception as exc:          # rate limit / judge failure: record, continue
                result[name] = None
                result[f"{name}_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        print(row["id"], {k: v for k, v in result.items() if not k.endswith("error")})
        rows.append(result)
    return {"samples": len(rows),
            "faithfulness": _mean([r.get("faithfulness") for r in rows]),
            "answer_relevancy": _mean([r.get("answer_relevancy") for r in rows]),
            "scored_faithfulness": sum(r.get("faithfulness") is not None for r in rows),
            "scored_answer_relevancy": sum(r.get("answer_relevancy") is not None for r in rows),
            "judge": os.environ["OPENROUTER_MODEL"], "per_sample": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="also run LLM-judged metrics (uses quota)")
    parser.add_argument("--max-samples", type=int, default=13)
    args = parser.parse_args()
    setup_logging(logging.WARNING)

    queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    output = {"retrieval": asyncio.run(part_a(SephraRetriever(), queries))}
    print(json.dumps(output["retrieval"], indent=2))
    if args.llm:
        output["generation"] = asyncio.run(part_b(args.max_samples))
        print(json.dumps({k: v for k, v in output["generation"].items() if k != "per_sample"}, indent=2))
    out_file = RESULTS_DIR / "ragas_results.json"
    previous = json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() else {}
    previous.update(output)
    out_file.write_text(json.dumps(previous, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
