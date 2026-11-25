"""Grounded answer generation.

1. Build the evidence context (generation/context.py).
2. Ask the LLM to answer using ONLY that context (generation/prompts.py).
3. Run automatic grounding checks (generation/grounding.py).
4. If the answer contains a price, rating or product ID that is not in the
   evidence, retry once and name the unsupported facts. If it still fails,
   WITHHOLD the answer: the user sees the retrieved evidence and an
   explanation, never an answer known to contain unsupported facts.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal

from sephra.generation.context import build_context
from sephra.generation.grounding import GroundingReport, check_answer_grounding
from sephra.generation.llm_client import LLMClient, LLMError, LLMRateLimitError
from sephra.generation.prompts import build_answer_messages
from sephra.retrieval.pipeline import SearchResponse

logger = logging.getLogger(__name__)

AnswerStatus = Literal["generated", "withheld", "skipped", "rate_limited", "error"]


@dataclass
class AnswerResult:
    status: AnswerStatus
    text: str | None
    model: str | None
    message: str
    grounding: GroundingReport | None = None
    attempts: int = 0
    latency_s: float = 0.0


def _problems(report: GroundingReport) -> str:
    parts = []
    if report.unknown_prices:
        parts.append("prices not in CONTEXT: " + ", ".join(report.unknown_prices))
    if report.unknown_ratings:
        parts.append("ratings not in CONTEXT: " + ", ".join(report.unknown_ratings))
    if report.unknown_product_ids:
        parts.append("product IDs not in CONTEXT: " + ", ".join(report.unknown_product_ids))
    if report.recommends_when_no_results:
        parts.append("recommended products although none were retrieved")
    return "; ".join(parts)


def generate_answer(response: SearchResponse, client: LLMClient | None) -> AnswerResult:
    if client is None:
        return AnswerResult("skipped", None, None, "No LLM configured (retrieval-only mode).")
    if response.status != "ok" or not response.products:
        # The retrieval message already explains this deterministically, so an
        # LLM call would only add risk and use up free-tier quota.
        return AnswerResult("skipped", None, client.model, response.message)

    system, user = build_answer_messages(build_context(response))
    start = time.perf_counter()
    report: GroundingReport | None = None
    for attempt in (1, 2):
        try:
            reply = client.complete(system, user, temperature=0.2, max_tokens=4000)
        except LLMRateLimitError as exc:
            return AnswerResult("rate_limited", None, client.model,
                                f"The free LLM tier is rate-limited right now ({exc}). "
                                "The retrieved evidence is shown below.", report, attempt,
                                time.perf_counter() - start)
        except LLMError as exc:
            return AnswerResult("error", None, client.model, f"LLM call failed: {exc}", report,
                                attempt, time.perf_counter() - start)
        report = check_answer_grounding(reply.text, response)
        if report.is_grounded:
            return AnswerResult("generated", reply.text, reply.model, "Answer passed grounding checks.",
                                report, attempt, time.perf_counter() - start)
        logger.warning("Ungrounded answer (attempt %d): %s", attempt, _problems(report))
        user = (user + "\n\nYour previous answer contained facts that are NOT in the CONTEXT ("
                + _problems(report) + "). Rewrite it using only values that appear in the CONTEXT.")
    return AnswerResult("withheld", None, client.model,
                        "The generated answer was withheld because it contained facts that are not "
                        f"in the retrieved evidence ({_problems(report)}).", report, 2,
                        time.perf_counter() - start)
