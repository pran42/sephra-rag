"""Standard ranking metrics (pure functions, unit-tested).

All functions take the ranked list of product ids a system returned and a
dict of graded labels {product_id: grade}; missing ids have grade 0.
"""

from __future__ import annotations

import math


def precision_at_k(ranked: list[str], labels: dict[str, int], k: int) -> float:
    """Share of the top-k results that are relevant (grade >= 1).
    Divides by k, so returning fewer than k results is penalised."""
    return sum(1 for pid in ranked[:k] if labels.get(pid, 0) > 0) / k


def strict_precision_at_k(ranked: list[str], labels: dict[str, int], k: int) -> float:
    """Like precision@k but only grade-2 ("highly relevant") items count.
    More discriminating when every system already satisfies the hard filters."""
    return sum(1 for pid in ranked[:k] if labels.get(pid, 0) >= 2) / k


def capped_recall_at_k(ranked: list[str], labels: dict[str, int], k: int) -> float:
    """Relevant items found in the top k / min(k, number of relevant items).

    Plain recall@k is misleading when hundreds of products are relevant
    (finding 5 of 284 fragrance-free skincare products gives "recall" 0.02
    even for a perfect top 5). Capping the denominator at k gives 1.0 to
    any list whose top k is entirely relevant.
    """
    relevant = sum(1 for grade in labels.values() if grade > 0)
    if relevant == 0:
        return 0.0
    hits = sum(1 for pid in ranked[:k] if labels.get(pid, 0) > 0)
    return hits / min(k, relevant)


def reciprocal_rank(ranked: list[str], labels: dict[str, int]) -> float:
    for position, pid in enumerate(ranked, start=1):
        if labels.get(pid, 0) > 0:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: list[str], labels: dict[str, int], k: int) -> float:
    """Graded nDCG with gain 2^grade - 1 and log2 position discount.
    The ideal ranking is computed from ALL labelled products."""
    def dcg(grades: list[int]) -> float:
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))

    actual = dcg([labels.get(pid, 0) for pid in ranked[:k]])
    ideal = dcg(sorted(labels.values(), reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0
