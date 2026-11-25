"""Reciprocal Rank Fusion (RRF).

Dense cosine similarities (about 0.6-0.9) and BM25 scores (0 to 30+) are on
completely different scales, so adding them directly is meaningless.
RRF ignores raw scores and only uses each list's RANKS:

    rrf(d) = sum over rankings r of  1 / (k + rank_r(d))

A product ranked highly by either retriever gets a good fused score, and
one ranked well by BOTH gets the best score. k = 60 (Cormack et al., 2009)
damps the difference between rank 1 and rank 2 so no single list dominates.
"""

from __future__ import annotations


def reciprocal_rank_fusion(rankings: dict[str, list[str]], k: int = 60
                           ) -> list[tuple[str, float, dict[str, int]]]:
    """Fuse several ranked id lists.

    Args:
        rankings: name -> ids in rank order, e.g. {"dense": [...], "bm25": [...]}.
        k: RRF damping constant.

    Returns:
        (id, fused_score, {ranking_name: 1-based rank}) sorted best first.
    """
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ids in rankings.items():
        for position, item_id in enumerate(ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + position)
            ranks.setdefault(item_id, {})[name] = position
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return [(item_id, score, ranks[item_id]) for item_id, score in ordered]
