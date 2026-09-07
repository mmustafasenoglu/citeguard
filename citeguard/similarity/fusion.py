"""Reciprocal Rank Fusion (RRF) for candidate retrieval.

Combines ranked lists from multiple retrieval signals (fingerprint,
TF-IDF, semantic) into a single fused ranking using the RRF algorithm.

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods" (SIGIR 2009).
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists:
        Each element is a list of document indices ordered by relevance
        (most relevant first).  Lists may have different lengths and
        may contain different indices.
    k:
        Constant that controls the influence of lower-ranked documents.
        Higher *k* reduces the impact of rank position.  Default 60
        follows the original paper.

    Returns
    -------
    list[tuple[int, float]]
        Fused list of ``(doc_index, rrf_score)`` sorted descending by
        score.  Ties are broken by document index (lower first).
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_idx in enumerate(ranked):
            rrf_score = 1.0 / (k + rank + 1)  # rank is 0-indexed
            scores[doc_idx] = scores.get(doc_idx, 0.0) + rrf_score

    result = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return result
