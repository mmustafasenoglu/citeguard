"""Hybrid reranker for similarity matches.

Separate from RRF (which handles candidate retrieval).
The reranker produces a final weighted score from raw feature values
for each candidate that passed RRF filtering.

Contract:
- ``semantic_rerank_score`` is the semantic *component* only.
- ``ranking_score`` is the final hybrid score (exact + lexical + semantic).
- Semantic normalization uses ``max(0, cosine)`` — no artificial 0.5 base.
"""

from __future__ import annotations

from citeguard.similarity.models import SimilarityConfig


def compute_ranking_score(
    exact_overlap: float,
    lexical_similarity: float,
    semantic_raw: float,
    config: SimilarityConfig,
) -> tuple[float, float]:
    """Compute the semantic component and final ranking score.

    Parameters
    ----------
    exact_overlap:
        Fingerprint overlap in [0, 1].
    lexical_similarity:
        TF-IDF cosine or char-n-gram Jaccard in [0, 1].
    semantic_raw:
        Raw semantic cosine in [-1, 1].
    config:
        Configuration with weight_fingerprint, weight_tfidf,
        weight_semantic.

    Returns
    -------
    (semantic_component, ranking_score)
        semantic_component: max(0, cosine) — semantic-only value.
        ranking_score: weighted hybrid for final ordering.
    """
    semantic_component = max(0.0, semantic_raw)

    ranking_score = (
        config.weight_fingerprint * exact_overlap
        + config.weight_tfidf * lexical_similarity
        + config.weight_semantic * semantic_component
    )

    return semantic_component, ranking_score
