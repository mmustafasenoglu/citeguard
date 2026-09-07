"""Hybrid reranker for similarity matches.

Separate from RRF (which handles candidate retrieval).
The reranker produces a final weighted score from raw feature values
for each candidate that passed RRF filtering.

Contract:
- ``semantic_rerank_score`` is an internal ranking value (may be
  calibrated/normalized).
- ``semantic_similarity_raw`` is the raw cosine for evidence/report.
"""

from __future__ import annotations

from citeguard.similarity.models import SimilarityConfig


def compute_rerank_score(
    exact_overlap: float,
    lexical_similarity: float,
    semantic_raw: float,
    config: SimilarityConfig,
) -> float:
    """Compute a combined rerank score from individual feature values.

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
    float
        Weighted rerank score.  Not normalized to any fixed range.
    """
    # Normalize raw cosine to [0, 1] for internal ranking only
    semantic_norm = max(0.0, min(1.0, (semantic_raw + 1.0) / 2.0))

    return (
        config.weight_fingerprint * exact_overlap
        + config.weight_tfidf * lexical_similarity
        + config.weight_semantic * semantic_norm
    )
