"""Citeguard similarity engine — v0.3.

Public API for deterministic similarity computation:
- Fingerprinting (BLAKE2b hashing, ordered shingles, winnowing)
- Lexical similarity (TF-IDF cosine, char-n-gram Jaccard)
- Match-type classification
- Engine orchestration
"""

from citeguard.similarity.fingerprint import (
    exact_overlap,
    find_matching_segments,
    generate_shingles,
    stable_hash,
    winnow,
)
from citeguard.similarity.lexical import (
    build_tfidf_index,
    char_ngram_jaccard,
    classify_match_type,
    compute_cosine_similarity,
    normalize_turkish,
)
from citeguard.similarity.models import (
    Fingerprint,
    FingerprintPoint,
    MatchType,
    RiskLevel,
    Shingle,
    SimilarityConfig,
    SimilarityEngineResult,
    SimilarityMatch,
    SimilarityMetrics,
    SimilarityResult,
)

__all__ = [
    "Fingerprint",
    "FingerprintPoint",
    "MatchType",
    "RiskLevel",
    "Shingle",
    "similarity_index",
    "build_tfidf_index",
    "char_ngram_jaccard",
    "classify_match_type",
    "compute_cosine_similarity",
    "exact_overlap",
    "find_matching_segments",
    "generate_shingles",
    "normalize_turkish",
    "winnow",
    "stable_hash",
    "SimilarityConfig",
    "SimilarityEngineResult",
    "SimilarityMatch",
    "SimilarityMetrics",
    "SimilarityResult",
]