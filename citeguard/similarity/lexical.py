"""Lexical similarity functions for Citeguard v0.3.

Provides TF-IDF-based cosine similarity, character-n-gram Jaccard, and
Turkish-aware text normalization for retrieval. All functions are deterministic
and dependency-light; heavy sklearn usage is confined to the ``build_tfidf_index``
entry point.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from citeguard.similarity.models import MatchType

# ---------------------------------------------------------------------------
# Turkish-aware normalization (retrieval-only, never replaces original text)
# ---------------------------------------------------------------------------

def normalize_turkish(text: str) -> str:
    """Lowercase + whitespace normalize with Turkish dotless/i handling.

    Transformations:
    - ``İ`` → ``i``, ``I`` → ``ı``
    - ``str.lower()`` for remaining Unicode (ç, ğ, ö, ş, ü are correct)
    - Whitespace collapsed to single space

    This is *intentionaly* used only for retrieval / matching.
    Original text is never replaced in the model.
    """
    text = text.replace("İ", "i").replace("I", "ı")
    lowered = text.lower()
    collapsed = re.sub(r"\s+", " ", lowered).strip()
    return collapsed


# ---------------------------------------------------------------------------
# TF-IDF index building
# ---------------------------------------------------------------------------

def build_tfidf_index(
    documents: list[str],
    *,
    analyzer: str | None = None,
    ngram_range: tuple[int, int] = (1, 3),
) -> tuple:
    """Fit a TfidfVectorizer on *documents* and return (vectorizer, matrix).

    Parameters
    ----------
    documents:
        List of document texts (one per document).
    analyzer:
        ``"turkish```` for word-level Turkish normalization, else ``None``
        (default Python tokenizer). When ``"turkish```` the analyzer lowercases
        with ``İ``/``I`` handling and preserves diacritics ``ç, ğ, ö, ş, ü``.
    ngram_range:
        Min/max n-gram size for the vectorizer.

    Returns
    -------
    tuple[TfidfVectorizer, scipy.sparse.csr_matrix]
    """
    if analyzer == "turkish":
        # Custom analyzer that applies Turkish-normalized tokenization.
        # We normalize *outside* the tokenizer to avoid scikit-learn's fit
        # re-normalizing and losing our intentional İ/I substitutions.
        def _turkish_analyzer(_text: str):
            return normalize_turkish(_text).split()

        vectorizer = TfidfVectorizer(analyzer=_turkish_analyzer, ngram_range=ngram_range)
    else:
        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=ngram_range)

    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


def compute_cosine_similarity(
    query: str,
    vectorizer,
    matrix: np.ndarray,
) -> list[tuple[int, float]]:
    """Compute cosine similarities between *query* and every row of *matrix*.

    Returns list of (doc_index, score) sorted descending by score.
    """
    from scipy.sparse import csr_matrix

    if not isinstance(matrix, csr_matrix):
        matrix = matrix.tocsr()

    # Transform query using the same vectorizer
    query_vec = vectorizer.transform([query])
    if not isinstance(query_vec, csr_matrix):
        query_vec = query_vec.tocsr()

    # Compute cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity

    scores = cosine_similarity(query_vec, matrix).flatten()
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: -x[1])
    return indexed


# ---------------------------------------------------------------------------
# Character-n-gram Jaccard
# ---------------------------------------------------------------------------

def char_ngram_jaccard(text1: str, text2: str, n: int = 3) -> float:
    """Jaccard similarity on character n-grams.

    Parameters
    ----------
    text1, text2:
        Input texts.
    n:
        N-gram size.

    Returns
    -------
    float
        Jaccard index in [0, 1].
    """
    if not text1 or not text2:
        return 0.0

    # Normalize both texts
    t1 = normalize_turkish(text1)
    t2 = normalize_turkish(text2)

    if len(t1) < n or len(t2) < n:
        # Fallback to word Jaccard if too short
        w1 = set(t1.split())
        w2 = set(t2.split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)

    # Generate char n-grams
    n1 = {t1[i : i + n] for i in range(len(t1) - n + 1)}
    n2 = {t2[i : i + n] for i in range(len(t2) - n + 1)}

    intersection = len(n1 & n2)
    union = len(n1 | n2)
    return intersection / union if union else 0.0


# ---------------------------------------------------------------------------
# Match-type classification
# ---------------------------------------------------------------------------

def classify_match_type(
    exact_overlap: float,
    lexical_similarity: float,
    *,
    exact_threshold: float = 0.95,
    near_duplicate_threshold: float = 0.70,
    lexical_overlap_threshold: float = 0.40,
    semantic_score: float | None = None,
    semantic_threshold: float | None = None,
) -> MatchType:
    """Classify match type based on exact, lexical, and optionally semantic scores.

    Parameters
    ----------
    exact_overlap:
        BLAKE2-based Jaccard or shingle overlap in [0, 1].
    lexical_similarity:
        TF-IDF cosine similarity in [0, 1].
    exact_threshold:
        Threshold for "exact" classification.
    near_duplicate_threshold:
        Threshold for "near_duplicate" classification.
    lexical_overlap_threshold:
        Threshold for "lexical_overlap" classification.
    semantic_score:
        Optional raw semantic cosine.  When provided and all lexical
        thresholds are missed, this is checked for SEMANTIC_OVERLAP.
    semantic_threshold:
        Optional threshold for semantic classification.  Only effective
        when ``semantic_score`` is also provided.

    Returns
    -------
    MatchType
    """
    if exact_overlap >= exact_threshold:
        return MatchType.EXACT
    if exact_overlap >= near_duplicate_threshold:
        return MatchType.NEAR_DUPLICATE
    if lexical_similarity >= lexical_overlap_threshold:
        return MatchType.LEXICAL_OVERLAP
    if (
        semantic_score is not None
        and semantic_threshold is not None
        and semantic_score >= semantic_threshold
    ):
        return MatchType.SEMANTIC_OVERLAP
    return MatchType.UNMATCHED