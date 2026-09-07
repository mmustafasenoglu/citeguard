"""Tests for RRF fusion and semantic retrieval."""

from __future__ import annotations

import numpy as np

from citeguard.similarity.fusion import reciprocal_rank_fusion
from citeguard.similarity.index import SimilarityIndex
from citeguard.similarity.models import Fingerprint

# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def test_rrf_single_list() -> None:
    """Single list → scores are 1/(k + rank+1)."""
    result = reciprocal_rank_fusion([[0, 1, 2]], k=60)
    assert len(result) == 3
    assert result[0][0] == 0  # highest rank → highest score
    assert result[0][1] > result[1][1] > result[2][1]


def test_rrf_two_lists_merge() -> None:
    """Two lists with overlapping items → higher combined score."""
    result = reciprocal_rank_fusion(
        [[0, 1, 2], [1, 0, 3]], k=60,
    )
    doc_ids = [idx for idx, _ in result]
    # Doc 0 and 1 appear in both lists → should rank highest
    assert doc_ids[0] in (0, 1)
    assert doc_ids[1] in (0, 1)


def test_rrf_empty_lists() -> None:
    result = reciprocal_rank_fusion([], k=60)
    assert result == []


def test_rrf_single_item() -> None:
    result = reciprocal_rank_fusion([[42]], k=60)
    assert len(result) == 1
    assert result[0] == (42, 1.0 / 61)


def test_rrf_tie_breaking_by_index() -> None:
    """Same rank position in different lists → lower index wins tie."""
    # Both lists have single element at rank 0
    result = reciprocal_rank_fusion([[5], [3]], k=60)
    # Both get same score; lower index (3) should come first
    assert result[0][0] == 3
    assert result[1][0] == 5


def test_rrf_k_parameter() -> None:
    """Higher k reduces rank influence."""
    r1 = reciprocal_rank_fusion([[0, 1]], k=1)
    r2 = reciprocal_rank_fusion([[0, 1]], k=100)
    # Both should have same ordering
    assert [x[0] for x in r1] == [x[0] for x in r2]
    # But scores differ
    assert r1[0][1] != r2[0][1]


# ---------------------------------------------------------------------------
# Semantic retrieval
# ---------------------------------------------------------------------------


def _make_index_with_embeddings():
    """Create a SimilarityIndex with mock embeddings."""
    emb = np.array([
        [1.0, 0.0, 0.0],  # doc-0
        [0.0, 1.0, 0.0],  # doc-1
        [0.0, 0.0, 1.0],  # doc-2
    ], dtype=np.float32)

    entries = [
        (f"doc-{i}", f"text {i}", Fingerprint(points=[], doc_id=f"doc-{i}"),
         None, i, None)
        for i in range(3)
    ]
    fingerprints = [e[2] for e in entries]

    idx = SimilarityIndex(
        entries=entries,
        fingerprints=fingerprints,
        embeddings=emb,
        embedding_model="test-model",
        embedding_dim=3,
    )
    return idx


def test_retrieve_semantic_returns_sorted() -> None:
    idx = _make_index_with_embeddings()
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    hits = idx.retrieve_semantic(query, top_k=3)
    assert len(hits) == 3
    assert hits[0] == (0, 1.0)  # exact match
    assert hits[1][1] == 0.0
    assert hits[2][1] == 0.0


def test_retrieve_semantic_empty_when_no_embeddings() -> None:
    idx = SimilarityIndex(entries=[], fingerprints=[])
    hits = idx.retrieve_semantic(np.array([1.0, 0.0]))
    assert hits == []


def test_retrieve_semantic_respects_top_k() -> None:
    idx = _make_index_with_embeddings()
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    hits = idx.retrieve_semantic(query, top_k=1)
    assert len(hits) == 1


def test_retrieve_semantic_raw_cosine_range() -> None:
    """Raw cosine scores are in [-1, 1] — no display normalization."""
    emb = np.array([
        [1.0, 0.0, 0.0],
        [0.8, 0.6, 0.0],   # cosine with [1,0,0] ≈ 0.8
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    entries = [
        (f"doc-{i}", f"text {i}", Fingerprint(points=[], doc_id=f"doc-{i}"),
         None, i, None)
        for i in range(3)
    ]
    idx = SimilarityIndex(
        entries=entries,
        fingerprints=[e[2] for e in entries],
        embeddings=emb,
    )
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    hits = idx.retrieve_semantic(query, top_k=3)
    # Doc 0 (cosine≈1.0) first, doc 1 (cosine≈0.8) second, doc 2 (cosine≈0) third
    assert hits[0][0] == 0
    assert hits[1][0] == 1
    assert hits[0][1] > hits[1][1] > hits[2][1]
    # All raw cosines are in [-1, 1]
    for _, score in hits:
        assert -1.0 <= score <= 1.0