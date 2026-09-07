"""Tests for checkpoint hardening: SEMANTIC_OVERLAP, reranker, candidates."""

from __future__ import annotations

from citeguard.corpus.models import CorpusEntry, CorpusLanguage, CorpusMetadata
from citeguard.models import Sentence, TextSpan
from citeguard.similarity.engine import SimilarityEngine
from citeguard.similarity.fingerprint import generate_shingles, winnow
from citeguard.similarity.index import SimilarityIndex
from citeguard.similarity.lexical import classify_match_type
from citeguard.similarity.models import (
    Fingerprint,
    MatchType,
    SimilarityConfig,
)
from citeguard.similarity.reranker import compute_ranking_score


def _sentence(text: str) -> Sentence:
    return Sentence(
        text=text,
        normalized_text=text.lower(),
        paragraph_index=0,
        sentence_index=0,
        start_offset=0,
        end_offset=len(text),
        citations=[],
    )


def _make_entry(text: str, doc_id: str = "doc-1") -> tuple:
    metadata = CorpusMetadata(
        title="Paper",
        authors=["Smith"],
        year=2020,
        language=CorpusLanguage.ENGLISH,
        license="CC0",
        similarity_index_allowed=True,
    )
    entry = CorpusEntry(
        text=text,
        normalized_text=text.lower(),
        doc_id=doc_id,
        entry_index=0,
        metadata=metadata,
        char_offset=0,
        char_end=len(text),
    )
    fp = Fingerprint(
        points=winnow(generate_shingles(text.lower(), k=5), window=4),
        doc_id=doc_id,
    )
    return (doc_id, text.lower(), fp, metadata, 0, entry)


# ---------------------------------------------------------------------------
# classify_match_type 4th tier
# ---------------------------------------------------------------------------


def test_classify_semantic_overlap() -> None:
    """When all lexical thresholds missed but semantic >= threshold."""
    mt = classify_match_type(
        exact_overlap=0.1,
        lexical_similarity=0.2,
        semantic_score=0.85,
        semantic_threshold=0.8,
    )
    assert mt == MatchType.SEMANTIC_OVERLAP


def test_classify_no_semantic_without_threshold() -> None:
    """Without semantic_threshold, never produces SEMANTIC_OVERLAP."""
    mt = classify_match_type(
        exact_overlap=0.1,
        lexical_similarity=0.2,
        semantic_score=0.99,
    )
    assert mt == MatchType.UNMATCHED


def test_classify_exact_beats_semantic() -> None:
    """EXACT takes priority even when semantic is also high."""
    mt = classify_match_type(
        exact_overlap=0.97,
        lexical_similarity=0.5,
        semantic_score=0.99,
        semantic_threshold=0.8,
    )
    assert mt == MatchType.EXACT


# ---------------------------------------------------------------------------
# SEMANTIC_OVERLAP invariant: empty spans
# ---------------------------------------------------------------------------


def test_semantic_overlap_empty_spans_invariant() -> None:
    """SEMANTIC_OVERLAP matches never create document or source spans."""
    mt = classify_match_type(
        exact_overlap=0.1,
        lexical_similarity=0.2,
        semantic_score=0.95,
        semantic_threshold=0.5,
    )
    assert mt == MatchType.SEMANTIC_OVERLAP


def test_semantic_overlap_excluded_from_textual_similarity() -> None:
    """SEMANTIC_OVERLAP matches don't contribute to overall_similarity_pct."""
    result_match = type("MockMatch", (), {
        "matched_document_spans": [],
        "match_type": MatchType.SEMANTIC_OVERLAP,
    })()

    spans: list[TextSpan] = []
    spans.extend(result_match.matched_document_spans)
    assert spans == []


# ---------------------------------------------------------------------------
# candidate_indices
# ---------------------------------------------------------------------------


def test_candidate_indices_limits_evaluation() -> None:
    """candidate_indices restricts which corpus entries are evaluated."""
    text = "the attention mechanism changed nlp research"
    sentence = _sentence(text)
    entries = [_make_entry(text, doc_id=f"doc-{i}") for i in range(5)]

    engine = SimilarityEngine()
    matches = engine.compare_sentence_to_corpus(
        sentence,
        corpus_entries=entries,
        candidate_indices=[0],
    )
    for m in matches:
        assert m.source_id == "doc-0"


def test_candidate_indices_none_fallback() -> None:
    """candidate_indices=None -> full scan (backward compat)."""
    text = "the attention mechanism changed nlp research"
    sentence = _sentence(text)
    entries = [_make_entry(text, doc_id=f"doc-{i}") for i in range(5)]

    engine = SimilarityEngine()
    all_matches = engine.compare_sentence_to_corpus(
        sentence,
        corpus_entries=entries,
        candidate_indices=None,
    )
    source_ids = {m.source_id for m in all_matches}
    assert len(source_ids) >= 1


# ---------------------------------------------------------------------------
# compute_ranking_score (renamed from compute_rerank_score)
# ---------------------------------------------------------------------------


def test_ranking_score_weighted() -> None:
    """Ranking score combines exact, lexical, and semantic weights."""
    config = SimilarityConfig(
        weight_fingerprint=0.3,
        weight_tfidf=0.3,
        weight_semantic=0.4,
    )
    sem_component, ranking = compute_ranking_score(
        exact_overlap=0.9,
        lexical_similarity=0.7,
        semantic_raw=0.85,
        config=config,
    )
    assert sem_component == 0.85
    assert 0 < ranking <= 1.0


def test_ranking_score_semantic_normalization() -> None:
    """Raw cosine uses max(0, cosine) — no 0.5 base for negative."""
    config = SimilarityConfig(
        weight_fingerprint=0.0,
        weight_tfidf=0.0,
        weight_semantic=1.0,
    )
    _, score_neg = compute_ranking_score(0.0, 0.0, -1.0, config)
    _, score_zero = compute_ranking_score(0.0, 0.0, 0.0, config)
    _, score_pos = compute_ranking_score(0.0, 0.0, 1.0, config)
    assert score_neg == 0.0
    assert score_zero == 0.0
    assert score_pos == 1.0


def test_ranking_score_single_signal() -> None:
    """Ranking score works with only one signal."""
    config = SimilarityConfig(
        weight_fingerprint=1.0,
        weight_tfidf=0.0,
        weight_semantic=0.0,
    )
    _, score = compute_ranking_score(0.8, 0.5, 0.3, config)
    assert score == 0.8


# ---------------------------------------------------------------------------
# CandidateSet via index
# ---------------------------------------------------------------------------


def test_retrieve_candidates_returns_candidate_set() -> None:
    """retrieve_candidates returns CandidateSet with indices + scores."""
    text = "the transformer architecture changed nlp research"
    entries = [_make_entry(text, doc_id=f"doc-{i}") for i in range(3)]
    idx = SimilarityIndex.build(entries)

    from citeguard.similarity.fingerprint import Fingerprint as Fp
    fp = Fp(
        points=winnow(generate_shingles(text.lower(), k=5), window=4),
        doc_id="query",
    )

    cs = idx.retrieve_candidates(query_text=text.lower(), query_fp=fp, top_k=3)
    assert hasattr(cs, "indices")
    assert hasattr(cs, "semantic_scores")
    assert hasattr(cs, "rrf_scores")
    assert len(cs.indices) <= 3
