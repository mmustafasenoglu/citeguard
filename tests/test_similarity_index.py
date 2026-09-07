"""Parity tests for SimilarityIndex abstraction.

Verifies that the new ``SimilarityIndex``-based code paths produce
identical results to the legacy ``corpus_entries`` list paths.
"""

from __future__ import annotations

from citeguard.corpus.models import CorpusEntry, CorpusLanguage, CorpusMetadata
from citeguard.models import Sentence
from citeguard.report import similarity_report
from citeguard.similarity.engine import SimilarityEngine
from citeguard.similarity.fingerprint import generate_shingles, winnow
from citeguard.similarity.index import SimilarityIndex
from citeguard.similarity.models import (
    Fingerprint,
    RiskLevel,
    SimilarityConfig,
    SimilarityEngineResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sentence(
    text: str,
    *,
    normalized: str | None = None,
    start: int = 0,
    citations: list | None = None,
) -> Sentence:
    return Sentence(
        text=text,
        normalized_text=normalized or text.lower(),
        paragraph_index=0,
        sentence_index=0,
        start_offset=start,
        end_offset=start + len(text),
        citations=citations or [],
    )


def _make_entry(
    text: str,
    doc_id: str = "doc-1",
    entry_index: int = 0,
) -> tuple:
    """Build a CorpusEntryTuple for the given text."""
    metadata = CorpusMetadata(
        title="Test Paper",
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
        entry_index=entry_index,
        metadata=metadata,
        char_offset=0,
        char_end=len(text),
    )
    fp = Fingerprint(
        points=winnow(generate_shingles(text.lower(), k=5), window=4),
        doc_id=doc_id,
    )
    return (doc_id, text.lower(), fp, metadata, entry_index, entry)


def _engine_result_fields(r: SimilarityEngineResult) -> dict:
    """Extract a deterministic snapshot of SimilarityEngineResult fields."""
    return {
        "overall_similarity_pct": r.overall_similarity_pct,
        "high_risk_count": r.high_risk_count,
        "medium_risk_count": r.medium_risk_count,
        "total_sentences": r.total_sentences,
        "matched_sentences": r.matched_sentences,
        "unique_matched_chars": r.unique_matched_chars,
        "eligible_chars": r.eligible_chars,
    }


def _match_fields(m) -> dict:
    return {
        "source_text": m.source_text,
        "source_title": m.source_title,
        "source_id": m.source_id,
        "exact_overlap": m.exact_overlap,
        "lexical_similarity": m.lexical_similarity,
        "combined_score": m.combined_score,
        "match_type": m.match_type,
        "matched_source_spans": [
            (s.paragraph_index, s.start, s.end) for s in m.matched_source_spans
        ],
        "matched_document_spans": [
            (s.paragraph_index, s.start, s.end) for s in m.matched_document_spans
        ],
    }


def _assert_matches_equal(old_matches, new_matches) -> None:
    """Assert two match lists are field-identical."""
    assert len(old_matches) == len(new_matches)
    for old_m, new_m in zip(old_matches, new_matches, strict=True):
        assert _match_fields(old_m) == _match_fields(new_m)


def _assert_engine_results_equal(
    old: SimilarityEngineResult,
    new: SimilarityEngineResult,
) -> None:
    """Assert two engine results are field-identical."""
    assert _engine_result_fields(old) == _engine_result_fields(new)
    assert len(old.results) == len(new.results)

    for old_r, new_r in zip(old.results, new.results, strict=True):
        assert old_r.attribution_risk == new_r.attribution_risk
        assert old_r.attribution_reason == new_r.attribution_reason
        assert old_r.best_match is None == (new_r.best_match is None)

        if old_r.best_match is not None:
            assert _match_fields(old_r.best_match) == _match_fields(new_r.best_match)

        assert len(old_r.matches) == len(new_r.matches)
        for old_m, new_m in zip(old_r.matches, new_r.matches, strict=True):
            assert _match_fields(old_m) == _match_fields(new_m)


# ---------------------------------------------------------------------------
# Test: SimilarityIndex.build
# ---------------------------------------------------------------------------


def test_index_build_creates_correct_structure() -> None:
    """SimilarityIndex.build() creates correct entry count and TF-IDF."""
    text = "the attention mechanism changed nlp research"
    entry = _make_entry(text)
    index = SimilarityIndex.build([entry])

    assert index.size == 1
    assert len(index.entries) == 1
    assert len(index.fingerprints) == 1
    assert index.tfidf_vectorizer is not None
    assert index.tfidf_matrix is not None
    assert index.embeddings is None
    assert index.embedding_model is None
    assert index.embedding_dim is None


def test_index_build_empty() -> None:
    """Building from empty list gives empty index."""
    index = SimilarityIndex.build([])
    assert index.size == 0
    assert index.tfidf_vectorizer is None
    assert index.tfidf_matrix is None


# ---------------------------------------------------------------------------
# Parity: analyze_document (end-to-end, single path through index)
# ---------------------------------------------------------------------------


def test_index_overall_similarity_parity() -> None:
    """overall_similarity_pct identical across both paths."""
    text = "the attention mechanism changed nlp research methods"
    sentence = _sentence(text)
    entry = _make_entry(text)

    corpus_entries = [entry]
    index = SimilarityIndex.build(corpus_entries)

    engine = SimilarityEngine()
    old_result = engine.analyze_document(
        [sentence], corpus_entries=corpus_entries,
    )
    new_result = engine.analyze_document(
        [sentence], index=index,
    )

    assert _engine_result_fields(old_result) == _engine_result_fields(new_result)


def test_index_bibliography_exclusion_parity() -> None:
    """Bibliography sentences excluded identically in both paths."""
    text = "the attention mechanism changed nlp research"
    bib_text = "vaswani et al attention is all you need 2017"

    sentence = _sentence(text)
    bib_sentence = Sentence(
        text=bib_text,
        normalized_text=bib_text.lower(),
        paragraph_index=1,
        sentence_index=0,
        start_offset=0,
        end_offset=len(bib_text),
        citations=[],
        is_bibliography=True,
    )

    entry = _make_entry(text)
    corpus_entries = [entry]
    index = SimilarityIndex.build(corpus_entries)

    engine = SimilarityEngine()
    old_result = engine.analyze_document(
        [sentence, bib_sentence], corpus_entries=corpus_entries,
    )
    new_result = engine.analyze_document(
        [sentence, bib_sentence], index=index,
    )

    assert _engine_result_fields(old_result) == _engine_result_fields(new_result)

    # Bib sentence should have no matches in both paths
    old_bib = old_result.results[1]
    new_bib = new_result.results[1]
    assert old_bib.matches == []
    assert new_bib.matches == []


# ---------------------------------------------------------------------------
# Parity: compare_sentence_to_corpus (with same TF-IDF source)
# ---------------------------------------------------------------------------


def test_index_compare_with_tfidf_parity() -> None:
    """compare_sentence_to_corpus with tfidf_info vs index — identical."""
    text = "the attention mechanism changed nlp research"
    sentence = _sentence(text)
    entry = _make_entry(text)

    corpus_entries = [entry]
    index = SimilarityIndex.build(corpus_entries)
    tfidf_info = (index.tfidf_vectorizer, index.tfidf_matrix)

    engine = SimilarityEngine()
    old_matches = engine.compare_sentence_to_corpus(
        sentence, corpus_entries, tfidf_info,
    )
    new_matches = engine.compare_sentence_to_corpus(sentence, index=index)

    _assert_matches_equal(old_matches, new_matches)


def test_index_near_duplicate_tfidf_parity() -> None:
    """Near-duplicate text with TF-IDF produces identical results both paths."""
    original = "the attention mechanism changed nlp research methods"
    sentence_text = "the attention mechanism modified nlp research methods"
    sentence = _sentence(sentence_text)
    entry = _make_entry(original)

    corpus_entries = [entry]
    index = SimilarityIndex.build(corpus_entries)
    tfidf_info = (index.tfidf_vectorizer, index.tfidf_matrix)

    engine = SimilarityEngine()
    old_matches = engine.compare_sentence_to_corpus(
        sentence, corpus_entries, tfidf_info,
    )
    new_matches = engine.compare_sentence_to_corpus(sentence, index=index)

    _assert_matches_equal(old_matches, new_matches)


def test_index_lexical_overlap_tfidf_parity() -> None:
    """Partial-overlap text with TF-IDF produces identical results both paths."""
    original = "the attention mechanism changed nlp research methods fundamentally"
    sentence_text = "attention based models changed the field"
    sentence = _sentence(sentence_text)
    entry = _make_entry(original)

    corpus_entries = [entry]
    index = SimilarityIndex.build(corpus_entries)
    tfidf_info = (index.tfidf_vectorizer, index.tfidf_matrix)

    engine = SimilarityEngine()
    old_matches = engine.compare_sentence_to_corpus(
        sentence, corpus_entries, tfidf_info,
    )
    new_matches = engine.compare_sentence_to_corpus(sentence, index=index)

    _assert_matches_equal(old_matches, new_matches)


def test_index_source_spans_parity() -> None:
    """matched_source_spans are identical across both paths."""
    text = "the attention mechanism changed nlp research"
    sentence = _sentence(text)
    entry = _make_entry(text)

    corpus_entries = [entry]
    index = SimilarityIndex.build(corpus_entries)
    tfidf_info = (index.tfidf_vectorizer, index.tfidf_matrix)

    engine = SimilarityEngine()
    old_matches = engine.compare_sentence_to_corpus(
        sentence, corpus_entries, tfidf_info,
    )
    new_matches = engine.compare_sentence_to_corpus(sentence, index=index)

    assert len(old_matches) == len(new_matches)
    for old_m, new_m in zip(old_matches, new_matches, strict=True):
        assert old_m.matched_source_spans == new_m.matched_source_spans
        assert old_m.matched_document_spans == new_m.matched_document_spans


def test_index_attribution_risk_parity() -> None:
    """Risk level and reason are identical across both paths."""
    text = "the attention mechanism changed nlp research"
    sentence = _sentence(text)
    entry = _make_entry(text)

    corpus_entries = [entry]
    index = SimilarityIndex.build(corpus_entries)
    tfidf_info = (index.tfidf_vectorizer, index.tfidf_matrix)

    engine = SimilarityEngine()

    # Old path
    old_matches = engine.compare_sentence_to_corpus(
        sentence, corpus_entries, tfidf_info,
    )
    old_risk, old_reason = engine.compute_attribution_risk(
        old_matches[0], [], [],
        sentence_text=sentence.text,
        sentence_start_offset=sentence.start_offset,
    ) if old_matches else (RiskLevel.NONE, "")

    # New path
    new_matches = engine.compare_sentence_to_corpus(sentence, index=index)
    new_risk, new_reason = engine.compute_attribution_risk(
        new_matches[0], [], [],
        sentence_text=sentence.text,
        sentence_start_offset=sentence.start_offset,
    ) if new_matches else (RiskLevel.NONE, "")

    assert old_risk == new_risk
    assert old_reason == new_reason


# ---------------------------------------------------------------------------
# Empty corpus
# ---------------------------------------------------------------------------


def test_index_empty_corpus() -> None:
    """Empty index produces empty matches."""
    sentence = _sentence("some text")
    index = SimilarityIndex.build([])

    engine = SimilarityEngine()
    matches = engine.compare_sentence_to_corpus(sentence, index=index)
    assert matches == []


# ---------------------------------------------------------------------------
# Retrieval methods
# ---------------------------------------------------------------------------


def test_index_retrieve_tfidf() -> None:
    """TF-IDF retrieval returns sorted entry indices."""
    text1 = "the attention mechanism changed nlp research"
    text2 = "convolutional networks for image classification"
    entry1 = _make_entry(text1, doc_id="doc-0")
    entry2 = _make_entry(text2, doc_id="doc-1")
    index = SimilarityIndex.build([entry1, entry2])

    hits = index.retrieve_tfidf("attention mechanism", top_k=10)
    assert len(hits) == 2
    # First hit should be the more relevant entry
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_index_retrieve_fingerprint() -> None:
    """Fingerprint retrieval returns sorted entry indices."""
    text = "the attention mechanism changed nlp research"
    entry = _make_entry(text)
    index = SimilarityIndex.build([entry])

    query_fp = Fingerprint(
        points=winnow(generate_shingles(text.lower(), k=5), window=4),
    )
    hits = index.retrieve_fingerprint(query_fp, top_k=10)
    assert len(hits) >= 1
    assert hits[0][1] > 0  # non-zero score


def test_index_retrieve_candidates() -> None:
    """Candidates are union of FP + TF-IDF, deduped."""
    text1 = "the attention mechanism changed nlp research"
    text2 = "convolutional networks for image classification"
    entry1 = _make_entry(text1, doc_id="doc-0")
    entry2 = _make_entry(text2, doc_id="doc-1")
    index = SimilarityIndex.build([entry1, entry2])

    query_fp = Fingerprint(
        points=winnow(generate_shingles(text1.lower(), k=5), window=4),
    )
    candidates = index.retrieve_candidates(text1.lower(), query_fp, top_k=10)
    # Should have at least 1 candidate, no duplicates
    assert len(candidates) == len(set(candidates))


# ---------------------------------------------------------------------------
# Report-level parity
# ---------------------------------------------------------------------------


def test_index_report_parity() -> None:
    """similarity_report(old_result) == similarity_report(index_result)
    for an end-to-end document analysis."""
    text1 = "the attention mechanism changed nlp research methods"
    text2 = "convolutional neural networks revolutionized image classification"

    sentence1 = _sentence(text1)
    sentence2 = _sentence(text2)
    entries = [_make_entry(text1, doc_id="doc-0"), _make_entry(text2, doc_id="doc-1")]

    corpus_entries = entries
    index = SimilarityIndex.build(entries)

    engine = SimilarityEngine()
    old_result = engine.analyze_document(
        [sentence1, sentence2], corpus_entries=corpus_entries,
    )
    new_result = engine.analyze_document(
        [sentence1, sentence2], index=index,
    )

    old_report = similarity_report(old_result)
    new_report = similarity_report(new_result)
    assert old_report == new_report


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_analyze_document_backward_compat() -> None:
    """analyze_document(sentences, corpus_entries, bib_entries) still works."""
    text = "the attention mechanism changed nlp research"
    sentence = _sentence(text)
    entry = _make_entry(text)

    engine = SimilarityEngine()
    result = engine.analyze_document(
        [sentence], corpus_entries=[entry],
    )
    assert isinstance(result, SimilarityEngineResult)
    assert result.total_sentences == 1
    assert result.matched_sentences >= 1


def test_analyze_document_with_index() -> None:
    """analyze_document(sentences, index=index, bib_entries) works."""
    text = "the attention mechanism changed nlp research"
    sentence = _sentence(text)
    entry = _make_entry(text)
    index = SimilarityIndex.build([entry])

    engine = SimilarityEngine()
    result = engine.analyze_document(
        [sentence], index=index,
    )
    assert isinstance(result, SimilarityEngineResult)
    assert result.total_sentences == 1
    assert result.matched_sentences >= 1


# ---------------------------------------------------------------------------
# Multiple entry parity (max_results_per_sentence)
# ---------------------------------------------------------------------------


def test_index_max_results_parity() -> None:
    """max_results_per_sentence caps identically in both paths."""
    config = SimilarityConfig(max_results_per_sentence=2)
    text = "the attention mechanism changed nlp research"

    sentence = _sentence(text)
    entries = [_make_entry(text, doc_id=f"doc-{i}", entry_index=i) for i in range(5)]

    corpus_entries = entries
    index = SimilarityIndex.build(entries)
    tfidf_info = (index.tfidf_vectorizer, index.tfidf_matrix)

    engine = SimilarityEngine(config=config)
    old_matches = engine.compare_sentence_to_corpus(
        sentence, corpus_entries, tfidf_info,
    )
    new_matches = engine.compare_sentence_to_corpus(sentence, index=index)

    _assert_matches_equal(old_matches, new_matches)
    assert len(new_matches) <= config.max_results_per_sentence
