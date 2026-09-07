"""Regression tests for similarity engine wiring."""

from __future__ import annotations

from citeguard.corpus.models import CorpusLanguage, CorpusMetadata
from citeguard.models import BibliographyEntry, ExistingCitation, Sentence, TextSpan
from citeguard.similarity.engine import SimilarityEngine
from citeguard.similarity.fingerprint import generate_shingles, winnow
from citeguard.similarity.models import Fingerprint, MatchType, RiskLevel, SimilarityMatch


def _sentence(text: str, *, start: int = 0, citations: list | None = None) -> Sentence:
    return Sentence(
        text=text,
        normalized_text=text.lower(),
        paragraph_index=0,
        sentence_index=0,
        start_offset=start,
        end_offset=start + len(text),
        citations=citations or [],
    )


def _match(**kwargs) -> SimilarityMatch:
    defaults = {
        "source_text": "copied passage",
        "source_title": "Paper",
        "source_id": "doc-1",
        "exact_overlap": 0.97,
        "lexical_similarity": 0.9,
        "combined_score": 0.94,
        "source_authors": ["Vaswani"],
        "source_year": 2017,
        "match_type": MatchType.EXACT,
    }
    defaults.update(kwargs)
    return SimilarityMatch(**defaults)


def test_no_citation_high_overlap_is_high_risk() -> None:
    engine = SimilarityEngine()
    risk, _ = engine.compute_attribution_risk(_match(), [], [])
    assert risk == RiskLevel.HIGH


def test_wrong_citation_is_high_risk() -> None:
    engine = SimilarityEngine()
    citation = ExistingCitation(
        raw_text="(Jones, 2021)",
        authors="Jones",
        year=2021,
        doi=None,
        numbered_ref=None,
        paragraph_index=0,
        char_offset=10,
    )
    risk, _ = engine.compute_attribution_risk(_match(), [citation], [])
    assert risk == RiskLevel.HIGH


def test_matching_citation_without_quote_is_medium() -> None:
    engine = SimilarityEngine()
    citation = ExistingCitation(
        raw_text="(Vaswani, 2017)",
        authors="Vaswani",
        year=2017,
        doi=None,
        numbered_ref=None,
        paragraph_index=0,
        char_offset=10,
    )
    risk, _ = engine.compute_attribution_risk(
        _match(),
        [citation],
        [],
        sentence_text="Transformer architectures were introduced in 2017.",
    )
    assert risk == RiskLevel.MEDIUM


def test_matching_citation_with_quote_is_low() -> None:
    engine = SimilarityEngine()
    citation = ExistingCitation(
        raw_text="(Vaswani, 2017)",
        authors="Vaswani",
        year=2017,
        doi=None,
        numbered_ref=None,
        paragraph_index=0,
        char_offset=10,
    )
    risk, _ = engine.compute_attribution_risk(
        _match(),
        [citation],
        [],
        sentence_text='"Attention is all you need" (Vaswani, 2017).',
    )
    assert risk == RiskLevel.LOW


def test_citation_can_match_via_bibliography_doi() -> None:
    engine = SimilarityEngine()
    citation = ExistingCitation(
        raw_text="[1]",
        authors=None,
        year=None,
        doi=None,
        numbered_ref=1,
        paragraph_index=0,
        char_offset=4,
    )
    bib = BibliographyEntry(
        raw_text="Vaswani (2017)",
        authors="Vaswani",
        year=2017,
        title="Attention",
        doi="10.1000/xyz",
        numbered_ref=1,
    )
    risk, _ = engine.compute_attribution_risk(
        _match(source_doi="10.1000/xyz"),
        [citation],
        [bib],
        sentence_text='"quoted" text',
    )
    assert risk == RiskLevel.LOW


def test_multi_citation_one_matches_source_not_high() -> None:
    """When one citation matches source and another doesn't, should not be HIGH."""
    engine = SimilarityEngine()
    citation1 = ExistingCitation(
        raw_text="(Vaswani, 2017)",
        authors="Vaswani",
        year=2017,
        doi=None,
        numbered_ref=None,
        paragraph_index=0,
        char_offset=10,
    )
    citation2 = ExistingCitation(
        raw_text="(Devlin, 2019)",
        authors="Devlin",
        year=2019,
        doi=None,
        numbered_ref=None,
        paragraph_index=0,
        char_offset=30,
    )
    risk, _ = engine.compute_attribution_risk(
        _match(source_authors=["Vaswani"], source_year=2017),
        [citation1, citation2],
        [],
        sentence_text="Transformer architectures were introduced in 2017.",
    )
    # At least one citation matches (Vaswani), so should not be HIGH
    assert risk != RiskLevel.HIGH


def test_multi_citation_none_match_source_is_high() -> None:
    """When no citations match the detected source, should be HIGH."""
    engine = SimilarityEngine()
    citation1 = ExistingCitation(
        raw_text="(Devlin, 2019)",
        authors="Devlin",
        year=2019,
        doi=None,
        numbered_ref=None,
        paragraph_index=0,
        char_offset=10,
    )
    citation2 = ExistingCitation(
        raw_text="(Brown, 2020)",
        authors="Brown",
        year=2020,
        doi=None,
        numbered_ref=None,
        paragraph_index=0,
        char_offset=30,
    )
    risk, _ = engine.compute_attribution_risk(
        _match(source_authors=["Vaswani"], source_year=2017),
        [citation1, citation2],
        [],
        sentence_text="Transformer architectures were introduced in 2017.",
    )
    assert risk == RiskLevel.HIGH


def test_document_spans_are_paragraph_relative() -> None:
    text = "the attention mechanism changed nlp research methods"
    prefix = "Earlier context. "
    sentence = _sentence(text, start=len(prefix))
    fp = Fingerprint(points=winnow(generate_shingles(text.lower(), k=5), window=4))
    metadata = CorpusMetadata(
        title="Attention",
        authors=["Vaswani"],
        year=2017,
        language=CorpusLanguage.ENGLISH,
        license="CC0",
        similarity_index_allowed=True,
    )
    from citeguard.corpus.models import CorpusEntry
    corpus_entry = CorpusEntry(
        text=text.lower(),
        normalized_text=text.lower(),
        doc_id="doc-1",
        entry_index=0,
        metadata=metadata,
        char_offset=0,
        char_end=len(text),
    )
    engine = SimilarityEngine()
    matches = engine.compare_sentence_to_corpus(
        sentence,
        [("doc-1", text.lower(), fp, metadata, 0, corpus_entry)],
    )
    assert matches
    span = matches[0].matched_document_spans[0]
    assert span.start >= sentence.start_offset
    assert matches[0].matched_source_spans
    assert matches[0].source_authors == ["Vaswani"]


def test_span_aware_quote_detection_with_real_spans() -> None:
    """Bug 3: When matched_document_spans fall within quotes, risk is LOW."""
    from citeguard.models import TextSpan
    engine = SimilarityEngine()
    citation = ExistingCitation(
        raw_text="(Vaswani, 2017)",
        authors="Vaswani",
        year=2017,
        doi=None,
        numbered_ref=None,
        paragraph_index=0,
        char_offset=30,
    )
    sentence_text = '"Attention is all you need" (Vaswani, 2017).'
    sentence = _sentence(sentence_text)
    # Simulate a match whose span covers the quoted portion (indices 0..26
    # exclusive, i.e. the opening quote through the closing quote)
    match = _match(
        matched_document_spans=[
            TextSpan(
                paragraph_index=0,
                start=sentence.start_offset,
                end=sentence.start_offset + 26,
            ),
        ],
    )
    risk, reason = engine.compute_attribution_risk(
        match,
        [citation],
        [],
        sentence_text=sentence_text,
        sentence_start_offset=sentence.start_offset,
    )
    assert risk == RiskLevel.LOW
    assert "quoted" in reason


def test_span_aware_quote_detection_outside_quotes() -> None:
    """Bug 3: When matched span is outside quotes, risk is MEDIUM."""
    from citeguard.models import TextSpan
    engine = SimilarityEngine()
    citation = ExistingCitation(
        raw_text="(Vaswani, 2017)",
        authors="Vaswani",
        year=2017,
        doi=None,
        numbered_ref=None,
        paragraph_index=0,
        char_offset=30,
    )
    sentence_text = 'She said "hello" (Vaswani, 2017) and walked away.'
    sentence = _sentence(sentence_text)
    # Span covers the part AFTER the quote (indices 28..end), not inside it
    match = _match(
        matched_document_spans=[
            TextSpan(
                paragraph_index=0,
                start=sentence.start_offset + 28,
                end=sentence.start_offset + len(sentence_text),
            ),
        ],
    )
    risk, reason = engine.compute_attribution_risk(
        match,
        [citation],
        [],
        sentence_text=sentence_text,
        sentence_start_offset=sentence.start_offset,
    )
    assert risk == RiskLevel.MEDIUM
    assert "high exact overlap" in reason


def test_max_results_per_sentence_limits_matches() -> None:
    """Bug 4: max_results_per_sentence caps the number of returned matches."""
    from citeguard.similarity.models import SimilarityConfig
    config = SimilarityConfig(max_results_per_sentence=2)
    engine = SimilarityEngine(config=config)

    text = "the attention mechanism changed nlp research"
    prefix = "Earlier context. "
    sentence = _sentence(text, start=len(prefix))
    metadata = CorpusMetadata(
        title="Paper",
        authors=["Vaswani"],
        year=2017,
        language=CorpusLanguage.ENGLISH,
        license="CC0",
        similarity_index_allowed=True,
    )
    from citeguard.corpus.models import CorpusEntry

    # Create 5 corpus entries that all match the sentence
    corpus_entries = []
    for i in range(5):
        ce = CorpusEntry(
            text=text.lower(),
            normalized_text=text.lower(),
            doc_id=f"doc-{i}",
            entry_index=i,
            metadata=metadata,
            char_offset=0,
            char_end=len(text),
        )
        fp = Fingerprint(
            points=winnow(generate_shingles(text.lower(), k=5), window=4)
        )
        corpus_entries.append(
            (f"doc-{i}", text.lower(), fp, metadata, i, ce)
        )

    matches = engine.compare_sentence_to_corpus(sentence, corpus_entries)
    assert len(matches) <= config.max_results_per_sentence


# ---------------------------------------------------------------------------
# P1-2: source_text invariant — spans index source_text directly
# ---------------------------------------------------------------------------


def test_source_text_is_original_corpus_text() -> None:
    """source_text uses CorpusEntry.text (original), not normalized."""
    text = "The attention mechanism changed NLP research methods"
    prefix = "Earlier context. "
    sentence = _sentence(text, start=len(prefix))
    from citeguard.corpus.models import CorpusEntry
    metadata = CorpusMetadata(
        title="Attention",
        authors=["Vaswani"],
        year=2017,
        language=CorpusLanguage.ENGLISH,
        license="CC0",
        similarity_index_allowed=True,
    )
    corpus_entry = CorpusEntry(
        text=text,  # original with caps
        normalized_text=text.lower(),
        doc_id="doc-1",
        entry_index=0,
        metadata=metadata,
        char_offset=0,
        char_end=len(text),
    )
    fp = Fingerprint(
        points=winnow(generate_shingles(text.lower(), k=5), window=4)
    )
    engine = SimilarityEngine()
    matches = engine.compare_sentence_to_corpus(
        sentence,
        [("doc-1", text.lower(), fp, metadata, 0, corpus_entry)],
    )
    assert matches
    # source_text must be the ORIGINAL text, not lowercased
    assert matches[0].source_text == text
    assert "NLP" in matches[0].source_text


def test_source_text_span_indices_are_valid() -> None:
    """source_text[start:end] extracts a valid substring from source_text."""
    text = "The attention mechanism changed NLP research methods"
    prefix = "Earlier context. "
    sentence = _sentence(text, start=len(prefix))
    from citeguard.corpus.models import CorpusEntry
    metadata = CorpusMetadata(
        title="Attention",
        authors=["Vaswani"],
        year=2017,
        language=CorpusLanguage.ENGLISH,
        license="CC0",
        similarity_index_allowed=True,
    )
    corpus_entry = CorpusEntry(
        text=text,
        normalized_text=text.lower(),
        doc_id="doc-1",
        entry_index=0,
        metadata=metadata,
        char_offset=0,
        char_end=len(text),
    )
    fp = Fingerprint(
        points=winnow(generate_shingles(text.lower(), k=5), window=4)
    )
    engine = SimilarityEngine()
    matches = engine.compare_sentence_to_corpus(
        sentence,
        [("doc-1", text.lower(), fp, metadata, 0, corpus_entry)],
    )
    assert matches
    for span in matches[0].matched_source_spans:
        # span indices must be within source_text bounds
        assert 0 <= span.start < span.end <= len(matches[0].source_text)
        extracted = matches[0].source_text[span.start:span.end]
        assert len(extracted) > 0


# ---------------------------------------------------------------------------
# P1-3: Coverage-based quote detection
# ---------------------------------------------------------------------------


def test_mixed_span_one_quoted_one_not_is_medium() -> None:
    """One span inside quotes, one outside → MEDIUM, not LOW."""
    from citeguard.models import TextSpan
    engine = SimilarityEngine()
    citation = ExistingCitation(
        raw_text="(Vaswani, 2017)",
        authors="Vaswani",
        year=2017,
        doi=None,
        numbered_ref=None,
        paragraph_index=0,
        char_offset=40,
    )
    sentence_text = (
        '"quoted part" and a very long copied passage outside quotes'
    )
    # Span 1: inside quotes (indices 0..14)
    # Span 2: outside quotes (indices 30..61)
    match = _match(
        matched_document_spans=[
            TextSpan(paragraph_index=0, start=0, end=14),
            TextSpan(paragraph_index=0, start=30, end=61),
        ],
    )
    risk, reason = engine.compute_attribution_risk(
        match,
        [citation],
        [],
        sentence_text=sentence_text,
        sentence_start_offset=0,
    )
    # Only ~30% of matched chars are quoted → MEDIUM
    assert risk == RiskLevel.MEDIUM


def test_high_coverage_quoted_is_low() -> None:
    """When >= 95% of matched chars are quoted → LOW."""
    from citeguard.models import TextSpan
    engine = SimilarityEngine()
    citation = ExistingCitation(
        raw_text="(Vaswani, 2017)",
        authors="Vaswani",
        year=2017,
        doi=None,
        numbered_ref=None,
        paragraph_index=0,
        char_offset=50,
    )
    # Long quoted portion + 1 char outside
    sentence_text = '"aaaa bbbb cccc dddd eeee ffff gggg" x'
    # Quote pair: (0, 36) — " at index 0, " at index 35
    # Matched span covers quote area (0..33, inside 0..36) + 1 char outside (37..38)
    match = _match(
        matched_document_spans=[
            TextSpan(paragraph_index=0, start=0, end=33),
            TextSpan(paragraph_index=0, start=37, end=38),
        ],
    )
    risk, _ = engine.compute_attribution_risk(
        match,
        [citation],
        [],
        sentence_text=sentence_text,
        sentence_start_offset=0,
    )
    # 33 / 34 = 97.1% → above 0.95 → LOW
    assert risk == RiskLevel.LOW


def test_low_coverage_quoted_is_medium() -> None:
    """When < 95% of matched chars are quoted → MEDIUM."""
    from citeguard.models import TextSpan
    engine = SimilarityEngine()
    citation = ExistingCitation(
        raw_text="(Vaswani, 2017)",
        authors="Vaswani",
        year=2017,
        doi=None,
        numbered_ref=None,
        paragraph_index=0,
        char_offset=40,
    )
    sentence_text = '"short" and a very long copied passage outside quotes'
    # 7 chars quoted, 30 chars not → 7/37 = 18.9% → MEDIUM
    match = _match(
        matched_document_spans=[
            TextSpan(paragraph_index=0, start=0, end=7),
            TextSpan(paragraph_index=0, start=15, end=45),
        ],
    )
    risk, _ = engine.compute_attribution_risk(
        match,
        [citation],
        [],
        sentence_text=sentence_text,
        sentence_start_offset=0,
    )
    assert risk == RiskLevel.MEDIUM


def test_overlapping_spans_not_double_counted() -> None:
    """Overlapping matched spans should not double-count characters."""
    from citeguard.models import TextSpan
    from citeguard.similarity.engine import _compute_quote_coverage
    sentence_text = '"quoted text here" outside'
    # Two overlapping spans that cover the same 10 chars inside quotes
    spans = [
        TextSpan(paragraph_index=0, start=0, end=10),
        TextSpan(paragraph_index=0, start=5, end=15),
    ]
    coverage = _compute_quote_coverage(sentence_text, spans, 0)
    # After merge: single span 0..15, intersection with quote [0..18) = 15
    # 15/15 = 1.0
    assert coverage == 1.0


# ---------------------------------------------------------------------------
# P1-4: Bibliography fully excluded
# ---------------------------------------------------------------------------


def test_bibliography_sentence_no_matches() -> None:
    """Bibliography sentences should not be matched against corpus."""
    text = "Vaswani et al. (2017) Attention is all you need."
    sentence = Sentence(
        text=text,
        normalized_text=text.lower(),
        paragraph_index=0,
        sentence_index=0,
        start_offset=0,
        end_offset=len(text),
        citations=[],
        is_bibliography=True,
    )
    metadata = CorpusMetadata(
        title="Attention",
        authors=["Vaswani"],
        year=2017,
        language=CorpusLanguage.ENGLISH,
        license="CC0",
        similarity_index_allowed=True,
    )
    from citeguard.corpus.models import CorpusEntry
    corpus_entry = CorpusEntry(
        text=text.lower(),
        normalized_text=text.lower(),
        doc_id="doc-1",
        entry_index=0,
        metadata=metadata,
        char_offset=0,
        char_end=len(text),
    )
    fp = Fingerprint(
        points=winnow(generate_shingles(text.lower(), k=5), window=4)
    )
    engine = SimilarityEngine()
    results = engine.analyze_document(
        [sentence],
        [("doc-1", text.lower(), fp, metadata, 0, corpus_entry)],
    )
    # Bibliography sentence gets no matches
    assert results.results[0].matches == []
    assert results.results[0].attribution_risk == RiskLevel.NONE
    assert "bibliography" in results.results[0].attribution_reason


def test_bibliography_not_counted_in_matched_sentences() -> None:
    """Bibliography sentences do not increment matched_sentences."""
    bib_text = "Vaswani et al. (2017) Attention is all you need."
    body_text = "The attention mechanism revolutionized NLP research."
    bib_sentence = Sentence(
        text=bib_text,
        normalized_text=bib_text.lower(),
        paragraph_index=0,
        sentence_index=0,
        start_offset=0,
        end_offset=len(bib_text),
        citations=[],
        is_bibliography=True,
    )
    body_sentence = Sentence(
        text=body_text,
        normalized_text=body_text.lower(),
        paragraph_index=1,
        sentence_index=1,
        start_offset=0,
        end_offset=len(body_text),
        citations=[],
    )
    metadata = CorpusMetadata(
        title="Attention",
        authors=["Vaswani"],
        year=2017,
        language=CorpusLanguage.ENGLISH,
        license="CC0",
        similarity_index_allowed=True,
    )
    from citeguard.corpus.models import CorpusEntry
    corpus_entry = CorpusEntry(
        text=bib_text.lower(),
        normalized_text=bib_text.lower(),
        doc_id="doc-1",
        entry_index=0,
        metadata=metadata,
        char_offset=0,
        char_end=len(bib_text),
    )
    fp = Fingerprint(
        points=winnow(generate_shingles(bib_text.lower(), k=5), window=4)
    )
    engine = SimilarityEngine()
    results = engine.analyze_document(
        [bib_sentence, body_sentence],
        [("doc-1", bib_text.lower(), fp, metadata, 0, corpus_entry)],
    )
    # Bib sentence matched=0, body might or might not match
    bib_result = results.results[0]
    assert bib_result.matches == []
    # matched_sentences only counts non-bibliography sentences with matches
    assert results.matched_sentences <= 1


# ---------------------------------------------------------------------------
# Checkpoint 1: Nested quote union
# ---------------------------------------------------------------------------


def test_nested_quotes_not_double_counted() -> None:
    """Nested quote pairs (e.g. \"He said 'hello'\") must not double-count."""
    from citeguard.similarity.engine import _compute_quote_coverage

    # Outer: " (0..22), Inner: ' (9..15)
    text = '"He said \'hello\' today"'
    spans = [TextSpan(paragraph_index=0, start=0, end=len(text))]
    coverage = _compute_quote_coverage(text, spans, 0)
    # All text is inside outer quote — nested inner quote chars counted once
    assert coverage == 1.0


def test_adjacent_quotes_merged() -> None:
    """Adjacent but non-overlapping quote pairs should not merge."""
    from citeguard.similarity.engine import _compute_quote_coverage

    # Two separate quotes: "abc" and "def"
    text = '"abc" and "def"'
    # Span covers only "abc" (0..5)
    spans = [TextSpan(paragraph_index=0, start=0, end=5)]
    coverage = _compute_quote_coverage(text, spans, 0)
    # 5 chars quoted / 5 total = 1.0
    assert coverage == 1.0


def test_overlapping_quotes_merged_correctly() -> None:
    """Overlapping quote pairs merge into a single range."""
    from citeguard.similarity.engine import _compute_quote_coverage

    # Simulated overlapping: "abc" (0..5) and "bcd" (2..7)
    # Merged: (0..7)
    text = '"abc" "bcd"'  # actual pairs: "abc" (0..5), "bcd" (7..12)
    # This test verifies non-overlapping separate quotes work independently
    spans = [TextSpan(paragraph_index=0, start=0, end=12)]
    coverage = _compute_quote_coverage(text, spans, 0)
    # "abc" (0..5) + "bcd" (7..12) = 10 quoted / 12 total
    assert abs(coverage - 10 / 12) < 0.01


def test_mixed_quotes_and_unquoted_span() -> None:
    """Span covering both quoted and unquoted portions."""
    from citeguard.similarity.engine import _compute_quote_coverage

    text = '"quoted text" and unquoted'
    spans = [TextSpan(paragraph_index=0, start=0, end=len(text))]
    coverage = _compute_quote_coverage(text, spans, 0)
    # "quoted text" = 12 chars (0..13 with quotes), total = 26
    assert 0.0 < coverage < 1.0


# ---------------------------------------------------------------------------
# Checkpoint 1: Source text invariants (additional)
# ---------------------------------------------------------------------------


def test_source_text_preserves_original_case() -> None:
    """source_text always uses CorpusEntry.text, preserving case."""
    original = "The CApitalized NLP ReseaRCH Methods"
    sentence = _sentence(original)
    from citeguard.corpus.models import CorpusEntry
    metadata = CorpusMetadata(
        title="Paper", authors=["Smith"], year=2020,
        language=CorpusLanguage.ENGLISH, license="CC0",
        similarity_index_allowed=True,
    )
    entry = CorpusEntry(
        text=original,
        normalized_text=original.lower(),
        doc_id="doc-1", entry_index=0, metadata=metadata,
        char_offset=0, char_end=len(original),
    )
    fp = Fingerprint(
        points=winnow(generate_shingles(original.lower(), k=5), window=4)
    )
    engine = SimilarityEngine()
    matches = engine.compare_sentence_to_corpus(
        sentence,
        [("doc-1", original.lower(), fp, metadata, 0, entry)],
    )
    assert matches
    assert matches[0].source_text == original
    assert "CApitalized" in matches[0].source_text


def test_source_text_not_truncated_for_long_entries() -> None:
    """Very long source_text is not truncated."""
    long_text = "word " * 200  # 1000 chars
    sentence = _sentence(long_text)
    from citeguard.corpus.models import CorpusEntry
    metadata = CorpusMetadata(
        title="Paper", authors=["Smith"], year=2020,
        language=CorpusLanguage.ENGLISH, license="CC0",
        similarity_index_allowed=True,
    )
    entry = CorpusEntry(
        text=long_text,
        normalized_text=long_text.lower(),
        doc_id="doc-1", entry_index=0, metadata=metadata,
        char_offset=0, char_end=len(long_text),
    )
    fp = Fingerprint(
        points=winnow(generate_shingles(long_text.lower(), k=5), window=4)
    )
    engine = SimilarityEngine()
    matches = engine.compare_sentence_to_corpus(
        sentence,
        [("doc-1", long_text.lower(), fp, metadata, 0, entry)],
    )
    assert matches
    assert len(matches[0].source_text) == len(long_text)


# ---------------------------------------------------------------------------
# Checkpoint 1: Edge-case fingerprint tests
# ---------------------------------------------------------------------------


def test_fingerprint_empty_string() -> None:
    """Empty string produces an empty fingerprint (no points)."""
    engine = SimilarityEngine()
    fp = engine.build_fingerprint("")
    assert fp.points == []


def test_fingerprint_single_char() -> None:
    """Single character produces a valid fingerprint."""
    engine = SimilarityEngine()
    fp = engine.build_fingerprint("a")
    assert isinstance(fp.points, list)
    assert len(fp.points) >= 1


def test_fingerprint_very_long_text() -> None:
    """Very long text (10k+ chars) produces a fingerprint without error."""
    engine = SimilarityEngine()
    long_text = "the attention mechanism " * 500
    fp = engine.build_fingerprint(long_text)
    assert isinstance(fp.points, list)
    assert len(fp.points) > 0


def test_compare_sentence_empty_corpus() -> None:
    """Empty corpus returns no matches."""
    sentence = _sentence("some text")
    engine = SimilarityEngine()
    matches = engine.compare_sentence_to_corpus(sentence, [])
    assert matches == []


def test_compare_sentence_no_corpus_no_index() -> None:
    """Neither corpus_entries nor index → empty matches."""
    sentence = _sentence("some text")
    engine = SimilarityEngine()
    matches = engine.compare_sentence_to_corpus(sentence)
    assert matches == []
