"""Regression tests for similarity engine wiring."""

from __future__ import annotations

from citeguard.corpus.models import CorpusLanguage, CorpusMetadata
from citeguard.models import BibliographyEntry, ExistingCitation, Sentence
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
