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
    engine = SimilarityEngine()
    matches = engine.compare_sentence_to_corpus(
        sentence,
        [("doc-1", text.lower(), fp, metadata, 0)],
    )
    assert matches
    span = matches[0].matched_document_spans[0]
    assert span.start >= sentence.start_offset
    assert matches[0].matched_source_spans
    assert matches[0].source_authors == ["Vaswani"]
