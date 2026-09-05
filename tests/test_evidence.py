"""Tests for abstract evidence extraction and ranking."""

from citeguard.evidence import (
    extract_evidence,
    split_evidence_sentences,
)
from citeguard.models import (
    Claim,
    ClaimType,
    EvidenceType,
    Severity,
    SourceCandidate,
    Verdict,
)


def _make_claim(text: str) -> Claim:
    return Claim(
        text=text,
        search_query=text,
        claim_type=ClaimType.GENERAL_FACT,
        severity=Severity.MEDIUM,
        paragraph_index=0,
        has_existing_citation=False,
    )


def _make_candidate(abstract: str, *, title: str = "Test Source") -> SourceCandidate:
    return SourceCandidate(
        title=title,
        authors=["Author"],
        year=2020,
        venue="Journal",
        doi="10.1000/test",
        url=None,
        abstract=abstract,
        source_api="test",
    )


def test_split_sentences_basic() -> None:
    text = "First sentence. Second sentence. Third sentence."
    result = split_evidence_sentences(text)
    assert len(result) == 3


def test_split_sentences_skips_short() -> None:
    text = "A valid sentence here. No. Another one."
    result = split_evidence_sentences(text)
    assert all("No" not in s or len(s.split()) >= 3 for s in result)


def test_claim_and_abstract_same_topic() -> None:
    claim = _make_claim("Social media use is associated with depression.")
    candidate = _make_candidate(
        "Social media use has increased among adolescents. "
        "Higher social media usage was associated with depressive symptoms."
    )
    evidence = extract_evidence(claim, candidate)
    assert len(evidence) >= 1
    assert evidence[0].lexical_score > 0
    assert evidence[0].evidence_type == EvidenceType.ABSTRACT


def test_unrelated_abstract_gives_low_score() -> None:
    claim = _make_claim("Quantum computing solves optimization problems.")
    candidate = _make_candidate(
        "The recipe for chocolate cake includes flour and sugar."
    )
    evidence = extract_evidence(claim, candidate)
    assert len(evidence) >= 1
    assert evidence[0].lexical_score < 20


def test_empty_abstract_returns_empty() -> None:
    claim = _make_claim("Some claim.")
    candidate = _make_candidate("")
    assert extract_evidence(claim, candidate) == []


def test_none_abstract_returns_empty() -> None:
    claim = _make_claim("Some claim.")
    candidate = SourceCandidate(
        title="Test",
        authors=[],
        year=None,
        venue=None,
        doi=None,
        url=None,
        abstract=None,
        source_api="test",
    )
    assert extract_evidence(claim, candidate) == []


def test_top_evidence_selection() -> None:
    claim = _make_claim("Smoking increases cancer risk.")
    candidate = _make_candidate(
        "Smoking was associated with increased cancer risk in the study. "
        "The methodology involved a large cohort. "
        "No significant effect was found for alcohol consumption."
    )
    evidence = extract_evidence(claim, candidate, max_passages=2)
    assert len(evidence) <= 2
    scores = [e.lexical_score for e in evidence]
    assert scores == sorted(scores, reverse=True)


def test_duplicate_sentences_not_duplicated() -> None:
    claim = _make_claim("Smoking increases cancer risk.")
    candidate = _make_candidate(
        "Smoking increases cancer risk. "
        "Smoking increases cancer risk."
    )
    evidence = extract_evidence(claim, candidate)
    texts = [e.text for e in evidence]
    assert len(texts) == len(set(texts))


def test_long_abstract_top_3() -> None:
    claim = _make_claim("Machine learning improves healthcare outcomes.")
    sentences = [f"Sentence {i} about machine learning in healthcare. " for i in range(10)]
    candidate = _make_candidate("".join(sentences))
    evidence = extract_evidence(claim, candidate)
    assert len(evidence) <= 3


def test_verdict_from_score() -> None:
    claim = _make_claim("Test claim about a topic.")
    candidate = _make_candidate("A highly relevant sentence about the topic.")
    evidence = extract_evidence(claim, candidate)
    for e in evidence:
        assert e.verdict in (Verdict.PARTIALLY_SUPPORTED, Verdict.INSUFFICIENT_INFORMATION)
