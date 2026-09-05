from citeguard.matcher import match_claim_to_source
from citeguard.models import Claim, ClaimType, Severity, SourceCandidate, Verdict


def _make_claim(text: str, query: str = "") -> Claim:
    return Claim(
        text=text,
        search_query=query or text,
        claim_type=ClaimType.GENERAL_FACT,
        severity=Severity.MEDIUM,
        paragraph_index=0,
        has_existing_citation=False,
    )


def _make_candidate(
    title: str, abstract: str = "", *, doi: str = "10.1000/test"
) -> SourceCandidate:
    return SourceCandidate(
        title=title,
        authors=["Author"],
        year=2020,
        venue="Journal",
        doi=doi,
        url=None,
        abstract=abstract,
        source_api="test",
    )


def test_strong_match_returns_supported() -> None:
    claim = _make_claim(
        "Transformers achieve state-of-the-art in NLP",
        "transformers state-of-the-art NLP",
    )
    candidate = _make_candidate(
        "Transformers for State-of-the-Art NLP",
        "We show that transformers achieve state-of-the-art results in NLP tasks.",
    )
    metadata_score, support_score, verdict, reasoning = match_claim_to_source(
        claim, candidate
    )
    assert verdict == Verdict.SUPPORTED
    assert support_score >= 60
    assert metadata_score > 0


def test_weak_overlap_returns_insufficient_information() -> None:
    claim = _make_claim(
        "Quantum computing breaks RSA encryption",
        "quantum computing RSA encryption",
    )
    candidate = _make_candidate(
        "Organic farming techniques in modern agriculture",
        "A study of organic farming and sustainable agriculture methods.",
    )
    _, support_score, verdict, _ = match_claim_to_source(claim, candidate)
    assert verdict in (Verdict.INSUFFICIENT_INFORMATION, Verdict.UNRELATED)
    assert support_score < 40


def test_partial_support() -> None:
    claim = _make_claim(
        "Deep learning models improve image classification accuracy",
        "deep learning models image classification accuracy",
    )
    candidate = _make_candidate(
        "Deep Learning Models for Image Classification",
        "Deep learning models have been shown to improve image classification "
        "accuracy across multiple benchmark datasets including ImageNet and CIFAR.",
    )
    _, support_score, verdict, _ = match_claim_to_source(claim, candidate)
    assert verdict in (Verdict.PARTIALLY_SUPPORTED, Verdict.SUPPORTED)
    assert support_score > 0


def test_metadata_score_uses_query_tokens() -> None:
    claim = _make_claim(
        "Some claim text",
        "machine learning neural networks",
    )
    candidate = _make_candidate(
        "Machine Learning with Neural Networks",
        "An overview of machine learning.",
    )
    metadata_score, _, _, _ = match_claim_to_source(claim, candidate)
    assert metadata_score > 0


def test_no_abstract_gives_low_support() -> None:
    claim = _make_claim(
        "Transformers improve NLP performance significantly",
        "transformers NLP performance",
    )
    candidate = _make_candidate(
        "Unrelated Title About Chemistry",
        "",
    )
    _, support_score, verdict, _ = match_claim_to_source(claim, candidate)
    assert support_score < 50


def test_reasoning_is_nonempty() -> None:
    claim = _make_claim("Some claim", "some claim query")
    candidate = _make_candidate("Some title", "Some abstract text.")
    _, _, _, reasoning = match_claim_to_source(claim, candidate)
    assert len(reasoning) > 10


def test_scores_are_bounded() -> None:
    claim = _make_claim("A claim", "claim query")
    candidate = _make_candidate("A title", "An abstract.")
    metadata_score, support_score, _, _ = match_claim_to_source(claim, candidate)
    assert 0 <= metadata_score <= 100
    assert 0 <= support_score <= 100
