"""Tests for entailment evaluation."""

import json

from citeguard.entailment import (
    contradiction_risk,
    evaluate_evidence_with_llm,
)
from citeguard.models import (
    Claim,
    ClaimType,
    Evidence,
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


def _make_evidence(text: str) -> Evidence:
    return Evidence(
        text=text,
        source_title="Test Source",
        source_api="test",
        evidence_type=EvidenceType.ABSTRACT,
    )


def _make_candidate() -> SourceCandidate:
    return SourceCandidate(
        title="Test",
        authors=["Author"],
        year=2020,
        venue="Journal",
        doi="10.1000/test",
        url=None,
        abstract="Test abstract.",
        source_api="test",
    )


def test_supported_no_negation_high_overlap() -> None:
    claim = _make_claim("Smoking increases cancer risk.")
    result = contradiction_risk(claim, "Smoking was associated with increased cancer risk.")
    assert result.verdict == Verdict.PARTIALLY_SUPPORTED
    assert result.confidence > 0


def test_contradicted_with_negation() -> None:
    claim = _make_claim("Vitamin D reduces cancer risk.")
    result = contradiction_risk(claim, "Vitamin D did not reduce cancer risk.")
    # Offline mode: negation + high overlap → INSUFFICIENT_INFORMATION
    # (contradiction risk elevated, but final CONTRADICTED requires LLM)
    assert result.verdict == Verdict.INSUFFICIENT_INFORMATION
    assert result.confidence > 30
    assert "contradiction" in result.reasoning.lower() or "negation" in result.reasoning.lower()


def test_insufficient_low_overlap() -> None:
    claim = _make_claim("Machine learning improves healthcare.")
    result = contradiction_risk(claim, "The weather is sunny today.")
    assert result.verdict == Verdict.INSUFFICIENT_INFORMATION


def test_negation_low_overlap() -> None:
    claim = _make_claim("Quantum computing solves optimization.")
    result = contradiction_risk(claim, "The results were not significant for the sample.")
    assert result.verdict == Verdict.INSUFFICIENT_INFORMATION


def test_reasoning_is_nonempty() -> None:
    claim = _make_claim("Test claim.")
    result = contradiction_risk(claim, "Some evidence text here.")
    assert len(result.reasoning) > 0


def test_confidence_bounded() -> None:
    claim = _make_claim("Test claim about topic.")
    result = contradiction_risk(claim, "Evidence about the same topic with details.")
    assert 0 <= result.confidence <= 100


def test_llm_returns_none_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    claim = _make_claim("Test.")
    evidence = [_make_evidence("Evidence text.")]
    candidate = _make_candidate()
    result = evaluate_evidence_with_llm(claim, evidence, candidate)
    assert result is None


def test_llm_returns_none_with_empty_evidence(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    claim = _make_claim("Test.")
    candidate = _make_candidate()
    result = evaluate_evidence_with_llm(claim, [], candidate)
    assert result is None


def test_llm_parse_verdict(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    fake_response = json.dumps({
        "verdict": "partially_supported",
        "confidence": 75,
        "reasoning": "The evidence partially supports the claim.",
    })

    def _fake_call(api_key, system, user_message, *, model="m", max_tokens=2048, timeout=30):
        return fake_response

    import citeguard.entailment as ent_mod
    monkeypatch.setattr(ent_mod, "_call_anthropic", _fake_call)

    claim = _make_claim("Smoking causes cancer.")
    evidence = [_make_evidence("Smoking was associated with cancer.")]
    candidate = _make_candidate()
    result = evaluate_evidence_with_llm(claim, evidence, candidate)
    assert result is not None
    assert result.verdict == Verdict.PARTIALLY_SUPPORTED
    assert result.confidence == 75


def test_llm_parse_invalid_verdict(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    fake_response = json.dumps({
        "verdict": "maybe",
        "confidence": "high",
        "reasoning": "",
    })

    def _fake_call(api_key, system, user_message, *, model="m", max_tokens=2048, timeout=30):
        return fake_response

    import citeguard.entailment as ent_mod
    monkeypatch.setattr(ent_mod, "_call_anthropic", _fake_call)

    claim = _make_claim("Test.")
    evidence = [_make_evidence("Evidence.")]
    candidate = _make_candidate()
    result = evaluate_evidence_with_llm(claim, evidence, candidate)
    assert result is not None
    assert result.verdict == Verdict.INSUFFICIENT_INFORMATION
    assert result.confidence == 0
