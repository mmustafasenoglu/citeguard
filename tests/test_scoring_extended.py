from citeguard.models import (
    Claim,
    ClaimType,
    MatchResult,
    Severity,
    SourceCandidate,
    Verdict,
    VerificationResult,
    VerificationStatus,
)
from citeguard.scoring import (
    compute_audit_metrics,
    priority_list,
    priority_score,
)


def _make_claim(text: str = "A claim", *, severity: Severity = Severity.MEDIUM) -> Claim:
    return Claim(
        text=text,
        search_query="claim query",
        claim_type=ClaimType.GENERAL_FACT,
        severity=severity,
        paragraph_index=0,
        has_existing_citation=False,
    )


def _make_match(
    confidence: int = 70, verdict: Verdict = Verdict.SUPPORTED
) -> MatchResult:
    return MatchResult(
        candidate=SourceCandidate(
            title="Test",
            authors=[],
            year=2020,
            venue="J",
            doi="10.1/x",
            url=None,
            abstract=None,
            source_api="test",
        ),
        source_exists=True,
        metadata_match_score=70,
        claim_support_score=70,
        overall_confidence=confidence,
        verdict=verdict,
        reasoning="test",
    )


def test_compute_audit_metrics_empty() -> None:
    metrics = compute_audit_metrics(
        claims=[],
        verification_results=[],
        bibliography_issues=[],
    )
    assert metrics.total_claims == 0
    assert metrics.health_score == 100
    assert metrics.citation_coverage == 1.0
    assert metrics.verification_ratio == 1.0


def test_compute_audit_metrics_with_claims() -> None:
    claims = [
        _make_claim(severity=Severity.HIGH),
        _make_claim(severity=Severity.HIGH),
        _make_claim(severity=Severity.MEDIUM),
    ]
    # All uncited, no verification results
    metrics = compute_audit_metrics(
        claims=claims,
        verification_results=[],
        bibliography_issues=[],
    )
    assert metrics.total_claims == 3
    assert metrics.claims_requiring_citations == 3
    assert metrics.uncited_high_severity_claims == 2
    assert metrics.health_score < 100


def test_priority_score_orders_higher_risk_first() -> None:
    high_risk = VerificationResult(
        claim=_make_claim(severity=Severity.HIGH),
        status=VerificationStatus.NOT_FOUND,
        citation=None,
        matched=None,
    )
    low_risk = VerificationResult(
        claim=_make_claim(severity=Severity.LOW),
        status=VerificationStatus.NOT_FOUND,
        citation=None,
        matched=_make_match(confidence=90, verdict=Verdict.SUPPORTED),
    )
    assert priority_score(high_risk) > priority_score(low_risk)


def test_priority_list_sorts_by_risk() -> None:
    results = [
        VerificationResult(
            claim=_make_claim(severity=Severity.LOW),
            status=VerificationStatus.SUGGESTED,
            citation=None,
            matched=_make_match(confidence=80),
        ),
        VerificationResult(
            claim=_make_claim(severity=Severity.HIGH),
            status=VerificationStatus.NOT_FOUND,
            citation=None,
            matched=None,
        ),
    ]
    sorted_results = priority_list(results)
    assert sorted_results[0].claim.severity == Severity.HIGH
    assert sorted_results[1].claim.severity == Severity.LOW
