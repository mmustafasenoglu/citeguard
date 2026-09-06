"""Deterministic audit metrics, health score, and priority scoring."""

from __future__ import annotations

from dataclasses import dataclass

from .models import (
    BibliographyIssue,
    Claim,
    Severity,
    Verdict,
    VerificationResult,
)

SEVERITY_WEIGHT = {Severity.HIGH: 3, Severity.MEDIUM: 2, Severity.LOW: 1}


@dataclass(frozen=True, slots=True)
class AuditMetrics:
    total_claims: int
    claims_requiring_citations: int
    cited_claims: int
    verified_citations: int
    weak_matches: int
    unresolved_citations: int
    uncited_high_severity_claims: int
    contradictions: int
    bibliography_issues: int
    citation_coverage: float
    verification_ratio: float
    support_ratio: float
    bibliography_consistency: float
    evidence_coverage: float
    health_score: int


def overall_confidence(
    metadata_match_score: int,
    claim_support_score: int,
    *,
    has_entailment: bool = False,
) -> int:
    """Combine metadata and support into a single confidence score.

    ``claim_support_score`` is a standalone signal that does NOT include
    metadata (it is evidence + entailment only).  Metadata is added here
    exactly once.

    With entailment:   metadata 20% + support 80%
    Without entailment: metadata 40% + support 60%
    """
    _validate_score(metadata_match_score)
    _validate_score(claim_support_score)
    if has_entailment:
        return round(metadata_match_score * 0.20 + claim_support_score * 0.80)
    return round(metadata_match_score * 0.40 + claim_support_score * 0.60)


def priority_score(result: VerificationResult) -> float:
    severity = SEVERITY_WEIGHT[result.claim.severity]
    confidence = result.matched.overall_confidence if result.matched else 0
    risk = 100 - confidence
    return severity * 100 + risk


def _has_entailment(result: VerificationResult) -> bool:
    if result.matched is None:
        return False
    return result.matched.entailment_score is not None


def compute_health_score(
    *,
    citation_coverage: float,
    verification_ratio: float,
    support_ratio: float,
    bibliography_consistency: float,
    evidence_coverage: float,
    uncited_high: int = 0,
    contradictions: int = 0,
    unresolved_high: int = 0,
) -> int:
    scores = (
        citation_coverage,
        verification_ratio,
        support_ratio,
        bibliography_consistency,
        evidence_coverage,
    )
    for value in scores:
        if not 0 <= value <= 1:
            raise ValueError("Score ratios must be between 0 and 1.")

    score = (
        citation_coverage * 100 * 0.25
        + verification_ratio * 100 * 0.25
        + support_ratio * 100 * 0.20
        + bibliography_consistency * 100 * 0.10
        + evidence_coverage * 100 * 0.20
    )
    score -= uncited_high * 2
    score -= contradictions * 4
    score -= unresolved_high * 2
    return max(0, min(round(score), 100))


def compute_audit_metrics(
    *,
    claims: list[Claim],
    verification_results: list[VerificationResult],
    bibliography_issues: list[BibliographyIssue],
) -> AuditMetrics:
    """Compute deterministic audit metrics for the Citation Health Score."""
    total_claims = len(claims)
    claims_requiring = [c for c in claims if not c.has_existing_citation]
    cited_claims = [c for c in claims if c.has_existing_citation]

    # Citation coverage: fraction of claims that have at least one citation.
    citation_coverage = (
        len(cited_claims) / total_claims if total_claims > 0 else 1.0
    )

    # Verification ratio: fraction of verification results that are verified.
    verified = [r for r in verification_results if r.status.value == "verified"]
    verification_ratio = (
        len(verified) / len(verification_results) if verification_results else 1.0
    )

    # Support ratio: fraction of results with supported/partially_supported verdict.
    supported = [
        r
        for r in verification_results
        if r.matched is not None
        and r.matched.verdict in (Verdict.SUPPORTED, Verdict.PARTIALLY_SUPPORTED)
    ]
    support_ratio = (
        len(supported) / len(verification_results) if verification_results else 1.0
    )

    # Evidence coverage: fraction of verified claims that have evidence.
    verified_with_evidence = [
        r
        for r in verified
        if r.matched is not None and r.matched.evidence
    ]
    evidence_coverage = (
        len(verified_with_evidence) / len(verified) if verified else 1.0
    )

    # Bibliography consistency: 1 minus normalized issue count.
    bib_issue_count = len(bibliography_issues)
    max_expected_issues = max(total_claims, 1)
    bibliography_consistency = max(0.0, 1.0 - (bib_issue_count / max_expected_issues))

    # Counts for penalties.
    uncited_high = sum(
        1 for c in claims_requiring if c.severity == Severity.HIGH
    )
    contradiction_count = sum(
        1
        for r in verification_results
        if r.matched is not None and r.matched.verdict == Verdict.CONTRADICTED
    )
    unresolved_high = sum(
        1
        for r in verification_results
        if r.status.value != "verified"
        and r.claim.severity == Severity.HIGH
    )

    weak_matches = sum(
        1 for r in verification_results if r.matched is not None and
        r.matched.overall_confidence < 60
    )

    health = compute_health_score(
        citation_coverage=citation_coverage,
        verification_ratio=verification_ratio,
        support_ratio=support_ratio,
        bibliography_consistency=bibliography_consistency,
        evidence_coverage=evidence_coverage,
        uncited_high=uncited_high,
        contradictions=contradiction_count,
        unresolved_high=unresolved_high,
    )

    return AuditMetrics(
        total_claims=total_claims,
        claims_requiring_citations=len(claims_requiring),
        cited_claims=len(cited_claims),
        verified_citations=len(verified),
        weak_matches=weak_matches,
        unresolved_citations=len(verification_results) - len(verified),
        uncited_high_severity_claims=uncited_high,
        contradictions=contradiction_count,
        bibliography_issues=bib_issue_count,
        citation_coverage=round(citation_coverage, 3),
        verification_ratio=round(verification_ratio, 3),
        support_ratio=round(support_ratio, 3),
        bibliography_consistency=round(bibliography_consistency, 3),
        evidence_coverage=round(evidence_coverage, 3),
        health_score=health,
    )


def priority_list(results: list[VerificationResult]) -> list[VerificationResult]:
    """Return verification results sorted by descending priority score."""
    return sorted(results, key=lambda r: priority_score(r), reverse=True)


def _validate_score(value: int) -> None:
    if not 0 <= value <= 100:
        raise ValueError("Scores must be between 0 and 100.")
