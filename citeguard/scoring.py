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


@dataclass(frozen=True, slots=True)
class ProductMetrics:
    """Product-level audit metrics with explicit availability semantics.

    Ratio fields are ``None`` when the underlying evaluation could not be
    performed (e.g. offline cache miss, total provider failure), which is
    distinct from a measured perfect score.  JSON consumers must treat
    ``None`` as "unavailable", never as 100%.
    """

    total_claims: int
    claims_requiring_citations: int
    cited_claims: int
    verified_citations: int
    partially_verified: int
    weak_cited_source_matches: int
    weak_suggestions: int
    unresolved_citations: int
    uncited_high_severity_claims: int
    contradictions: int
    bibliography_issues: int
    citation_coverage: float
    verification_ratio: float | None
    support_ratio: float | None
    bibliography_consistency: float
    evidence_coverage: float | None
    health_score: int | None
    health_score_complete: bool
    unavailable_metrics: tuple[str, ...]


def compute_product_metrics(
    *,
    total_claims: int,
    cited_claims: int,
    verified_citations: int,
    partially_verified: int,
    bib_evaluated: int,
    bib_total: int,
    supported_cited_claims: int,
    contradicted_assessments: int,
    cited_evaluated: int,
    cited_with_evidence: int,
    weak_cited_source_matches: int,
    weak_suggestions: int,
    uncited_high: int,
    unresolved_high: int,
    bibliography_issues: int,
    bibliography_entries: int,
    distinct_citations: int,
) -> ProductMetrics:
    """Compute product metrics from already-separated audit signals.

    Definitions:
    - ``citation_coverage`` = cited / total (1.0 when there are no claims).
    - ``verification_ratio`` = VERIFIED bib entries / evaluated bib entries;
      1.0 when there are no bibliography entries; None when entries exist
      but none could be evaluated.  PARTIALLY_VERIFIED never counts.
    - ``support_ratio`` = cited claims with >= 1 supported resolved source /
      cited claims evaluated for support; None when none were evaluated.
      Uncited suggestions are excluded.
    - ``evidence_coverage`` = evaluated cited claims with usable evidence /
      evaluated cited claims; None when none were evaluated.
    - ``bibliography_consistency`` = 1 - issues /
      max(entries + distinct in-text citations, 1), clamped to 0..1.
    - ``health_score`` uses the established SPEC weights; None (with
      ``health_score_complete`` False) when any required component is
      unavailable instead of substituting 100%.
    """
    claims_requiring = total_claims - cited_claims
    citation_coverage = cited_claims / total_claims if total_claims > 0 else 1.0

    if bib_total <= 0:
        verification_ratio: float | None = 1.0
    elif bib_evaluated <= 0:
        verification_ratio = None
    else:
        verification_ratio = verified_citations / bib_evaluated

    if cited_evaluated <= 0:
        support_ratio: float | None = None
        evidence_coverage: float | None = None
    else:
        support_ratio = supported_cited_claims / cited_evaluated
        evidence_coverage = cited_with_evidence / cited_evaluated

    denom = max(bibliography_entries + distinct_citations, 1)
    bibliography_consistency = max(
        0.0, min(1.0, 1.0 - (bibliography_issues / denom))
    )

    unavailable = [
        name
        for name, value in (
            ("verification_ratio", verification_ratio),
            ("support_ratio", support_ratio),
            ("evidence_coverage", evidence_coverage),
        )
        if value is None
    ]
    if unavailable:
        health: int | None = None
        complete = False
    else:
        assert verification_ratio is not None
        assert support_ratio is not None
        assert evidence_coverage is not None
        health = compute_health_score(
            citation_coverage=citation_coverage,
            verification_ratio=verification_ratio,
            support_ratio=support_ratio,
            bibliography_consistency=bibliography_consistency,
            evidence_coverage=evidence_coverage,
            uncited_high=uncited_high,
            contradictions=contradicted_assessments,
            unresolved_high=unresolved_high,
        )
        complete = True

    return ProductMetrics(
        total_claims=total_claims,
        claims_requiring_citations=claims_requiring,
        cited_claims=cited_claims,
        verified_citations=verified_citations,
        partially_verified=partially_verified,
        weak_cited_source_matches=weak_cited_source_matches,
        weak_suggestions=weak_suggestions,
        unresolved_citations=cited_claims - supported_cited_claims,
        uncited_high_severity_claims=uncited_high,
        contradictions=contradicted_assessments,
        bibliography_issues=bibliography_issues,
        citation_coverage=round(citation_coverage, 3),
        verification_ratio=(
            round(verification_ratio, 3) if verification_ratio is not None else None
        ),
        support_ratio=round(support_ratio, 3) if support_ratio is not None else None,
        bibliography_consistency=round(bibliography_consistency, 3),
        evidence_coverage=(
            round(evidence_coverage, 3) if evidence_coverage is not None else None
        ),
        health_score=health,
        health_score_complete=complete,
        unavailable_metrics=tuple(unavailable),
    )


def _validate_score(value: int) -> None:
    if not 0 <= value <= 100:
        raise ValueError("Scores must be between 0 and 100.")
