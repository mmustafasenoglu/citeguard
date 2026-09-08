"""Before/after measurements for attribution-risk reduction."""

from __future__ import annotations

from dataclasses import dataclass

from ..similarity.models import SimilarityEngineResult


@dataclass(frozen=True, slots=True)
class ReductionMetrics:
    """Comparable measurements from two similarity analyses."""

    before_textual_similarity_pct: float
    after_textual_similarity_pct: float
    absolute_reduction_pct: float
    relative_reduction_pct: float | None
    before_high_risk: int
    after_high_risk: int
    rewritten_passages: int = 0
    rejected_candidates: int = 0
    meaning_preservation_avg: float | None = None
    citation_integrity_passed: bool | None = None
    new_unsupported_claims: int = 0


def compute_reduction_metrics(
    before: SimilarityEngineResult,
    after: SimilarityEngineResult,
    *,
    rewritten_passages: int = 0,
    rejected_candidates: int = 0,
    meaning_scores: list[float] | None = None,
    citation_integrity_passed: bool | None = None,
    new_unsupported_claims: int = 0,
) -> ReductionMetrics:
    """Compute overlap reduction without interpreting it as plagiarism removal."""
    absolute = before.overall_similarity_pct - after.overall_similarity_pct
    relative = (
        absolute / before.overall_similarity_pct
        if before.overall_similarity_pct > 0
        else None
    )
    average = (
        sum(meaning_scores) / len(meaning_scores)
        if meaning_scores
        else None
    )
    return ReductionMetrics(
        before_textual_similarity_pct=before.overall_similarity_pct,
        after_textual_similarity_pct=after.overall_similarity_pct,
        absolute_reduction_pct=absolute,
        relative_reduction_pct=relative,
        before_high_risk=before.high_risk_count,
        after_high_risk=after.high_risk_count,
        rewritten_passages=rewritten_passages,
        rejected_candidates=rejected_candidates,
        meaning_preservation_avg=average,
        citation_integrity_passed=citation_integrity_passed,
        new_unsupported_claims=new_unsupported_claims,
    )
