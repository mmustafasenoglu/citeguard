"""Before/after measurements for attribution-risk reduction."""

from __future__ import annotations

from dataclasses import dataclass

from ..similarity.models import MatchType, SimilarityEngineResult


@dataclass(frozen=True, slots=True)
class ReductionMetrics:
    """Comparable measurements from two similarity analyses."""

    before_textual_similarity_pct: float
    after_textual_similarity_pct: float
    absolute_reduction_points: float
    relative_reduction_pct: float | None
    before_high_risk: int
    after_high_risk: int
    before_medium_risk: int = 0
    after_medium_risk: int = 0
    rewritten_passages: int = 0
    unchanged_passages: int = 0
    manual_review_passages: int = 0
    generated_candidates: int = 0
    rejected_candidates: int = 0
    accepted_candidates: int = 0
    meaning_preservation_avg: float | None = None
    citation_integrity_passed: bool | None = None
    numeric_integrity_passed: bool | None = None
    named_entity_integrity_passed: bool | None = None
    new_unsupported_claims: int = 0
    iterations_completed: int = 0
    stop_reason: str = "not_started"
    application_format: str | None = None
    before_exact_overlap_pct: float = 0.0
    after_exact_overlap_pct: float = 0.0
    before_lexical_overlap_pct: float = 0.0
    after_lexical_overlap_pct: float = 0.0
    before_semantic_matches: int = 0
    after_semantic_matches: int = 0

    @property
    def absolute_reduction_pct(self) -> float:
        """Deprecated compatibility alias; the value is percentage points."""
        return self.absolute_reduction_points


def compute_reduction_metrics(
    before: SimilarityEngineResult,
    after: SimilarityEngineResult,
    *,
    rewritten_passages: int = 0,
    rejected_candidates: int = 0,
    meaning_scores: list[float] | None = None,
    citation_integrity_passed: bool | None = None,
    new_unsupported_claims: int = 0,
    unchanged_passages: int = 0,
    manual_review_passages: int = 0,
    generated_candidates: int = 0,
    accepted_candidates: int = 0,
    numeric_integrity_passed: bool | None = None,
    named_entity_integrity_passed: bool | None = None,
    iterations_completed: int = 0,
    stop_reason: str = "complete",
    application_format: str | None = None,
) -> ReductionMetrics:
    """Compute overlap reduction without interpreting it as plagiarism removal."""
    absolute = before.overall_similarity_pct - after.overall_similarity_pct
    relative = (
        (absolute / before.overall_similarity_pct) * 100
        if before.overall_similarity_pct > 0
        else None
    )
    average = sum(meaning_scores) / len(meaning_scores) if meaning_scores else None
    before_exact, before_lexical, before_semantic = _overlap_breakdown(before)
    after_exact, after_lexical, after_semantic = _overlap_breakdown(after)
    return ReductionMetrics(
        before_textual_similarity_pct=before.overall_similarity_pct,
        after_textual_similarity_pct=after.overall_similarity_pct,
        absolute_reduction_points=absolute,
        relative_reduction_pct=relative,
        before_high_risk=before.high_risk_count,
        after_high_risk=after.high_risk_count,
        before_medium_risk=before.medium_risk_count,
        after_medium_risk=after.medium_risk_count,
        rewritten_passages=rewritten_passages,
        unchanged_passages=unchanged_passages,
        manual_review_passages=manual_review_passages,
        generated_candidates=generated_candidates,
        rejected_candidates=rejected_candidates,
        accepted_candidates=accepted_candidates,
        meaning_preservation_avg=average,
        citation_integrity_passed=citation_integrity_passed,
        numeric_integrity_passed=numeric_integrity_passed,
        named_entity_integrity_passed=named_entity_integrity_passed,
        new_unsupported_claims=new_unsupported_claims,
        iterations_completed=iterations_completed,
        stop_reason=stop_reason,
        application_format=application_format,
        before_exact_overlap_pct=before_exact,
        after_exact_overlap_pct=after_exact,
        before_lexical_overlap_pct=before_lexical,
        after_lexical_overlap_pct=after_lexical,
        before_semantic_matches=before_semantic,
        after_semantic_matches=after_semantic,
    )


def _overlap_breakdown(result: SimilarityEngineResult) -> tuple[float, float, int]:
    """Measure unique exact/lexical character coverage and semantic match count."""
    exact: set[tuple[int, int]] = set()
    lexical: set[tuple[int, int]] = set()
    semantic = 0
    for item in result.results:
        for match in item.matches:
            if match.match_type == MatchType.SEMANTIC_OVERLAP:
                semantic += 1
                continue
            target = exact if match.match_type == MatchType.EXACT else lexical
            for span in match.matched_document_spans:
                target.update(
                    (span.paragraph_index, offset) for offset in range(span.start, span.end)
                )
    lexical -= exact
    denominator = max(result.eligible_chars, 1)
    return (
        round(len(exact) * 100.0 / denominator, 2),
        round(len(lexical) * 100.0 / denominator, 2),
        semantic,
    )
