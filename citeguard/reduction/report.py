"""JSON-safe reduction report serialization."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from .metrics import ReductionMetrics
from .models import FixAction, FixPlan, MeaningVerdict, PassageRisk, RewriteCandidate

if TYPE_CHECKING:
    from .service import ReductionResult

REDUCTION_SCHEMA_VERSION = "2"


def _academic_action(action: FixAction, risk_type: str | None = None) -> str:
    """Expose the product action while retaining legacy action values."""
    if risk_type == "exact_copy":
        return FixAction.QUOTE_AND_CITE.value
    if risk_type == "missing_citation" and action == FixAction.ADD_CITATION:
        return FixAction.PARAPHRASE_WITH_CITATION.value
    return {
        FixAction.ADD_CITATION: FixAction.ADD_CITATION.value,
        FixAction.ADD_QUOTATION: FixAction.QUOTE_AND_CITE.value,
        FixAction.PARAPHRASE: FixAction.PARAPHRASE_WITH_CITATION.value,
        FixAction.MANUAL_REVIEW: FixAction.MANUAL_REVIEW.value,
        FixAction.LEAVE: "no_change",
    }.get(action, action.value)


def reduction_report(
    risks: list[PassageRisk],
    plans: list[FixPlan],
    *,
    candidates: dict[str, list[RewriteCandidate]] | None = None,
) -> dict[str, Any]:
    """Serialize reduction planning and candidate evidence."""
    candidate_map = candidates or {}
    risk_types = {risk.passage_id: risk.risk_type.value for risk in risks}
    return {
        "schema_version": REDUCTION_SCHEMA_VERSION,
        "risks": [
            {
                "passage_id": risk.passage_id,
                "text": risk.text,
                "paragraph_index": risk.paragraph_index,
                "start_offset": risk.start_offset,
                "end_offset": risk.end_offset,
                "exact_overlap": risk.exact_overlap,
                "lexical_similarity": risk.lexical_similarity,
                "semantic_similarity_raw": risk.semantic_similarity_raw,
                "attribution_risk": risk.attribution_risk.value,
                "has_citation": risk.has_citation,
                "citation_texts": list(risk.citation_texts),
                "risk_type": risk.risk_type.value,
                "recommended_action": risk.recommended_action.value,
                "confidence": risk.confidence,
                "source_title": risk.source_title,
                "source_id": risk.source_id,
            }
            for risk in risks
        ],
        "plans": [
            {
                "passage_id": plan.passage_id,
                "action": plan.action.value,
                "academic_action": _academic_action(plan.action, risk_types.get(plan.passage_id)),
                "reason": plan.reason,
                "preserve_citations": list(plan.preserve_citations),
                "must_preserve_numbers": list(plan.must_preserve_numbers),
                "rewrite_allowed": plan.rewrite_allowed,
            }
            for plan in plans
        ],
        "candidates": {
            passage_id: [
                {
                    "text": candidate.text,
                    "generator": candidate.generator,
                    "score": candidate.score,
                    "accepted": (
                        not candidate.rejection_reasons
                        and candidate.citations_preserved is True
                        and candidate.numeric_integrity is True
                        and candidate.named_entity_integrity is True
                        and candidate.factual_integrity is True
                        and candidate.meaning_verdict == MeaningVerdict.PRESERVED
                        and not candidate.introduced_claims
                        and candidate.source_overlap_improved is True
                    ),
                    "rejection_reasons": candidate.rejection_reasons,
                    "meaning_score": candidate.meaning_score,
                    "source_support_score": candidate.source_support_score,
                    "citations_preserved": candidate.citations_preserved,
                    "numeric_integrity": candidate.numeric_integrity,
                    "named_entity_integrity": candidate.named_entity_integrity,
                    "factual_integrity": candidate.factual_integrity,
                    "semantic_similarity_to_original": (candidate.semantic_similarity_to_original),
                    "source_exact_overlap_before": candidate.source_exact_overlap_before,
                    "source_exact_overlap_after": candidate.source_exact_overlap_after,
                    "source_lexical_similarity_before": (
                        candidate.source_lexical_similarity_before
                    ),
                    "source_lexical_similarity_after": (candidate.source_lexical_similarity_after),
                    "source_overlap_improved": candidate.source_overlap_improved,
                    "source_overlap_delta": candidate.source_overlap_delta,
                    "forward_entailment_score": candidate.forward_entailment_score,
                    "backward_entailment_score": candidate.backward_entailment_score,
                    "meaning_verdict": (
                        candidate.meaning_verdict.value if candidate.meaning_verdict else None
                    ),
                }
                for candidate in values
            ]
            for passage_id, values in candidate_map.items()
        },
    }


def metrics_report(metrics: ReductionMetrics) -> dict[str, Any]:
    """Serialize before/after measurements for JSON and Markdown callers."""
    return {
        "before_textual_similarity_pct": metrics.before_textual_similarity_pct,
        "after_textual_similarity_pct": metrics.after_textual_similarity_pct,
        "absolute_reduction_points": metrics.absolute_reduction_points,
        "relative_reduction_pct": metrics.relative_reduction_pct,
        "before_high_risk": metrics.before_high_risk,
        "after_high_risk": metrics.after_high_risk,
        "before_medium_risk": metrics.before_medium_risk,
        "after_medium_risk": metrics.after_medium_risk,
        "rewritten_passages": metrics.rewritten_passages,
        "unchanged_passages": metrics.unchanged_passages,
        "manual_review_passages": metrics.manual_review_passages,
        "generated_candidates": metrics.generated_candidates,
        "rejected_candidates": metrics.rejected_candidates,
        "accepted_candidates": metrics.accepted_candidates,
        "meaning_preservation_avg": metrics.meaning_preservation_avg,
        "citation_integrity_passed": metrics.citation_integrity_passed,
        "numeric_integrity_passed": metrics.numeric_integrity_passed,
        "named_entity_integrity_passed": metrics.named_entity_integrity_passed,
        "new_unsupported_claims": metrics.new_unsupported_claims,
        "iterations_completed": metrics.iterations_completed,
        "stop_reason": metrics.stop_reason,
        "application_format": metrics.application_format,
        "before_exact_overlap_pct": metrics.before_exact_overlap_pct,
        "after_exact_overlap_pct": metrics.after_exact_overlap_pct,
        "before_lexical_overlap_pct": metrics.before_lexical_overlap_pct,
        "after_lexical_overlap_pct": metrics.after_lexical_overlap_pct,
        "before_semantic_matches": metrics.before_semantic_matches,
        "after_semantic_matches": metrics.after_semantic_matches,
    }


def reduction_result_report(result: ReductionResult) -> dict[str, Any]:
    """Serialize a complete end-to-end reduction result as schema v2."""
    planning = reduction_report(result.risks, result.plans)
    before = {
        "textual_similarity_pct": result.before.overall_similarity_pct,
        "high_risk": result.before.high_risk_count,
        "medium_risk": result.before.medium_risk_count,
    }
    after = {
        "textual_similarity_pct": result.after.overall_similarity_pct,
        "high_risk": result.after.high_risk_count,
        "medium_risk": result.after.medium_risk_count,
    }
    changes = [asdict(change) for change in result.changes]
    return {
        "schema_version": REDUCTION_SCHEMA_VERSION,
        "source": str(result.source),
        "output": str(result.output) if result.output else None,
        "dry_run": result.dry_run,
        "applied": result.applied,
        "iterations": result.iterations,
        "stop_reason": result.stop_reason,
        "before": before,
        "after": after,
        "metrics": metrics_report(result.metrics),
        "plans": planning["plans"],
        "changes": changes,
        "rejected_candidates": [
            change for change in changes if change["application_status"] == "rejected"
        ],
        "manual_review": result.manual_review,
        "execution": result.execution,
        "privacy": {
            "source_overwritten": False,
            "bibliography_protected": True,
            "full_document_sent_to_rewrite_provider": False,
            "textual_overlap_is_not_plagiarism_probability": True,
        },
    }


def reduction_result_markdown(result: ReductionResult) -> str:
    """Render a concise factual Markdown summary."""
    metrics = result.metrics
    relative = (
        f"{metrics.relative_reduction_pct:.1f}%"
        if metrics.relative_reduction_pct is not None
        else "n/a"
    )
    return "\n".join(
        [
            "# citeguard Attribution Improvement",
            "",
            f"- Source: `{result.source}`",
            f"- Output: `{result.output}`" if result.output else "- Output: none (preview)",
            f"- Applied: {result.applied}",
            f"- Stop reason: `{result.stop_reason}`",
            "",
            "## Before / after",
            "",
            f"- Textual similarity: {metrics.before_textual_similarity_pct:.1f}% → "
            f"{metrics.after_textual_similarity_pct:.1f}%",
            f"- Absolute reduction: {metrics.absolute_reduction_points:.1f} percentage points",
            f"- Relative reduction from baseline: {relative}",
            f"- High attribution risk: {metrics.before_high_risk} → {metrics.after_high_risk}",
            f"- Exact overlap: {metrics.before_exact_overlap_pct:.1f}% → "
            f"{metrics.after_exact_overlap_pct:.1f}%",
            f"- Lexical overlap: {metrics.before_lexical_overlap_pct:.1f}% → "
            f"{metrics.after_lexical_overlap_pct:.1f}%",
            f"- Semantic matches: {metrics.before_semantic_matches} → "
            f"{metrics.after_semantic_matches}",
            f"- Validated rewrites: {metrics.rewritten_passages}",
            f"- Manual review passages: {metrics.manual_review_passages}",
            f"- Rejected candidates: {metrics.rejected_candidates}",
            "",
            "## Integrity",
            "",
            f"- Citation integrity: {metrics.citation_integrity_passed}",
            f"- Numeric integrity: {metrics.numeric_integrity_passed}",
            f"- Named-entity integrity: {metrics.named_entity_integrity_passed}",
            f"- Unsupported new claims: {metrics.new_unsupported_claims}",
            "",
            "Textual overlap reduction is not a plagiarism verdict.",
        ]
    )
