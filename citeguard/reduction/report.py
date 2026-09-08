"""JSON-safe reduction report serialization."""

from __future__ import annotations

from typing import Any

from .metrics import ReductionMetrics
from .models import FixPlan, PassageRisk, RewriteCandidate

REDUCTION_SCHEMA_VERSION = "1"


def reduction_report(
    risks: list[PassageRisk],
    plans: list[FixPlan],
    *,
    candidates: dict[str, list[RewriteCandidate]] | None = None,
) -> dict[str, Any]:
    """Serialize reduction planning and candidate evidence."""
    candidate_map = candidates or {}
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
                    "accepted": not candidate.rejection_reasons,
                    "rejection_reasons": candidate.rejection_reasons,
                    "meaning_score": candidate.meaning_score,
                    "source_support_score": candidate.source_support_score,
                    "citations_preserved": candidate.citations_preserved,
                    "numeric_integrity": candidate.numeric_integrity,
                    "factual_integrity": candidate.factual_integrity,
                    "semantic_similarity_to_original": (
                        candidate.semantic_similarity_to_original
                    ),
                    "forward_entailment_score": candidate.forward_entailment_score,
                    "backward_entailment_score": candidate.backward_entailment_score,
                    "meaning_verdict": (
                        candidate.meaning_verdict.value
                        if candidate.meaning_verdict
                        else None
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
        "absolute_reduction_pct": metrics.absolute_reduction_pct,
        "relative_reduction_pct": metrics.relative_reduction_pct,
        "before_high_risk": metrics.before_high_risk,
        "after_high_risk": metrics.after_high_risk,
        "rewritten_passages": metrics.rewritten_passages,
        "rejected_candidates": metrics.rejected_candidates,
        "meaning_preservation_avg": metrics.meaning_preservation_avg,
        "citation_integrity_passed": metrics.citation_integrity_passed,
        "new_unsupported_claims": metrics.new_unsupported_claims,
    }
