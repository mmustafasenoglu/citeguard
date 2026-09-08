"""Conservative action planning for passage-level attribution issues."""

from __future__ import annotations

from .analyzer import extract_numbers
from .models import FixAction, FixPlan, PassageRisk, ReductionRiskType


def build_fix_plans(risks: list[PassageRisk]) -> list[FixPlan]:
    """Build explicit plans; only close paraphrases are rewrite-enabled."""
    plans: list[FixPlan] = []
    for risk in risks:
        citations = risk.citation_texts
        rewrite_allowed = risk.recommended_action == FixAction.PARAPHRASE
        action = risk.recommended_action
        if risk.risk_type in {
            ReductionRiskType.EXACT_COPY,
            ReductionRiskType.MISSING_CITATION,
        }:
            action = FixAction.ADD_CITATION
        plans.append(
            FixPlan(
                passage_id=risk.passage_id,
                action=action,
                reason=risk.attribution_reason or risk.risk_type.value,
                preserve_citations=citations,
                must_preserve_numbers=extract_numbers(risk.text),
                rewrite_allowed=rewrite_allowed,
            )
        )
    return plans
