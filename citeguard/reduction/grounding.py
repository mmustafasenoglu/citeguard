"""Deterministic bridge from passage risks to verified rewrite grounding."""

from __future__ import annotations

from collections.abc import Iterable

from ..audit import AuditResult
from ..rewrite.models import RewriteMode
from ..rewrite.models import RewriteRequest as GroundedRewriteRequest
from ..rewrite.service import build_context
from ..similarity.lexical import normalize_turkish
from .models import FixPlan, PassageRisk


def _contained(left: str, right: str) -> bool:
    left_norm = normalize_turkish(left).strip(" .")
    right_norm = normalize_turkish(right).strip(" .")
    return bool(left_norm and right_norm) and (left_norm in right_norm or right_norm in left_norm)


def build_grounded_reduction_requests(
    audit: AuditResult,
    risks: Iterable[PassageRisk],
    plans: Iterable[FixPlan],
    *,
    mode: RewriteMode = RewriteMode.CLARIFY,
    max_context_chars: int = 2000,
) -> tuple[dict[str, GroundedRewriteRequest], dict[str, str]]:
    """Map eligible passages to one unambiguous verified claim/source context.

    Mapping first requires paragraph identity, then contained normalized claim
    text and citation identity. Ambiguous or unsupported mappings fail closed.
    """
    risk_map = {risk.passage_id: risk for risk in risks}
    grounded: dict[str, GroundedRewriteRequest] = {}
    manual: dict[str, str] = {}
    for plan in plans:
        if not plan.rewrite_allowed:
            continue
        risk = risk_map.get(plan.passage_id)
        if risk is None or not risk.citation_texts:
            manual[plan.passage_id] = "verified citation grounding is unavailable"
            continue
        matches: list[GroundedRewriteRequest] = []
        for assessment in audit.claim_assessments:
            claim = assessment.claim
            if claim.paragraph_index != risk.paragraph_index:
                continue
            if not _contained(claim.text, risk.text):
                continue
            for source in assessment.sources:
                if not source.evaluated:
                    continue
                if source.citation_raw not in risk.citation_texts:
                    continue
                context = build_context(
                    original_text=risk.text,
                    claim_text=claim.text,
                    claim_type=claim.claim_type.value,
                    citation_raw=source.citation_raw,
                    verification_status=(
                        source.bib_status.value if source.bib_status else "unknown"
                    ),
                    verdict=source.verdict,
                    candidate=source.candidate,
                    evidence=source.evidence,
                )
                if context is not None:
                    matches.append(
                        GroundedRewriteRequest(
                            context=context,
                            mode=mode,
                            max_context_chars=max_context_chars,
                        )
                    )
        if len(matches) == 1:
            grounded[plan.passage_id] = matches[0]
        elif not matches:
            manual[plan.passage_id] = "no verified supporting evidence maps to passage"
        else:
            manual[plan.passage_id] = "ambiguous verified claim mapping"
    return grounded, manual
