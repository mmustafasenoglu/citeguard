"""Deterministic and LLM-assisted claim-to-source matching.

The pipeline separates metadata matching, evidence retrieval, and
entailment evaluation into distinct stages:

    metadata_score  →  evidence retrieval  →  entailment  →  aggregate

LLM calls happen only inside ``_evaluate_evidence`` via the entailment
classifier.  The legacy ``match_source_with_llm`` endpoint is no longer
called from the matcher; it is retained in ``llm.py`` for backward
compatibility but should not be used in the main pipeline.
"""

from __future__ import annotations

import re

from .entailment import EntailmentResult
from .models import Claim, Evidence, SourceCandidate, Verdict

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def match_claim_to_source(
    claim: Claim,
    candidate: SourceCandidate,
) -> tuple[int, int, Verdict, str, list[Evidence]]:
    """Full evidence-aware matching pipeline.

    Returns ``(metadata_score, support_score, verdict, reasoning, evidence)``.

    ``support_score`` is a standalone signal (evidence + entailment) that
    does NOT include metadata.  ``overall_confidence`` in ``scoring.py``
    combines ``metadata_score`` and ``support_score`` exactly once.
    """
    metadata_score = _metadata_similarity(claim, candidate)

    from .evidence import extract_evidence

    evidence_list = extract_evidence(claim, candidate)

    entailment = _evaluate_evidence(claim, candidate, evidence_list)

    _propagate_entailment(evidence_list, entailment)

    verdict, reasoning = _aggregate_verdict(metadata_score, evidence_list, entailment)
    support_score = _compute_support(evidence_list, entailment)

    return metadata_score, support_score, verdict, reasoning, evidence_list


def _propagate_entailment(
    evidence_list: list[Evidence],
    entailment: EntailmentResult | None,
) -> None:
    """Write entailment scores onto individual evidence objects.

    This allows downstream code (CLI, reports, scoring) to detect whether
    entailment was actually evaluated by checking
    ``any(e.entailment_score is not None for e in evidence)``.
    """
    if entailment is None:
        return
    for ev in evidence_list:
        ev.entailment_score = entailment.confidence
        ev.verdict = entailment.verdict


def _evaluate_evidence(
    claim: Claim,
    candidate: SourceCandidate,
    evidence_list: list[Evidence],
) -> EntailmentResult | None:
    """Run entailment evaluation: try LLM first, fall back to offline."""
    from .entailment import contradiction_risk, evaluate_evidence_with_llm

    llm_result = evaluate_evidence_with_llm(claim, evidence_list, candidate)
    if llm_result is not None:
        return llm_result

    if evidence_list:
        return contradiction_risk(claim, evidence_list[0].text)
    return None


def _aggregate_verdict(
    metadata_score: int,
    evidence_list: list[Evidence],
    entailment: EntailmentResult | None,
) -> tuple[Verdict, str]:
    """Combine metadata, evidence, and entailment into a final verdict."""
    if entailment is not None and entailment.verdict != Verdict.INSUFFICIENT_INFORMATION:
        return entailment.verdict, entailment.reasoning

    if evidence_list:
        best = evidence_list[0]
        if best.lexical_score >= 70:
            return Verdict.PARTIALLY_SUPPORTED, (
                f"Source content has strong topical overlap with the claim "
                f"(evidence relevance: {best.lexical_score})."
            )
        if best.lexical_score >= 50:
            return Verdict.PARTIALLY_SUPPORTED, (
                f"Source has moderate topical overlap with the claim "
                f"(evidence relevance: {best.lexical_score})."
            )

    if metadata_score >= 50 and (not evidence_list or evidence_list[0].lexical_score < 20):
        return Verdict.UNRELATED, (
            f"Source metadata matches but content does not overlap with the claim "
            f"(metadata score: {metadata_score})."
        )

    return Verdict.INSUFFICIENT_INFORMATION, (
        f"Insufficient overlap between source content and claim "
        f"(metadata score: {metadata_score})."
    )


def _compute_support(
    evidence_list: list[Evidence],
    entailment: EntailmentResult | None,
) -> int:
    """Compute a standalone support score from evidence + entailment.

    This score does NOT include metadata; the caller (``overall_confidence``
    in ``scoring.py``) combines metadata and support exactly once.
    """
    evidence_score = evidence_list[0].lexical_score if evidence_list else 0
    entailment_score = entailment.confidence if entailment else 0
    has_entailment = (
        entailment is not None
        and entailment.verdict != Verdict.INSUFFICIENT_INFORMATION
    )

    if has_entailment:
        return round(evidence_score * 0.25 + entailment_score * 0.75)
    return evidence_score


def _metadata_similarity(claim: Claim, candidate: SourceCandidate) -> int:
    query_tokens = set(_TOKEN_RE.findall(claim.search_query.lower()))
    title_tokens = set(_TOKEN_RE.findall(candidate.title.lower()))
    if not query_tokens or not title_tokens:
        return 0
    overlap = len(query_tokens & title_tokens)
    coverage = overlap / max(len(query_tokens), 1)
    title_coverage = overlap / max(len(title_tokens), 1)
    return round((coverage * 0.6 + title_coverage * 0.4) * 100)


def _clamp_score(value: object) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0


def _parse_verdict(value: object) -> Verdict:
    try:
        return Verdict(str(value))
    except ValueError:
        return Verdict.INSUFFICIENT_INFORMATION
