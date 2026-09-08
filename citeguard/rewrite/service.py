"""Rewrite orchestration over verified audit results.

Builds evidence-grounded :class:`RewriteRequest` objects from typed
audit assessments.  Verification output is read, never reinterpreted:
a request is only built when a claim has an evaluated source with a
supporting verdict and non-empty evidence passages.  Otherwise the
claim maps to ``INSUFFICIENT_EVIDENCE`` with zero network calls.
"""

from __future__ import annotations

from ..models import Evidence, MatchResult, SourceCandidate, Verdict
from .models import (
    RewriteContext,
    RewriteMode,
    RewriteRequest,
    RewriteResult,
    RewriteStatus,
)

# Verdicts that license an evidence-grounded rewrite.  CONTRADICTED,
# UNRELATED, and INSUFFICIENT_INFORMATION never do: rewriting a claim
# the evidence rejects (or says nothing about) cannot be grounded.
_GROUNDING_VERDICTS = frozenset(
    {Verdict.SUPPORTED, Verdict.PARTIALLY_SUPPORTED}
)


def _evidence_texts(evidence: list[Evidence], *, limit: int = 3) -> tuple[str, ...]:
    texts: list[str] = []
    for item in evidence:
        text = (item.text or "").strip()
        if text and text not in texts:
            texts.append(text)
        if len(texts) >= limit:
            break
    return tuple(texts)


def build_context(
    *,
    original_text: str,
    claim_text: str,
    claim_type: str,
    citation_raw: str | None,
    verification_status: str,
    verdict: Verdict | None,
    candidate: SourceCandidate | None,
    evidence: list[Evidence],
) -> RewriteContext | None:
    """Build verified context, or None when grounding is insufficient.

    Returns None (→ INSUFFICIENT_EVIDENCE, zero network) unless the
    claim has an evaluated source with a supporting verdict and at
    least one non-empty evidence passage.
    """
    if candidate is None or verdict not in _GROUNDING_VERDICTS:
        return None
    evidence_texts = _evidence_texts(evidence)
    if not evidence_texts:
        return None
    return RewriteContext(
        original_text=original_text,
        claim_text=claim_text,
        claim_type=claim_type,
        citation_raw=citation_raw,
        verification_status=verification_status,
        verdict=verdict.value,
        source_title=candidate.title,
        source_authors=tuple(candidate.authors),
        source_year=candidate.year,
        source_doi=candidate.doi,
        evidence_texts=evidence_texts,
    )


def insufficient_result(
    *,
    original_text: str,
    mode: RewriteMode,
    reason: str,
) -> RewriteResult:
    """Deterministic INSUFFICIENT_EVIDENCE result (no provider involved)."""
    return RewriteResult(
        original_text=original_text,
        rewritten_text=None,
        status=RewriteStatus.INSUFFICIENT_EVIDENCE,
        provider="none",
        model="none",
        warnings=(reason,),
        evidence_used=(),
        citation_preserved=None,
        metadata={"mode": mode.value},
    )


def collect_rewrite_requests(
    audit_result: object,
    *,
    mode: RewriteMode,
    max_context_chars: int = 2000,
    style_constraints: tuple[str, ...] = (),
) -> tuple[list[RewriteRequest], list[RewriteResult]]:
    """Collect grounded requests from a typed :class:`AuditResult`.

    Returns ``(requests, insufficient)`` where ``insufficient`` holds
    INSUFFICIENT_EVIDENCE results for claims without grounding.  Makes
    zero network calls; providers are invoked separately per request.
    """
    from ..audit import AuditResult

    assert isinstance(audit_result, AuditResult)
    requests: list[RewriteRequest] = []
    insufficient: list[RewriteResult] = []

    def _emit(
        *,
        original_text: str,
        claim_text: str,
        claim_type: str,
        citation_raw: str | None,
        verification_status: str,
        verdict: Verdict | None,
        candidate: SourceCandidate | None,
        evidence: list[Evidence],
    ) -> None:
        context = build_context(
            original_text=original_text,
            claim_text=claim_text,
            claim_type=claim_type,
            citation_raw=citation_raw,
            verification_status=verification_status,
            verdict=verdict,
            candidate=candidate,
            evidence=evidence,
        )
        if context is None:
            insufficient.append(
                insufficient_result(
                    original_text=original_text,
                    mode=mode,
                    reason=(
                        "no evaluated source with a supporting verdict and "
                        "evidence passages; rewrite not attempted."
                    ),
                )
            )
            return
        requests.append(
            RewriteRequest(
                context=context,
                mode=mode,
                style_constraints=style_constraints,
                max_context_chars=max_context_chars,
            )
        )

    for assessment in audit_result.claim_assessments:
        claim = assessment.claim
        evaluated = [s for s in assessment.sources if s.evaluated]
        if not evaluated:
            _emit(
                original_text=claim.text,
                claim_text=claim.text,
                claim_type=claim.claim_type.value,
                citation_raw=(
                    claim.linked_citation.raw_text
                    if claim.linked_citation is not None
                    else None
                ),
                verification_status=assessment.summary,
                verdict=None,
                candidate=None,
                evidence=[],
            )
            continue
        best = max(evaluated, key=lambda s: s.confidence or 0)
        _emit(
            original_text=claim.text,
            claim_text=claim.text,
            claim_type=claim.claim_type.value,
            citation_raw=(
                claim.linked_citation.raw_text
                if claim.linked_citation is not None
                else None
            ),
            verification_status=assessment.summary,
            verdict=best.verdict,
            candidate=best.candidate,
            evidence=list(best.evidence or []),
        )

    for suggestion in audit_result.suggestions:
        claim = suggestion.claim
        matched: MatchResult | None = suggestion.matched
        _emit(
            original_text=claim.text,
            claim_text=claim.text,
            claim_type=claim.claim_type.value,
            citation_raw=None,
            verification_status=suggestion.status.value,
            verdict=matched.verdict if matched is not None else None,
            candidate=matched.candidate if matched is not None else None,
            evidence=list(matched.evidence) if matched is not None else [],
        )

    return requests, insufficient
