"""Hard integrity gates for rewrite candidates."""

from __future__ import annotations

import re

from ..models import Verdict
from .analyzer import extract_numbers
from .models import FixPlan, RewriteCandidate, ValidationResult

_TOKEN_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def _citation_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if token.isdigit() or token[:1].isupper()
    }


def validate_candidate(
    original_text: str,
    candidate: RewriteCandidate,
    plan: FixPlan,
    *,
    supported_verdict: Verdict | None = None,
    unsupported_claims: list[str] | None = None,
) -> ValidationResult:
    """Apply non-negotiable citation, numeric, and support gates."""
    reasons: list[str] = []
    original_citations = _citation_tokens(original_text)
    candidate_citations = _citation_tokens(candidate.text)
    citations_preserved = not original_citations.difference(candidate_citations)
    if plan.preserve_citations and not citations_preserved:
        reasons.append("required citation tokens were removed")

    original_numbers = extract_numbers(original_text)
    candidate_numbers = extract_numbers(candidate.text)
    numeric_integrity = all(number in candidate_numbers for number in original_numbers)
    if not numeric_integrity:
        reasons.append("numeric values were changed or removed")

    unsupported = tuple(unsupported_claims or ())
    factual_integrity = supported_verdict not in {Verdict.CONTRADICTED}
    if not factual_integrity:
        reasons.append("candidate contradicts source support")
    if unsupported:
        reasons.append("candidate introduces unsupported claims")

    accepted = not reasons
    candidate.citations_preserved = citations_preserved
    candidate.numeric_integrity = numeric_integrity
    candidate.factual_integrity = factual_integrity
    candidate.verdict = supported_verdict
    candidate.introduced_claims = list(unsupported)
    candidate.rejection_reasons.extend(reasons)
    return ValidationResult(
        accepted=accepted,
        citations_preserved=citations_preserved,
        numeric_integrity=numeric_integrity,
        factual_integrity=factual_integrity,
        unsupported_new_claims=unsupported,
        reasons=tuple(reasons),
    )
