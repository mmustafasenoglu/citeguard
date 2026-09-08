"""Hard integrity gates for rewrite candidates."""

from __future__ import annotations

from ..models import Verdict
from .analyzer import extract_numbers
from .models import FixPlan, MeaningValidation, MeaningVerdict, RewriteCandidate, ValidationResult


def validate_candidate(
    original_text: str,
    candidate: RewriteCandidate,
    plan: FixPlan,
    *,
    supported_verdict: Verdict | None = None,
    unsupported_claims: list[str] | None = None,
    meaning_validation: MeaningValidation | None = None,
) -> ValidationResult:
    """Apply non-negotiable citation, numeric, and support gates."""
    reasons: list[str] = []
    citations_preserved = all(
        citation in candidate.text for citation in plan.preserve_citations
    )
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
    if (
        meaning_validation is not None
        and meaning_validation.verdict != MeaningVerdict.PRESERVED
    ):
        reasons.append("meaning preservation was not established")

    accepted = not reasons
    candidate.citations_preserved = citations_preserved
    candidate.numeric_integrity = numeric_integrity
    candidate.factual_integrity = factual_integrity
    candidate.verdict = supported_verdict
    if meaning_validation is not None:
        candidate.meaning_verdict = meaning_validation.verdict
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
