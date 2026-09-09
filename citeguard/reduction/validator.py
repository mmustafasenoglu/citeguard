"""Hard integrity gates for rewrite candidates."""
# ruff: noqa: E501

from __future__ import annotations

from ..models import Verdict
from ..similarity.lexical import normalize_turkish
from .analyzer import extract_protected_tokens
from .models import FixPlan, MeaningValidation, MeaningVerdict, RewriteCandidate, ValidationResult


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _strengthens_claim(original: str, candidate: str) -> bool:
    """Detect explicit English or Turkish epistemic-strength escalation."""
    original_norm = normalize_turkish(original)
    candidate_norm = normalize_turkish(candidate)
    # A negated strong phrase does not assert the stronger claim.
    negation_markers = ("değil", "değildir", "değildi", "gösterilmemiş", "kanıtlanmamış", "kurulamaz", "görülmedi")
    if any(marker in candidate_norm for marker in negation_markers):
        return False
    transitions = (
        (("associated with",), ("caused", "causes")),
        (("may ",), ("definitely", "certainly")),
        (("could ",), ("will ", "eliminates")),
        (("some ",), ("all ", "every ")),
        (
            ("ilişkili", "bağlantılı", "korelasyon", "birlikte değişim"),
            ("neden oldu", "neden olur", "sebep oldu", "sebep olur", "yol açtı"),
        ),
        (
            (
                "olabilir",
                "olabileceği",
                "artırabilir",
                "azaltabilir",
                "uygulanabilir",
                "muhtemeldir",
                "gösterebilir",
                "işaret etmektedir",
            ),
            ("kesindir", "kesin olarak", "mutlaka", "kanıtlamaktadır"),
        ),
        (
            ("bazı", "bir kısmı", "belirli katılımcılar"),
            ("tümü", "tüm ", "herkes", "bütün katılımcılar"),
        ),
        (
            ("ön bulgular", "sınırlı kanıt", "gözlemsel sonuç"),
            ("kesin kanıt", "kanıtlanmıştır", "nedensellik gösterilmiştir"),
        ),
    )
    return any(
        _contains_any(original_norm, weak) and _contains_any(candidate_norm, strong)
        for weak, strong in transitions
    )


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
    citations_preserved = all(citation in candidate.text for citation in plan.preserve_citations)
    if plan.preserve_citations and not citations_preserved:
        reasons.append("required citation tokens were removed")

    original_numbers = extract_protected_tokens(original_text)
    candidate_numbers = extract_protected_tokens(candidate.text)
    numeric_integrity = set(original_numbers) == set(candidate_numbers)
    if not numeric_integrity:
        reasons.append("numeric values were changed or removed")

    unsupported = tuple(unsupported_claims or ())
    factual_integrity = supported_verdict not in {Verdict.CONTRADICTED}
    original_lower = original_text.casefold()
    candidate_lower = candidate.text.casefold()
    if _strengthens_claim(original_lower, candidate_lower):
        factual_integrity = False
        reasons.append("candidate strengthens the claim beyond the original")
    if not factual_integrity:
        reasons.append("candidate contradicts source support")
    if unsupported:
        reasons.append("candidate introduces unsupported claims")
    if meaning_validation is not None and meaning_validation.verdict != MeaningVerdict.PRESERVED:
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
