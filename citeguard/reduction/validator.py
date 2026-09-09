"""Hard integrity gates for rewrite candidates."""

from __future__ import annotations

import re

from ..models import Verdict
from ..similarity.lexical import normalize_turkish
from .analyzer import extract_protected_tokens
from .models import FixPlan, MeaningValidation, MeaningVerdict, RewriteCandidate, ValidationResult


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


_CLAUSE_SPLIT_RE = re.compile(r"[;.!?]+")
_NEGATION_MARKERS = (
    "not ",
    "no ",
    "never ",
    "does not ",
    "did not ",
    "cannot ",
    "değil",
    "gösterilmemiş",
    "kanıtlanmamış",
    "kurulamaz",
    "görülmedi",
    "söylenemez",
    "doğrulanmamış",
)
_ENTITY_TOKEN_RE = re.compile(r"\b[^\W_]+\b", re.UNICODE)


def _named_entities(text: str) -> tuple[str, ...]:
    """Extract conservative proper names with a linear token scan."""
    tokens = list(_ENTITY_TOKEN_RE.finditer(text))
    entities: list[str] = []
    title_run: list[re.Match[str]] = []

    def flush_title_run() -> None:
        if len(title_run) >= 2:
            entities.append(text[title_run[0].start() : title_run[-1].end()])
        title_run.clear()

    for match in tokens:
        token = match.group(0)
        is_acronym = len(token) >= 2 and token.isupper()
        is_camel = any(char.isupper() for char in token[1:])
        is_title = token[:1].isupper() and token[1:].islower()
        if is_acronym or is_camel:
            flush_title_run()
            entities.append(token)
            continue
        if is_title:
            if title_run and text[title_run[-1].end() : match.start()].strip():
                flush_title_run()
            title_run.append(match)
            continue
        flush_title_run()
    flush_title_run()
    return tuple(dict.fromkeys(entities))


def _contains_asserted_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    """Return whether a strong phrase occurs outside a locally negated clause."""
    for clause in _CLAUSE_SPLIT_RE.split(text):
        if _contains_any(clause, phrases) and not _contains_any(clause, _NEGATION_MARKERS):
            return True
    return False


def _strengthens_claim(original: str, candidate: str) -> bool:
    """Detect explicit English or Turkish epistemic-strength escalation.

    These are deliberately relation-level transitions, not a vocabulary
    blacklist: a strong marker is relevant only when the source includes the
    corresponding weaker relation.  ``_contains_asserted_phrase`` keeps a
    negation in a different clause from suppressing a real assertion.
    """
    original_norm = normalize_turkish(original)
    candidate_norm = normalize_turkish(candidate)
    transitions = (
        (
            ("associated with", "correlated with", "linked to", "related to"),
            ("caused", "causes", "led to", "results in"),
        ),
        (("may ", "might ", "possible"), ("definitely", "certainly", "proves")),
        (("could ",), ("will ", "eliminates")),
        (("some ", "a subset", "limited to"), ("all ", "every ", "universally")),
        (
            ("preliminary evidence", "initial evidence", "limited evidence", "suggests"),
            (
                "proves",
                "conclusively established",
                "conclusively demonstrated",
                "definitive evidence",
            ),
        ),
        (
            (
                "ilişkili",
                "ilişki bulundu",
                "ilişki gözlendi",
                "bağlantılı",
                "korelasyon",
                "birlikte değişim",
            ),
            (
                "neden oldu",
                "neden olur",
                "neden olduğu",
                "sebep oldu",
                "sebep olur",
                "yol açtı",
                "yol açar",
            ),
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
                "düşündürmektedir",
            ),
            (
                "kesindir",
                "kesin olarak",
                "mutlaka",
                "kanıtlamaktadır",
                "kesin olarak gösterilmiştir",
                "kesin olarak etkilidir",
            ),
        ),
        (
            ("bazı", "bir kısmı", "belirli katılımcılar"),
            ("tümü", "tüm ", "herkes", "bütün katılımcılar"),
        ),
        (
            (
                "ön bulgular",
                "ilk bulgular",
                "sınırlı kanıt",
                "işaret etmektedir",
                "düşündürmektedir",
                "gözlemsel sonuç",
            ),
            (
                "kesin kanıt",
                "kanıtlanmıştır",
                "kanıtlamaktadır",
                "kesin olarak gösterilmiştir",
                "kesin olarak etkilidir",
                "nedensellik gösterilmiştir",
            ),
        ),
    )
    return any(
        _contains_any(original_norm, weak) and _contains_asserted_phrase(candidate_norm, strong)
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

    original_entities = _named_entities(original_text)
    candidate_folded = candidate.text.casefold()
    named_entity_integrity = all(
        entity.casefold() in candidate_folded for entity in original_entities
    )
    if not named_entity_integrity:
        reasons.append("important named entities were changed or removed")

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
    candidate.named_entity_integrity = named_entity_integrity
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
        named_entity_integrity=named_entity_integrity,
        unsupported_new_claims=unsupported,
        reasons=tuple(reasons),
    )
