"""Deterministic claim extraction and citation-to-claim linking."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .models import Claim, ClaimType, ExistingCitation, Severity

_STATISTIC_SIGNALS = re.compile(
    r"\b\d[\d,.]*\s*(?:%|percent|million|billion|trillion|thousand|hundred)"
    r"|\b(?:approximately|roughly|about|nearly|over|around|up to)\s+\d"
    r"|\b(?:increase|decrease|growth|rate|ratio|proportion|average|median|mean)"
    r"\b.*\d",
    re.IGNORECASE,
)
_CAUSAL_SIGNALS = re.compile(
    r"\b(?:causes?|leads? to|results? in|due to|because of|contributes? to"
    r"|is associated with|is linked to|correlates? with|affects?|influences?)\b",
    re.IGNORECASE,
)
_COMPARATIVE_SIGNALS = re.compile(
    r"\b(?:more than|less than|fewer than|greater than|higher than|lower than"
    r"|compared to|in comparison|than the|outperform|underperform|exceeds?"
    r"|surpasses?|exceeds?)\b",
    re.IGNORECASE,
)
_HISTORICAL_SIGNALS = re.compile(
    r"\b(?:first|introduced|discovered|established|founded|proposed|published"
    r"|invented|developed|created|originated|began|emerged)\b",
    re.IGNORECASE,
)
_QUOTATION_SIGNALS = re.compile(r"\u201c[^\u201d]+\u201d|\"[^\"]+\"|'[^']+'")
_DEFINITION_SIGNALS = re.compile(
    r"\b(?:is defined as|refers to|is known as|is called|can be defined"
    r"|means that|is understood as|is characterized by)\b",
    re.IGNORECASE,
)
_PRIOR_WORK_SIGNALS = re.compile(
    r"\b(?:previous|prior|earlier|following|subsequent|existing|established)\s+"
    r"(?:work|study|research|studies|findings|evidence|literature|approach"
    r"|method|technique|framework|model)",
    re.IGNORECASE,
)

_INSTRUCTION_PATTERNS = re.compile(
    r"^(?:#+\s*)?(?:note|todo|fixme|hack|xxx|important|attention|warning"
    r"|please|ensure|make sure|do not|never|always|must|should|shall)\b",
    re.IGNORECASE,
)
_DECISION_PATTERNS = re.compile(
    r"\b(?:we (?:decided|chose|selected|opted|will use|plan to)"
    r"|the (?:goal|purpose|aim) (?:of this|is to))\b",
    re.IGNORECASE,
)


def extract_claims(
    paragraphs: list[str],
    citations: Sequence[ExistingCitation],
    *,
    bibliography_start: int | None = None,
    max_claims: int | None = None,
) -> list[Claim]:
    """Extract citation-worthy claims from non-bibliography paragraphs.

    This is a deterministic offline baseline. Claim extraction quality improves
    significantly with an optional LLM integration, but the deterministic rules
    ensure the tool remains functional without API keys.
    """
    limit: int | float = max_claims if max_claims is not None else float("inf")
    claims: list[Claim] = []

    for index, paragraph in enumerate(paragraphs):
        if len(claims) >= limit:
            break
        if bibliography_start is not None and index >= bibliography_start:
            continue
        if _is_non_claim_paragraph(paragraph):
            continue

        paragraph_citations = [c for c in citations if c.paragraph_index == index]
        sentences = _split_sentences(paragraph)
        for sentence in sentences:
            if len(claims) >= limit:
                break
            claim = _extract_claim_from_sentence(
                sentence, index, paragraph_citations, paragraph
            )
            if claim is not None:
                claims.append(claim)

        _tag_paragraph_final_links(claims, paragraph_citations, paragraph)
        _tag_ambiguous_links(claims)

    return claims


def _is_non_claim_paragraph(paragraph: str) -> bool:
    stripped = paragraph.strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return True
    if _INSTRUCTION_PATTERNS.match(stripped):
        return True
    if _DECISION_PATTERNS.search(stripped):
        return True
    return len(stripped.split()) < 5


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ])")


def _split_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_BOUNDARY_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _extract_claim_from_sentence(
    sentence: str,
    paragraph_index: int,
    paragraph_citations: list[ExistingCitation],
    full_paragraph: str,
) -> Claim | None:
    if len(sentence.split()) < 4:
        return None

    claim_type = _classify_claim_type(sentence)
    severity = _classify_severity(sentence, claim_type)
    text = _cap_claim_text(sentence)
    search_query = _build_search_query(sentence)
    linked = _find_citations_in_sentence(sentence, paragraph_citations)

    return Claim(
        text=text,
        search_query=search_query,
        claim_type=claim_type,
        severity=severity,
        paragraph_index=paragraph_index,
        has_existing_citation=len(linked) > 0,
        linked_citations=linked,
        linked_citation=linked[0] if linked else None,
        link_confidence=_link_confidence(linked, sentence),
    )


def _classify_claim_type(sentence: str) -> ClaimType:
    if _QUOTATION_SIGNALS.search(sentence):
        return ClaimType.QUOTATION
    if _COMPARATIVE_SIGNALS.search(sentence):
        return ClaimType.COMPARATIVE
    if _CAUSAL_SIGNALS.search(sentence):
        return ClaimType.CAUSAL
    if _STATISTIC_SIGNALS.search(sentence):
        return ClaimType.STATISTIC
    if _PRIOR_WORK_SIGNALS.search(sentence):
        return ClaimType.PRIOR_WORK
    if _HISTORICAL_SIGNALS.search(sentence):
        return ClaimType.HISTORICAL
    if _DEFINITION_SIGNALS.search(sentence):
        return ClaimType.DEFINITION
    return ClaimType.GENERAL_FACT


def _classify_severity(sentence: str, claim_type: ClaimType) -> Severity:
    if claim_type in (ClaimType.STATISTIC, ClaimType.CAUSAL, ClaimType.QUOTATION):
        return Severity.HIGH
    if claim_type in (ClaimType.COMPARATIVE, ClaimType.HISTORICAL):
        return Severity.MEDIUM
    if claim_type == ClaimType.DEFINITION:
        return Severity.MEDIUM
    return Severity.LOW


def _cap_claim_text(text: str, max_words: int = 25) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def _build_search_query(sentence: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", sentence)
    words = cleaned.split()
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "under", "again",
        "that", "this", "these", "those", "and", "but", "or", "nor", "not",
        "so", "very", "just", "than", "too", "also", "about", "such",
        "it", "its", "they", "them", "their", "we", "our", "he", "she",
        "his", "her", "which", "who", "whom", "what", "where", "when",
    }
    meaningful = [w for w in words if w.lower() not in stop_words and len(w) > 2]
    return " ".join(meaningful[:12]) if meaningful else " ".join(words[:8])


def _sentence_has_citation(sentence: str, citations: list[ExistingCitation]) -> bool:
    return any(citation.raw_text in sentence for citation in citations)


def _find_citations_in_sentence(
    sentence: str, citations: list[ExistingCitation]
) -> list[ExistingCitation]:
    """Return all citations whose raw text appears in *sentence*.

    Implements SPEC §9 proximity rule 1: a citation inside the same sentence
    is linked to claims in that sentence.
    """
    return [c for c in citations if c.raw_text in sentence]


def _is_paragraph_final_citation(
    citation: ExistingCitation, paragraph: str
) -> bool:
    """Return True when *citation* sits at the end of *paragraph*.

    Implements SPEC §9 rule 3: paragraph-final citation detection.
    """
    if citation.char_offset < 0 or citation.char_offset >= len(paragraph):
        return False
    after = paragraph[citation.char_offset + len(citation.raw_text) :]
    remaining = after.strip()
    return not remaining or all(ch in ".,;:!?)\\]}" for ch in remaining)


def _tag_paragraph_final_links(
    claims: list[Claim],
    paragraph_citations: list[ExistingCitation],
    paragraph: str,
) -> None:
    """Tag claims whose paragraph-final citation confidence could be upgraded.

    SPEC §9 rule 3: a paragraph-final citation first targets claims in the
    final sentence.  When a citation sits at the end of the paragraph and
    already links to a claim in the last sentence, upgrade confidence to 100
    if it was lower.
    """
    sentences = _split_sentences(paragraph)
    if not sentences:
        return
    last_sentence = sentences[-1]
    para_final = [
        c for c in paragraph_citations if _is_paragraph_final_citation(c, paragraph)
    ]
    for claim in claims:
        if claim.paragraph_index != claims[0].paragraph_index:
            continue
        if claim.text not in last_sentence:
            continue
        for cit in para_final:
            if (
                cit in claim.linked_citations
                and claim.link_confidence is not None
                and claim.link_confidence < 100
            ):
                claim.link_confidence = 100


def _tag_ambiguous_links(claims: list[Claim]) -> None:
    """Produce ``ambiguous_citation_link`` warnings where SPEC §9 rule 5 applies.

    A citation that could reasonably belong to more than one claim in the
    same paragraph is flagged with a warning on every claim it touches.
    """
    cit_to_claims: dict[int, list[int]] = {}
    for idx, claim in enumerate(claims):
        for cit in claim.linked_citations:
            cit_to_claims.setdefault(id(cit), []).append(idx)

    for cit_id, claim_idxs in cit_to_claims.items():
        if len(claim_idxs) <= 1:
            continue
        raw_text = ""
        for claim in claims:
            for cit in claim.linked_citations:
                if id(cit) == cit_id:
                    raw_text = cit.raw_text
                    break
            if raw_text:
                break
        warning = f"ambiguous_citation_link: {raw_text} targets {len(claim_idxs)} claims"
        for ci in claim_idxs:
            if warning not in claims[ci].warnings:
                claims[ci].warnings.append(warning)


def _link_confidence(
    linked: list[ExistingCitation], sentence: str
) -> int | None:
    """Deterministic confidence for the citation-to-claim link.

    Returns 100 when a citation is found at the end of the sentence
    (sentence-final position), 80 when found mid-sentence, and None
    when no citation is linked.
    """
    if not linked:
        return None
    stripped = sentence.rstrip()
    for citation in linked:
        offset = stripped.rfind(citation.raw_text)
        if offset >= 0:
            after = stripped[offset + len(citation.raw_text) :].strip(" .,;:)]}")
            if not after:
                return 100
    return 80
