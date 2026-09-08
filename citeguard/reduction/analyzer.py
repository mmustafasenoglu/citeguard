"""Classification of similarity evidence into reduction-specific passage risks."""

from __future__ import annotations

import re

from ..models import Claim, Verdict
from ..similarity.models import (
    RiskLevel,
    SimilarityEngineResult,
    SimilarityResult,
)
from .models import FixAction, PassageRisk, ReductionRiskType

_NUMBER_RE = re.compile(r"(?<!\w)(?:\d+(?:[.,]\d+)?%?|n\s*=\s*\d[\d,]*)", re.I)


def _citation_context(
    item: SimilarityResult,
    claims: list[Claim] | None,
) -> tuple[bool, bool | None, Verdict | None]:
    citations = item.sentence.citations
    has_citation = bool(citations)
    if not claims:
        return has_citation, None, None

    matching = [
        claim for claim in claims
        if claim.paragraph_index == item.sentence.paragraph_index
        and claim.has_existing_citation
    ]
    if not matching:
        return has_citation, None, None

    support = None
    for claim in matching:
        if claim.linked_citation is not None:
            support = Verdict.INSUFFICIENT_INFORMATION
            break
    return has_citation, None, support


def _classify(
    item: SimilarityResult,
    *,
    has_citation: bool,
    citation_verified: bool | None,
) -> tuple[ReductionRiskType, FixAction, float]:
    match = item.best_match
    if match is None:
        return ReductionRiskType.UNKNOWN, FixAction.LEAVE, 0.35

    quoted = any(
        char in item.sentence.text
        for char in ('"', "\u201c", "\u201d", "\u00ab", "\u00bb")
    )
    if quoted and match.exact_overlap >= 0.70 and has_citation:
        return ReductionRiskType.QUOTE_NEEDED, FixAction.ADD_QUOTATION, 0.98
    if match.exact_overlap >= 0.70:
        if has_citation:
            return ReductionRiskType.TOO_CLOSE_PARAPHRASE, FixAction.PARAPHRASE, 0.92
        return ReductionRiskType.EXACT_COPY, FixAction.ADD_CITATION, 0.97
    if match.lexical_similarity >= 0.70 and (
        match.semantic_similarity_raw >= 0.85 or match.semantic_similarity_raw == 0.0
    ):
        if has_citation:
            return ReductionRiskType.TOO_CLOSE_PARAPHRASE, FixAction.PARAPHRASE, 0.84
        return ReductionRiskType.MISSING_CITATION, FixAction.ADD_CITATION, 0.88
    if item.attribution_risk == RiskLevel.HIGH and not has_citation:
        return ReductionRiskType.MISSING_CITATION, FixAction.ADD_CITATION, 0.82
    if item.attribution_risk == RiskLevel.MEDIUM:
        return ReductionRiskType.ACCEPTABLE_OVERLAP, FixAction.MANUAL_REVIEW, 0.65
    if citation_verified is True:
        return ReductionRiskType.ACCEPTABLE_OVERLAP, FixAction.LEAVE, 0.80
    return ReductionRiskType.UNKNOWN, FixAction.LEAVE, 0.40


def analyze_passage_risks(
    result: SimilarityEngineResult,
    *,
    claims: list[Claim] | None = None,
) -> list[PassageRisk]:
    """Convert similarity results into conservative, reviewable risk records."""
    risks: list[PassageRisk] = []
    for _index, item in enumerate(result.results):
        if item.sentence.is_bibliography:
            continue
        has_citation, citation_verified, citation_support = _citation_context(item, claims)
        risk_type, action, confidence = _classify(
            item,
            has_citation=has_citation,
            citation_verified=citation_verified,
        )
        match = item.best_match
        risks.append(
            PassageRisk(
                passage_id=f"p{item.sentence.paragraph_index}s{item.sentence.sentence_index}",
                text=item.sentence.text,
                paragraph_index=item.sentence.paragraph_index,
                start_offset=item.sentence.start_offset,
                end_offset=item.sentence.end_offset,
                exact_overlap=match.exact_overlap if match else 0.0,
                lexical_similarity=match.lexical_similarity if match else 0.0,
                semantic_similarity_raw=(
                    match.semantic_similarity_raw if match else None
                ),
                attribution_risk=item.attribution_risk,
                has_citation=has_citation,
                citation_verified=citation_verified,
                citation_support=citation_support,
                citation_texts=tuple(citation.raw_text for citation in item.sentence.citations),
                risk_type=risk_type,
                recommended_action=action,
                confidence=confidence,
                source_title=match.source_title if match else None,
                source_id=match.source_id if match else None,
                attribution_reason=item.attribution_reason,
            )
        )
    return risks


def extract_numbers(text: str) -> tuple[str, ...]:
    """Return normalized numeric tokens that a rewrite must preserve."""
    return tuple(_NUMBER_RE.findall(text))
