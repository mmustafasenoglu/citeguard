"""Deterministic and LLM-assisted claim-to-source matching."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .models import Claim, SourceCandidate, Verdict

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def match_claim_to_source(
    claim: Claim,
    candidate: SourceCandidate,
) -> tuple[int, int, Verdict, str]:
    """Deterministic lexical baseline for claim-source matching.

    Returns (metadata_match_score, claim_support_score, verdict, reasoning).
    This baseline does not understand semantic contradiction; it can only detect
    topic overlap. A more capable matcher may be layered on top via the optional
    LLM integration, but this function always provides a safe offline fallback.
    """
    llm_result = _try_llm_match(claim, candidate)
    if llm_result is not None:
        return llm_result

    metadata_score = _metadata_similarity(claim, candidate)
    support_score = _title_abstract_overlap(claim, candidate)
    verdict = _determine_verdict(support_score, metadata_score)
    reasoning = _build_reasoning(verdict, support_score, metadata_score)
    return metadata_score, support_score, verdict, reasoning


def _try_llm_match(
    claim: Claim, candidate: SourceCandidate
) -> tuple[int, int, Verdict, str] | None:
    """Attempt LLM-based matching; returns None when unavailable or on failure."""
    from .llm import match_source_with_llm

    authors_str = ", ".join(candidate.authors[:5])
    if len(candidate.authors) > 5:
        authors_str += " et al."
    result = match_source_with_llm(
        claim,
        source_title=candidate.title,
        source_authors=authors_str,
        source_year=candidate.year,
        source_abstract=candidate.abstract,
    )
    if not result or not isinstance(result, dict):
        return None

    metadata_score = _clamp_score(result.get("metadata_match_score", 0))
    support_score = _clamp_score(result.get("claim_support_score", 0))
    verdict = _parse_verdict(result.get("verdict", "insufficient_information"))
    reasoning = str(result.get("reasoning", ""))[:200]
    if not reasoning:
        reasoning = _build_reasoning(verdict, support_score, metadata_score)
    return metadata_score, support_score, verdict, reasoning


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


def _metadata_similarity(claim: Claim, candidate: SourceCandidate) -> int:
    query_tokens = set(_TOKEN_RE.findall(claim.search_query.lower()))
    title_tokens = set(_TOKEN_RE.findall(candidate.title.lower()))
    if not query_tokens or not title_tokens:
        return 0
    overlap = len(query_tokens & title_tokens)
    coverage = overlap / max(len(query_tokens), 1)
    title_coverage = overlap / max(len(title_tokens), 1)
    return round((coverage * 0.6 + title_coverage * 0.4) * 100)


def _title_abstract_overlap(claim: Claim, candidate: SourceCandidate) -> int:
    claim_tokens = set(_TOKEN_RE.findall(claim.text.lower()))
    source_text = candidate.title or ""
    if candidate.abstract:
        source_text += " " + candidate.abstract
    source_tokens = set(_TOKEN_RE.findall(source_text.lower()))
    if not claim_tokens or not source_tokens:
        return 0
    overlap = len(claim_tokens & source_tokens)
    claim_coverage = overlap / max(len(claim_tokens), 1)
    source_coverage = overlap / max(len(source_tokens), 1)
    sequence_score = SequenceMatcher(
        None,
        " ".join(sorted(claim_tokens)),
        " ".join(sorted(source_tokens)),
    ).ratio()
    combined = claim_coverage * 0.50 + source_coverage * 0.20 + sequence_score * 0.30
    return round(combined * 100)


def _determine_verdict(support_score: int, metadata_score: int) -> Verdict:
    if support_score >= 70:
        return Verdict.SUPPORTED
    if support_score >= 50:
        return Verdict.PARTIALLY_SUPPORTED
    if metadata_score >= 50 and support_score < 20:
        return Verdict.UNRELATED
    return Verdict.INSUFFICIENT_INFORMATION


def _build_reasoning(verdict: Verdict, support_score: int, metadata_score: int) -> str:
    if verdict == Verdict.SUPPORTED:
        return (
            f"Source title and abstract share significant overlap with the claim "
            f"(support score: {support_score})."
        )
    if verdict == Verdict.PARTIALLY_SUPPORTED:
        return (
            f"Source has moderate topical overlap with the claim "
            f"(support score: {support_score})."
        )
    if verdict == Verdict.UNRELATED:
        return (
            f"Source metadata matches but content does not overlap with the claim "
            f"(support score: {support_score})."
        )
    return (
        f"Insufficient overlap between source content and claim to determine support "
        f"(support score: {support_score}, metadata score: {metadata_score})."
    )
