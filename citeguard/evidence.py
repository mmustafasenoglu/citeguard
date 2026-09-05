"""Abstract evidence extraction and ranking.

This module finds the most relevant passages from a source's abstract
that may support or contradict a claim.  The first version operates on
abstracts only; full-text evidence is deferred to a later release.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .models import Claim, Evidence, EvidenceType, SourceCandidate, Verdict

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def split_evidence_sentences(text: str) -> list[str]:
    """Split *text* into sentences suitable for evidence ranking."""
    boundaries = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ])")
    sentences = boundaries.split(text)
    return [s.strip() for s in sentences if s.strip() and len(s.split()) >= 2]


def evidence_relevance(claim: Claim, sentence: str) -> int:
    """Score how relevant *sentence* is to *claim* using lexical signals.

    Returns an integer 0-100.  The scoring is deliberately simple so that
    an embedder or NLI model can be layered on top later without changing
    the public API.

    Formula:
        claim_token_coverage  60 %
        sequence_similarity   20 %
        sentence_token_overlap 20 %
    """
    claim_tokens = set(_TOKEN_RE.findall(claim.text.lower()))
    sentence_tokens = set(_TOKEN_RE.findall(sentence.lower()))

    if not claim_tokens or not sentence_tokens:
        return 0

    coverage = len(claim_tokens & sentence_tokens) / len(claim_tokens)
    overlap = len(claim_tokens & sentence_tokens) / max(len(sentence_tokens), 1)
    sequence = SequenceMatcher(
        None, claim.text.lower(), sentence.lower()
    ).ratio()

    return round(coverage * 60 + sequence * 20 + overlap * 20)


def extract_evidence(
    claim: Claim,
    candidate: SourceCandidate,
    *,
    max_passages: int = 3,
) -> list[Evidence]:
    """Extract and rank evidence passages from *candidate*'s abstract.

    Returns up to *max_passages* passages sorted by descending relevance.
    When the candidate has no abstract the result is an empty list.
    """
    abstract = candidate.abstract
    if not abstract or not abstract.strip():
        return []

    sentences = split_evidence_sentences(abstract)
    if not sentences:
        return []

    seen: set[str] = set()
    unique: list[str] = []
    for s in sentences:
        normalized = s.lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(s)

    scored = [
        (evidence_relevance(claim, sentence), sentence)
        for sentence in unique
    ]
    scored.sort(key=lambda item: item[0], reverse=True)

    results: list[Evidence] = []
    for score, sentence in scored[:max_passages]:
        results.append(
            Evidence(
                text=sentence,
                source_title=candidate.title,
                source_api=candidate.source_api,
                evidence_type=EvidenceType.ABSTRACT,
                lexical_score=score,
                verdict=_verdict_from_score(score),
            )
        )
    return results


def _verdict_from_score(score: int) -> Verdict:
    """Map a lexical relevance score to a preliminary verdict.

    This is *not* the final verdict -- it is a hint for downstream
    entailment evaluation.  A high lexical score does not mean the
    source supports the claim; it means the source is topically
    relevant.
    """
    if score >= 70:
        return Verdict.PARTIALLY_SUPPORTED
    return Verdict.INSUFFICIENT_INFORMATION
