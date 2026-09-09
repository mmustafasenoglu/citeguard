"""Candidate ranking after hard integrity validation."""

from __future__ import annotations

from .models import MeaningVerdict, RewriteCandidate


def rank_candidates(candidates: list[RewriteCandidate]) -> list[RewriteCandidate]:
    """Return accepted-looking candidates in deterministic quality order.

    Validation is intentionally separate: candidates with failed hard gates
    are excluded by their rejection reasons before scoring.
    """
    eligible = [
        candidate for candidate in candidates
        if not candidate.rejection_reasons
        and candidate.citations_preserved is True
        and candidate.numeric_integrity is True
        and candidate.factual_integrity is True
        and candidate.meaning_verdict == MeaningVerdict.PRESERVED
    ]

    def score(candidate: RewriteCandidate) -> float:
        meaning = candidate.meaning_score if candidate.meaning_score is not None else 0.0
        support = (
            candidate.source_support_score
            if candidate.source_support_score is not None
            else 0.0
        )
        structural = 1.0 - (candidate.lexical_overlap or 0.0)
        overlap_penalty = (candidate.lexical_overlap or 0.0) + (candidate.exact_overlap or 0.0)
        value = 0.30 * meaning + 0.25 * support + 0.15 * structural - 0.15 * overlap_penalty
        candidate.score = round(value, 6)
        return candidate.score

    return sorted(eligible, key=lambda candidate: (-score(candidate), candidate.text))
