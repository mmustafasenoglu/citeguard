"""Candidate ranking after hard integrity validation."""

from __future__ import annotations

from .models import MeaningVerdict, RewriteCandidate


def rank_candidates(candidates: list[RewriteCandidate]) -> list[RewriteCandidate]:
    """Return accepted-looking candidates in deterministic quality order.

    Validation is intentionally separate: candidates with failed hard gates
    are excluded by their rejection reasons before scoring.
    """
    eligible = [
        candidate
        for candidate in candidates
        if not candidate.rejection_reasons
        and candidate.citations_preserved is True
        and candidate.numeric_integrity is True
        and candidate.factual_integrity is True
        and candidate.meaning_verdict == MeaningVerdict.PRESERVED
        and not candidate.introduced_claims
        and candidate.source_overlap_improved is True
    ]

    def score(candidate: RewriteCandidate) -> float:
        meaning = candidate.meaning_score if candidate.meaning_score is not None else 0.0
        support = (
            candidate.source_support_score if candidate.source_support_score is not None else 0.0
        )
        source_reduction = max(candidate.source_overlap_delta or 0.0, 0.0)
        structural = 1.0 - (candidate.lexical_overlap or 0.0)
        value = 0.40 * meaning + 0.30 * support + 0.20 * source_reduction + 0.10 * structural
        candidate.score = round(value, 6)
        return candidate.score

    return sorted(eligible, key=lambda candidate: (-score(candidate), candidate.text))
