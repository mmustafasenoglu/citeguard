"""Deterministic candidate measurements before integrity validation."""

from __future__ import annotations

from difflib import SequenceMatcher

from .models import RewriteCandidate


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def lexical_overlap(original: str, candidate: str) -> float:
    """Return token-set overlap, independent of any model provider."""
    original_tokens = set(_tokens(original))
    candidate_tokens = set(_tokens(candidate))
    if not original_tokens or not candidate_tokens:
        return 0.0
    return len(original_tokens & candidate_tokens) / len(original_tokens | candidate_tokens)


def evaluate_candidate(original: str, candidate: RewriteCandidate) -> RewriteCandidate:
    """Populate basic overlap and meaning-proxy measurements.

    The sequence score is only a screening signal, not an entailment verdict.
    A future semantic/NLI evaluator may replace ``meaning_score`` while keeping
    this provider-independent interface.
    """
    candidate.lexical_overlap = lexical_overlap(original, candidate.text)
    candidate.exact_overlap = SequenceMatcher(
        None, original.casefold(), candidate.text.casefold()
    ).ratio()
    candidate.semantic_similarity_to_original = candidate.exact_overlap
    if candidate.meaning_score is None:
        candidate.meaning_score = candidate.semantic_similarity_to_original
    return candidate
