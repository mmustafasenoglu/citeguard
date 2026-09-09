"""Source-aware textual overlap measurement for rewrite candidates."""

from __future__ import annotations

from ..similarity.fingerprint import exact_overlap, generate_shingles, winnow
from ..similarity.lexical import char_ngram_jaccard, normalize_turkish
from ..similarity.models import Fingerprint
from .models import RewriteCandidate

SOURCE_OVERLAP_EPSILON = 1e-9


def _fingerprint(text: str) -> Fingerprint:
    normalized = normalize_turkish(text)
    return Fingerprint(points=winnow(generate_shingles(normalized)))


def evaluate_source_overlap(
    original: str,
    candidate: RewriteCandidate,
    matched_source_text: str,
    *,
    epsilon: float = SOURCE_OVERLAP_EPSILON,
) -> RewriteCandidate:
    """Measure source↔original and source↔candidate textual overlap.

    Improvement requires at least one textual signal to decrease and neither
    to increase. Semantic cosine is deliberately not part of this objective.
    Missing source text fails closed.
    """
    if not matched_source_text.strip():
        candidate.source_overlap_improved = False
        candidate.rejection_reasons.append("matched source passage is unavailable")
        return candidate

    source_fp = _fingerprint(matched_source_text)
    before_exact = exact_overlap(_fingerprint(original), source_fp)
    after_exact = exact_overlap(_fingerprint(candidate.text), source_fp)
    before_lexical = char_ngram_jaccard(original, matched_source_text)
    after_lexical = char_ngram_jaccard(candidate.text, matched_source_text)
    improved = (
        after_exact <= before_exact + epsilon
        and after_lexical <= before_lexical + epsilon
        and (
            after_exact < before_exact - epsilon
            or after_lexical < before_lexical - epsilon
        )
    )
    before = max(before_exact, before_lexical)
    after = max(after_exact, after_lexical)
    candidate.source_exact_overlap_before = before_exact
    candidate.source_exact_overlap_after = after_exact
    candidate.source_lexical_similarity_before = before_lexical
    candidate.source_lexical_similarity_after = after_lexical
    candidate.source_overlap_delta = before - after
    candidate.source_overlap_improved = improved
    if not improved:
        candidate.rejection_reasons.append(
            "candidate did not reduce textual overlap with the matched source"
        )
    return candidate
