"""Normalized <-> Original offset mapping for accurate span translation.

When text is normalized (whitespace collapsed, Turkish case-folded, citation markers
removed), character positions shift. This module provides bidirectional mapping
between normalized and original coordinates so that spans detected on normalized
text can be accurately translated back to original text coordinates.
"""

from __future__ import annotations


def build_offset_map(original: str, normalized: str) -> list[int]:
    """Build a mapping from normalized indices to original indices.

    Walks both strings simultaneously using a two-pointer approach, tracking
    the original index for each normalized character. This handles:
    - Whitespace collapse (multiple spaces -> single space)
    - Turkish case folding (İ -> i, I -> ı, etc.)
    - Character removals (citation markers, reference brackets)

    Args:
        original: The original text before normalization.
        normalized: The normalized text.

    Returns:
        A list of length len(normalized) where each entry i contains the
        corresponding index in the original string. The mapping is monotonic
        non-decreasing.
    """
    if not original or not normalized:
        return []

    mapping: list[int] = []
    orig_idx = 0
    norm_idx = 0

    while norm_idx < len(normalized) and orig_idx < len(original):
        orig_char = original[orig_idx]
        norm_char = normalized[norm_idx]

        # Handle Turkish İ/I case folding
        if _chars_match_after_normalization(orig_char, norm_char):
            mapping.append(orig_idx)
            orig_idx += 1
            norm_idx += 1
        # Handle whitespace collapse: multiple spaces in original -> single in normalized
        elif orig_char.isspace() and norm_char == " ":
            # Skip all consecutive whitespace in original
            while orig_idx < len(original) and original[orig_idx].isspace():
                orig_idx += 1
            mapping.append(orig_idx)
            norm_idx += 1
        # Character removed during normalization (citation markers, etc.)
        else:
            # Original has extra char not in normalized -> skip it
            orig_idx += 1

    # Fill remaining normalized positions with last valid original index
    last_valid = mapping[-1] if mapping else len(original)
    while len(mapping) < len(normalized):
        mapping.append(min(last_valid, len(original) - 1))

    return mapping


def _chars_match_after_normalization(orig: str, norm: str) -> bool:
    """Check if original char matches normalized char after Turkish normalization."""
    if orig == norm:
        return True
    # Turkish İ -> i, I -> ı
    if orig == "İ" and norm == "i":
        return True
    if orig == "I" and norm == "ı":
        return True
    # Lowercase for other characters
    return orig.lower() == norm


def remap_span(
    normalized_start: int,
    normalized_end: int,
    offset_map: list[int],
) -> tuple[int, int]:
    """Map a span from normalized coordinates to original coordinates.

    Args:
        normalized_start: Start index in normalized text (inclusive).
        normalized_end: End index in normalized text (exclusive).
        offset_map: Output from build_offset_map().

    Returns:
        Tuple of (original_start, original_end) in original text coordinates.
        Both inclusive-exclusive like the input.
    """
    if not offset_map:
        return (normalized_start, normalized_end)

    # Clamp to valid range
    clamped_start = max(0, min(normalized_start, len(offset_map) - 1))
    clamped_end = max(0, min(normalized_end, len(offset_map)))

    if clamped_start >= clamped_end:
        return (offset_map[clamped_start], offset_map[clamped_start])

    original_start = offset_map[clamped_start]
    # For end, we want the position after the last character
    original_end = (
        offset_map[clamped_end - 1] + 1 if clamped_end > 0
        else original_start
    )

    return (original_start, original_end)


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize text and return both normalized text and offset map.

    Uses the same normalization pipeline as the corpus/similarity code:
    1. Turkish-aware lowercasing (İ->i, I->ı)
    2. Whitespace collapse
    2. Character removal (handled externally in normalize_corpus_text)

    This is a simplified version for the document side where we only do
    Turkish case-folding and whitespace collapse.
    """
    from citeguard.similarity.lexical import normalize_turkish

    # For document sentences, we only apply Turkish normalization + whitespace collapse
    # (no citation marker removal - that's corpus-side only)
    normalized = normalize_turkish(text)
    # The normalize_turkish already collapses whitespace
    offset_map = build_offset_map(text, normalized)
    return normalized, offset_map