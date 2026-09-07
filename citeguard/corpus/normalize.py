"""Corpus-specific text normalization.

Wraps the shared ``normalize_turkish`` function with additional
corpus-level cleaning: reference marker removal, header stripping,
and whitespace normalization. Also provides offset mapping for
accurate span translation.

The offset map is **transformation-aware**: every normalization step
operates on ``(char, original_index)`` pairs, so the final map is
always an exact alignment between normalized characters and their
original positions.  No heuristic ``build_offset_map`` realignment
is performed.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

MappedChar = tuple[str, int]

# ---------------------------------------------------------------------------
# Compiled patterns (module-level for reuse)
# ---------------------------------------------------------------------------

_REF_PATTERN = re.compile(r"\[\d+(?:[,\-–]\s*\d+)*\]")
_CITATION_PATTERN = re.compile(
    r"\([A-ZÀ-ÖØ-öø-ÿÇĞİÖŞÜ][a-zà-öø-ÿçğıöşü]+(?:\s+(?:et\s+al\.|ve\s+ark\.|"
    r"vd\.))?(?:,\s*\d{4})\)"
)
_WS_PATTERN = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_corpus_text(text: str) -> str:
    """Normalize a corpus text segment for similarity comparison.

    Steps:
    1. Strip leading/trailing whitespace.
    2. Remove common reference markers like ``[1]``, ``[2,3]``.
    3. Remove inline citations ``(Author, 2020)`` patterns.
    4. Apply Turkish-aware lowercasing.
    5. Collapse multiple whitespace into single spaces.
    """
    if not text:
        return ""
    normalized, _ = normalize_corpus_text_with_map(text)
    return normalized


def normalize_corpus_text_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize corpus text and return (normalized_text, offset_map).

    The offset map is built **transformation-aware**: every step
    propagates ``(char, original_index)`` pairs, so the final map
    exactly aligns normalized characters to their original positions.

    No heuristic ``build_offset_map(original, final)`` is used.

    Args:
        text: Original text to normalize.

    Returns:
        Tuple of (normalized_text, offset_map) where offset_map maps
        each normalized character index to its original index.
    """
    if not text:
        return "", []

    # Step 1: Initialise mapped pairs
    mapped: list[MappedChar] = [(ch, i) for i, ch in enumerate(text)]

    mapped = _strip_whitespace(mapped)
    mapped = _remove_pattern(mapped, _REF_PATTERN)
    mapped = _remove_pattern(mapped, _CITATION_PATTERN)
    mapped = _turkish_lower(mapped)
    mapped = _collapse_whitespace(mapped)

    normalized = "".join(ch for ch, _ in mapped)
    offset_map = [idx for _, idx in mapped]
    return normalized, offset_map


# ---------------------------------------------------------------------------
# Internal transformation helpers
# ---------------------------------------------------------------------------


def _strip_whitespace(mapped: list[MappedChar]) -> list[MappedChar]:
    """Remove leading and trailing whitespace while keeping indices."""
    if not mapped:
        return mapped
    start = 0
    while start < len(mapped) and mapped[start][0].isspace():
        start += 1
    end = len(mapped)
    while end > start and mapped[end - 1][0].isspace():
        end -= 1
    return mapped[start:end]


def _remove_pattern(mapped: list[MappedChar], pattern: re.Pattern) -> list[MappedChar]:
    """Remove regex matches from mapped chars.

    Rebuilds the character list by scanning the *current* text for
    pattern matches and dropping the matched characters while keeping
    the surrounding ones.
    """
    if not mapped:
        return mapped

    # Reconstruct the current text for regex matching
    current_text = "".join(ch for ch, _ in mapped)

    # Collect ranges to remove
    remove_ranges: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in pattern.finditer(current_text)
    ]

    if not remove_ranges:
        return mapped

    result: list[MappedChar] = []
    prev_end = 0
    for range_start, range_end in remove_ranges:
        result.extend(mapped[prev_end:range_start])
        prev_end = range_end
    result.extend(mapped[prev_end:])
    return result


def _turkish_lower(mapped: list[MappedChar]) -> list[MappedChar]:
    """Apply Turkish-aware lowercasing: İ→i, I→ı, then str.lower()."""
    result: list[MappedChar] = []
    for ch, idx in mapped:
        if ch == "İ":
            result.append(("i", idx))
        elif ch == "I":
            result.append(("ı", idx))
        else:
            result.append((ch.lower(), idx))
    return result


def _collapse_whitespace(mapped: list[MappedChar]) -> list[MappedChar]:
    """Collapse consecutive whitespace into a single space.

    Keeps the *first* whitespace character's original index.
    """
    if not mapped:
        return mapped
    result: list[MappedChar] = []
    prev_was_ws = False
    for ch, idx in mapped:
        if ch.isspace():
            if not prev_was_ws:
                result.append((" ", idx))
                prev_was_ws = True
        else:
            result.append((ch, idx))
            prev_was_ws = False
    return result


# ---------------------------------------------------------------------------
# Section header stripping (no offset tracking needed)
# ---------------------------------------------------------------------------


def strip_section_headers(text: str) -> str:
    """Remove numbered section headers (e.g. ``1. Giriş``, ``2.1 Yöntem``).

    Useful when pre-processing full documents before sentence splitting.
    """
    return re.sub(
        r"^\d+(?:\.\d+)*\.?\s+[A-ZÀ-ÖÇĞİÖŞÜ].*$",
        "",
        text,
        flags=re.MULTILINE,
    ).strip()
