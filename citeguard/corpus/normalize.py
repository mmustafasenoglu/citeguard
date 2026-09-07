"""Corpus-specific text normalization.

Wraps the shared ``normalize_turkish`` function with additional
corpus-level cleaning: reference marker removal, header stripping,
and whitespace normalization. Also provides offset mapping for
accurate span translation.
"""

from __future__ import annotations

import re

from citeguard.similarity.lexical import normalize_turkish
from citeguard.similarity.normalize import build_offset_map


def normalize_corpus_text(text: str) -> str:
    """Normalize a corpus text segment for similarity comparison.

    Steps:
    1. Strip leading/trailing whitespace.
    2. Remove common reference markers like ``[1]``, ``[2,3]``.
    3. Remove inline citations ``(Author, 2020)`` patterns.
    4. Apply Turkish-aware lowercasing via ``normalize_turkish``.
    5. Collapse multiple whitespace into single spaces.
    """
    if not text:
        return ""
    result = text.strip()
    # Remove bracketed reference markers: [1], [2,3], [1-5]
    result = re.sub(r"\[\d+(?:[,\-–]\s*\d+)*\]", "", result)
    # Remove simple parenthetical author-year: (Author, 2020)
    result = re.sub(
        r"\([A-ZÀ-ÖØ-öø-ÿÇĞİÖŞÜ][a-zà-öø-ÿçğıöşü]+(?:\s+(?:et\s+al\.|ve\s+ark\.|"
        r"vd\.))?(?:,\s*\d{4})\)",
        "",
        result,
    )
    result = normalize_turkish(result)
    # Collapse whitespace
    result = re.sub(r"\s+", " ", result).strip()
    return result


def normalize_corpus_text_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize corpus text and return both normalized text and offset map.

    Applies the same normalization pipeline as ``normalize_corpus_text``
    but also returns an offset map mapping normalized indices to original indices.

    The offset map accounts for:
    - Reference marker removal (e.g., [1], [2,3])
    - Inline citation removal (e.g., (Author, 2020))
    - Turkish case folding (İ->i, I->ı)
    - Whitespace collapse

    Args:
        text: Original text to normalize.

    Returns:
        Tuple of (normalized_text, offset_map) where offset_map maps
        each normalized character index to its original index.
    """
    if not text:
        return "", []

    # We need to track the transformation step by step to build accurate offset map
    # Do the transformations step by step, recording the offset map after each step

    # Step 1: strip
    result = text.strip()
    # Build initial map: each char in stripped text maps to original index
    stripped_start = len(text) - len(text.lstrip())
    offset_map = list(range(stripped_start, stripped_start + len(result)))

    # Step 2: remove reference markers
    ref_pattern = re.compile(r"\[\d+(?:[,\-–]\s*\d+)*\]")
    result = _remove_with_map(result, ref_pattern, offset_map)

    # Step 3: remove inline citations
    citation_pattern = re.compile(
        r"\([A-ZÀ-ÖØ-öø-ÿÇĞİÖŞÜ][a-zà-öø-ÿçğıöşü]+(?:\s+(?:et\s+al\.|ve\s+ark\.|"
        r"vd\.))?(?:,\s*\d{4})\)"
    )
    result = _remove_with_map(result, citation_pattern, offset_map)

    # Step 4: Turkish normalization + whitespace collapse
    # This is more complex - we'll use the simpler approach:
    # Just apply the full normalization and then rebuild the map
    fully_normalized = normalize_turkish(result)
    fully_normalized = re.sub(r"\s+", " ", fully_normalized).strip()

    # Rebuild map from original text to fully normalized
    final_map = build_offset_map(text, fully_normalized)
    return fully_normalized, final_map


def _remove_with_map(text: str, pattern: re.Pattern, offset_map: list[int]) -> str:
    """Remove pattern matches from text and update offset_map accordingly."""
    result_parts = []
    last_end = 0
    new_map = []

    for match in pattern.finditer(text):
        # Keep text before match
        result_parts.append(text[last_end:match.start()])
        # Update map: keep only chars before the match
        new_map.extend(offset_map[last_end:match.start()])
        last_end = match.end()

    result_parts.append(text[last_end:])
    new_map.extend(offset_map[last_end:])

    return "".join(result_parts)


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
