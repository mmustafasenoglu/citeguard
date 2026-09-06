"""Corpus-specific text normalization.

Wraps the shared ``normalize_turkish`` function with additional
corpus-level cleaning: reference marker removal, header stripping,
and whitespace normalization.
"""

from __future__ import annotations

import re

from citeguard.similarity.lexical import normalize_turkish


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
