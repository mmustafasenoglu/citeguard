"""Corpus deduplication using BLAKE2b fingerprint hashes.

Removes exact and near-duplicate entries from a corpus to avoid
inflated similarity scores from redundant reference material.
"""

from __future__ import annotations

import hashlib

from citeguard.corpus.models import CorpusEntry


def _stable_hash(text: str) -> int:
    """Deterministic 64-bit hash via BLAKE2b."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "big")


def deduplicate_entries(
    entries: list[CorpusEntry],
    *,
    cross_document: bool = True,
) -> list[CorpusEntry]:
    """Remove duplicate entries based on normalized text hash.

    Parameters
    ----------
    entries:
        Flat list of corpus entries (may span multiple documents).
    cross_document:
        If ``True``, deduplicate across documents.
        If ``False``, only deduplicate within the same ``doc_id``.

    Returns
    -------
    list[CorpusEntry]
        Deduplicated entries with ``fingerprint_hash`` populated.
    """
    seen: set[int | tuple[str, int]] = set()
    result: list[CorpusEntry] = []

    for entry in entries:
        h = _stable_hash(entry.normalized_text)
        entry.fingerprint_hash = h

        if cross_document:
            key: int | tuple[str, int] = h
        else:
            key = (entry.doc_id, h)

        if key in seen:
            continue
        seen.add(key)
        result.append(entry)

    return result


def find_duplicates(
    entries: list[CorpusEntry],
) -> dict[int, list[CorpusEntry]]:
    """Group entries by their normalized text hash.

    Returns only groups with 2+ entries (actual duplicates).
    """
    groups: dict[int, list[CorpusEntry]] = {}
    for entry in entries:
        h = _stable_hash(entry.normalized_text)
        entry.fingerprint_hash = h
        groups.setdefault(h, []).append(entry)
    return {h: g for h, g in groups.items() if len(g) > 1}
