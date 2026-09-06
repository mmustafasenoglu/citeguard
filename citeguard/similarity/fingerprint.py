"""Deterministic fingerprinting for Citeguard similarity.

Uses BLAKE2b for stable hashing, ordered k-shingles, and sliding-window
winnowing. All functions are pure and dependency-free (no numpy/scipy).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from citeguard.similarity.models import Fingerprint, FingerprintPoint, Shingle


@dataclass(frozen=True, slots=True)
class MatchingSegments:
    """Matching spans in both compared fingerprint coordinate spaces."""

    document_spans: list[tuple[int, int]]
    source_spans: list[tuple[int, int]]

# ---------------------------------------------------------------------------
# Stable 64-bit hash
# ---------------------------------------------------------------------------

def stable_hash(text: str) -> int:
    """Deterministic 64-bit hash via BLAKE2b.

    Python's built-in ``hash()`` is randomized per-process; this function
    gives process-independent results critical for fingerprint indexing and
    cross-machine reproducibility.

    Parameters
    ----------
    text:
        Input string to hash.

    Returns
    -------
    int
        Unsigned 64-bit integer.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "big")


# ---------------------------------------------------------------------------
# Ordered k-shingles
# ---------------------------------------------------------------------------

def generate_shingles(text: str, k: int = 5) -> list[Shingle]:
    """Return ordered k-grams with character positions.

    Unlike a set, the returned list preserves order and position information.
    This is essential for winnowing and contiguous span detection later.

    Parameters
    ----------
    text:
        Input text to shingle.
    k:
        Size of each k-gram in characters.

    Returns
    -------
    list[Shingle]
    """
    if k <= 0 or not text:
        return []
        
    if len(text) < k:
        return [Shingle(hash=stable_hash(text), start=0, end=len(text), position=0)]

    shingles: list[Shingle] = []
    
    for i in range(len(text) - k + 1):
        chunk = text[i : i + k]
        h = stable_hash(chunk)
        shingles.append(
            Shingle(hash=h, start=i, end=i + k, position=i)
        )

    return shingles


# ---------------------------------------------------------------------------
# Sliding-window winnowing
# ---------------------------------------------------------------------------

def winnow(shingles: list[Shingle], window: int = 4) -> list[FingerprintPoint]:
    """Select one fingerprint point per window using minimum-hash.

    Implements true sliding-window winnowing (Schleimer et al.).
    For every consecutive group of *window* shingles, the shingle with the
    smallest hash value is selected. Tie-breaking favors the right-most match.

    Parameters
    ----------
    shingles:
        Ordered list of Shingle objects from ``generate_shingles``.
    window:
        Number of shingles per selection window (default 4).

    Returns
    -------
    list[FingerprintPoint]
    """
    if not shingles:
        return []
        
    if len(shingles) <= window:
        # Shorter than window, pick absolute minimum
        min_shingle = min(shingles, key=lambda s: s.hash)
        return [FingerprintPoint(
            hash=min_shingle.hash, position=min_shingle.position,
            start=min_shingle.start, end=min_shingle.end
        )]

    points: list[FingerprintPoint] = []
    
    # Initialize first window
    min_idx = 0
    for i in range(1, window):
        if shingles[i].hash <= shingles[min_idx].hash:
            min_idx = i
            
    points.append(FingerprintPoint(
        hash=shingles[min_idx].hash, position=shingles[min_idx].position,
        start=shingles[min_idx].start, end=shingles[min_idx].end
    ))
    
    # Slide window
    for i in range(1, len(shingles) - window + 1):
        new_shingle_idx = i + window - 1
        
        # If previous minimum fell out of the window, scan the whole window
        if min_idx < i:
            min_idx = i
            for j in range(i + 1, i + window):
                if shingles[j].hash <= shingles[min_idx].hash:
                    min_idx = j
            points.append(FingerprintPoint(
                hash=shingles[min_idx].hash, position=shingles[min_idx].position,
                start=shingles[min_idx].start, end=shingles[min_idx].end
            ))
        else:
            # Previous minimum is still in window. Compare only the new shingle
            if shingles[new_shingle_idx].hash <= shingles[min_idx].hash:
                min_idx = new_shingle_idx
                points.append(FingerprintPoint(
                    hash=shingles[min_idx].hash, position=shingles[min_idx].position,
                    start=shingles[min_idx].start, end=shingles[min_idx].end
                ))

    return points


# ---------------------------------------------------------------------------
# Overlap metrics
# ---------------------------------------------------------------------------

def exact_overlap(fp1: Fingerprint, fp2: Fingerprint) -> float:
    """Jaccard-style overlap on fingerprint hashes.

    ``|hash_intersection| / |hash_union|`` where intersection/union are
    computed on the *set* of hash values contained in each fingerprint.

    Parameters
    ----------
    fp1, fp2:
        Fingerprints to compare.

    Returns
    -------
    float
        Overlap in [0, 1].
    """
    if not fp1.points or not fp2.points:
        return 0.0

    set1 = {p.hash for p in fp1.points}
    set2 = {p.hash for p in fp2.points}
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union else 0.0


def find_matching_segments(
    fp1: Fingerprint,
    fp2: Fingerprint,
    *,
    k: int = 5,
) -> list[tuple[int, int]]:
    """Contiguous character spans from matching fingerprint points.

    Given two fingerprints, this identifies regions where fingerprint points
    overlap in both hash value and approximate position. Adjacent overlapping
    points are merged into contiguous spans.

    Parameters
    ----------
    fp1, fp2:
        Fingerprints to compare.
    k:
        Shingle size used originally (informational only).

    Returns
    -------
    list[tuple[int, int]]
        (start, end) character spans (in the *first* document's coordinate
        system) that match between the two fingerprints.
    """
    return find_matching_segments_pair(fp1, fp2, k=k).document_spans


def find_matching_segments_pair(
    fp1: Fingerprint,
    fp2: Fingerprint,
    *,
    k: int = 5,
) -> MatchingSegments:
    """Return matching spans in both fingerprint coordinate systems."""
    if not fp1.points or not fp2.points:
        return MatchingSegments(document_spans=[], source_spans=[])

    # Map hash -> list of positions for each fingerprint
    pos1: dict[int, list[FingerprintPoint]] = {}
    for p in fp1.points:
        pos1.setdefault(p.hash, []).append(p)

    pos2: dict[int, list[FingerprintPoint]] = {}
    for p in fp2.points:
        pos2.setdefault(p.hash, []).append(p)

    # Find hashes present in both fingerprints
    common_hashes = set(pos1.keys()) & set(pos2.keys())
    if not common_hashes:
        return MatchingSegments(document_spans=[], source_spans=[])

    # Collect matching spans as (start, end) in fp1's coordinate system
    # We use the start position from fp1 and extend to the end of the
    # corresponding fp1 point.
    raw_spans: list[tuple[int, int]] = []
    raw_source_spans: list[tuple[int, int]] = []
    for h in common_hashes:
        for p1 in pos1[h]:
            raw_spans.append((p1.start, p1.end))
        for p2 in pos2[h]:
            raw_source_spans.append((p2.start, p2.end))

    if not raw_spans:
        return MatchingSegments(document_spans=[], source_spans=[])

    # Merge adjacent/overlapping spans
    raw_spans.sort()
    merged: list[tuple[int, int]] = []
    for span in raw_spans:
        if not merged:
            merged.append(span)
        else:
            last_start, last_end = merged[-1]
            cur_start, cur_end = span
            # Merge if overlapping or touching (cur_start <= last_end)
            if cur_start <= last_end:
                merged[-1] = (last_start, max(last_end, cur_end))
            else:
                merged.append(span)

    return MatchingSegments(
        document_spans=merged,
        source_spans=_merge_tuple_spans(raw_source_spans),
    )


def _merge_tuple_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge adjacent/overlapping tuple spans."""
    if not spans:
        return []
    spans.sort()
    merged: list[tuple[int, int]] = []
    for span in spans:
        if not merged:
            merged.append(span)
            continue
        last_start, last_end = merged[-1]
        cur_start, cur_end = span
        if cur_start <= last_end:
            merged[-1] = (last_start, max(last_end, cur_end))
        else:
            merged.append(span)
    return merged
