"""Deterministic fingerprinting for Citeguard similarity.

Uses BLAKE2b for stable hashing, ordered k-shingles, and sliding-window
winnowing. All functions are pure and dependency-free (no numpy/scipy).
"""

from __future__ import annotations

import hashlib

from citeguard.similarity.models import Fingerprint, FingerprintPoint, Shingle

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

    shingles: list[Shingle] = []
    text_bytes = text.encode("utf-8")

    for i in range(len(text_bytes) - k + 1):
        chunk = text_bytes[i : i + k]
        h = stable_hash(chunk.decode("utf-8", errors="ignore"))
        shingles.append(
            Shingle(hash=h, start=i, end=i + k, position=i)
        )

    return shingles


# ---------------------------------------------------------------------------
# Sliding-window winnowing
# ---------------------------------------------------------------------------

def winnow(shingles: list[Shingle], window: int = 4) -> list[FingerprintPoint]:
    """Select one fingerprint point per window using minimum-hash.

    For every consecutive group of *window* shingles, the shingle with the
    smallest hash value becomes a fingerprint point. This compresses the
    signature while preserving roughly even coverage across the text.

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

    points: list[FingerprintPoint] = []
    num_shingles = len(shingles)

    # Slide window; last window may be smaller
    start = 0
    while start < num_shingles:
        end = min(start + window, num_shingles)
        window_shingles = shingles[start:end]

        # Find shingle with minimum hash
        min_shingle = min(window_shingles, key=lambda s: s.hash)

        points.append(
            FingerprintPoint(
                hash=min_shingle.hash,
                position=min_shingle.position,
                start=min_shingle.start,
                end=min_shingle.end,
            )
        )
        start += window

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
    if not fp1.points or not fp2.points:
        return []

    # Map hash -> list of positions for each fingerprint
    pos1: dict[int, list[int]] = {}
    for p in fp1.points:
        pos1.setdefault(p.hash, []).append(p.position)

    pos2: dict[int, list[int]] = {}
    for p in fp2.points:
        pos2.setdefault(p.hash, []).append(p.position)

    # Find hashes present in both fingerprints
    common_hashes = set(pos1.keys()) & set(pos2.keys())
    if not common_hashes:
        return []

    # Collect matching spans as (start, end) in fp1's coordinate system
    # We use the start position from fp1 and extend to the end of the
    # corresponding fp1 point.
    raw_spans: list[tuple[int, int]] = []
    for h in common_hashes:
        for p1_pos in pos1[h]:
            # Find matching fp2 point with same position (or closest)
            matching_fp2_positions = pos2[h]
            # Use the first matching position as anchor
            matching_fp2_positions[0]
            # Find the fp1 point at this position
            matching_p1 = None
            for p in fp1.points:
                if p.hash == h and p.position == p1_pos:
                    matching_p1 = p
                    break
            if matching_p1 is None:
                continue
            raw_spans.append((matching_p1.start, matching_p1.end))

    if not raw_spans:
        return []

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

    return merged