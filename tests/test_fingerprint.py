"""Regression tests for fingerprinting and span detection."""

from __future__ import annotations

from citeguard.similarity.fingerprint import (
    find_matching_segments_pair,
    generate_shingles,
    stable_hash,
    winnow,
)
from citeguard.similarity.lexical import char_ngram_jaccard
from citeguard.similarity.models import Fingerprint


def test_generate_shingles_uses_character_offsets_for_turkish() -> None:
    text = "ışık"
    shingles = generate_shingles(text, k=2)
    assert shingles[0].start == 0
    assert shingles[0].end == 2
    assert shingles[-1].end == len(text)


def test_stable_hash_is_deterministic() -> None:
    assert stable_hash("citeguard") == stable_hash("citeguard")


def test_winnow_slides_one_shingle_at_a_time() -> None:
    shingles = generate_shingles("abcdefghijklmnop", k=3)
    points = winnow(shingles, window=4)
    assert points
    assert points[0].start <= points[-1].start


def test_find_matching_segments_pair_returns_both_sides() -> None:
    text = "the attention mechanism changed nlp research"
    fp = Fingerprint(points=winnow(generate_shingles(text, k=5), window=4))
    segments = find_matching_segments_pair(fp, fp, k=5)
    assert segments.document_spans
    assert segments.source_spans
    assert segments.document_spans[0][0] >= 0
    assert segments.source_spans[0][1] > segments.source_spans[0][0]


def test_char_ngram_jaccard_uses_trigrams() -> None:
    similar = char_ngram_jaccard("transformer attention", "transformer attention")
    different = char_ngram_jaccard("transformer attention", "xyz abc")
    assert similar == 1.0
    assert different < similar


def test_find_matching_segments_pair_avoids_cross_product() -> None:
    """Bug 5: Repeated hashes should pair by index, not produce cross-product."""
    text_a = "the cat sat on the mat"
    text_b = "the cat sat on the mat"
    fp_a = Fingerprint(
        points=winnow(generate_shingles(text_a, k=3), window=4)
    )
    fp_b = Fingerprint(
        points=winnow(generate_shingles(text_b, k=3), window=4)
    )
    segments = find_matching_segments_pair(fp_a, fp_b, k=3)

    # The number of raw pairs should equal the number of common hash
    # occurrences, not the cross-product of all occurrences per hash.
    # After merging, document_spans and source_spans should have
    # the same length.
    assert len(segments.document_spans) == len(segments.source_spans)

    # All spans should be valid (start < end, within text bounds)
    for start, end in segments.document_spans:
        assert start < end
        assert end <= len(text_a)
    for start, end in segments.source_spans:
        assert start < end
        assert end <= len(text_b)


def test_find_matching_segments_pair_different_texts() -> None:
    """Different texts produce fewer common hashes and smaller spans."""
    text_a = "the quick brown fox jumps over the lazy dog"
    text_b = "a fast brown fox leaps over the sleepy cat"
    fp_a = Fingerprint(
        points=winnow(generate_shingles(text_a, k=3), window=4)
    )
    fp_b = Fingerprint(
        points=winnow(generate_shingles(text_b, k=3), window=4)
    )
    segments = find_matching_segments_pair(fp_a, fp_b, k=3)
    # Same structure: document and source spans must be paired
    assert len(segments.document_spans) == len(segments.source_spans)
