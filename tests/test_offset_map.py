"""Regression tests for normalized <-> original offset mapping (Bug 2)."""

from __future__ import annotations

from citeguard.similarity.normalize import (
    build_offset_map,
    normalize_with_map,
    remap_span,
)


def test_build_offset_map_identity() -> None:
    """When original equals normalized, mapping is 1:1."""
    text = "hello world"
    norm = "hello world"
    mapping = build_offset_map(text, norm)
    assert mapping == list(range(len(text)))


def test_build_offset_map_collapses_whitespace() -> None:
    """Multiple spaces in original collapse to single space in normalized."""
    original = "hello   world"
    normalized = "hello world"
    mapping = build_offset_map(original, normalized)
    assert len(mapping) == len(normalized)
    # 'h' at 0 -> 0, 'e' at 1 -> 1, ..., 'w' should map to index 5 (after
    # the three spaces)
    assert mapping[6] >= 5


def test_build_offset_map_turkish_case_fold() -> None:
    """Turkish İ folds to i and I folds to ı."""
    original = "İSTANBUL"
    normalized = "istanbul"
    mapping = build_offset_map(original, normalized)
    assert len(mapping) == len(normalized)
    assert mapping[0] == 0  # İ -> i


def test_remap_span_simple() -> None:
    """Remap a span from normalized coordinates to original coordinates."""
    original = "hello   world"
    normalized = "hello world"
    mapping = build_offset_map(original, normalized)
    # Remap "world" in normalized coords (6..11) to original
    orig_start, orig_end = remap_span(6, 11, mapping)
    assert original[orig_start:orig_end] == "world"


def test_remap_span_empty_map_passthrough() -> None:
    """Empty offset_map returns the input coords unchanged."""
    assert remap_span(2, 5, []) == (2, 5)


def test_remap_span_clamps_out_of_range() -> None:
    """Coords beyond map length are clamped."""
    original = "abc"
    normalized = "abc"
    mapping = build_offset_map(original, normalized)
    orig_start, orig_end = remap_span(0, 100, mapping)
    assert orig_start == 0
    assert orig_end == len(original)


def test_normalize_with_map_returns_tuple() -> None:
    """normalize_with_map returns (normalized_text, offset_map)."""
    text = "Attention  is  all  you  need"
    normalized, offset_map = normalize_with_map(text)
    assert normalized == "attention is all you need"
    assert len(offset_map) == len(normalized)


def test_offset_map_is_monotonic() -> None:
    """Mapping from normalized to original is monotonic non-decreasing."""
    original = "The   quick   brown   fox"
    normalized = "the quick brown fox"
    mapping = build_offset_map(original, normalized)
    for i in range(1, len(mapping)):
        assert mapping[i] >= mapping[i - 1]
