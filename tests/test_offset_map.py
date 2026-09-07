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


# ---------------------------------------------------------------------------
# Adversarial tests for corpus/normalize.py transformation-aware mapping
# ---------------------------------------------------------------------------


def _corpus_remap(original: str, norm_start: int, norm_end: int) -> str:
    """Helper: normalize with map, remap a span, return original slice."""
    from citeguard.corpus.normalize import normalize_corpus_text_with_map
    normalized, mapping = normalize_corpus_text_with_map(original)
    orig_start, orig_end = remap_span(norm_start, norm_end, mapping)
    return original[orig_start:orig_end]


def test_citation_removed_repeated_token_after() -> None:
    """'foo (Bar, 2020) bar' → last 'bar' maps to original last 'bar'."""
    original = "foo (Bar, 2020) bar"
    from citeguard.corpus.normalize import normalize_corpus_text_with_map
    normalized, mapping = normalize_corpus_text_with_map(original)
    assert normalized == "foo bar"
    start = normalized.rindex("bar")
    end = start + 3
    extracted = _corpus_remap(original, start, end)
    assert extracted == "bar"
    # Must map to the ORIGINAL last "bar", not the one inside citation
    orig_start, orig_end = remap_span(start, end, mapping)
    assert orig_start == original.rindex("bar")


def test_bracket_removed_repeated_token() -> None:
    """'foo [1] foo' → second 'foo' maps to original second 'foo'."""
    original = "foo [1] foo"
    from citeguard.corpus.normalize import normalize_corpus_text_with_map
    normalized, mapping = normalize_corpus_text_with_map(original)
    assert normalized == "foo foo"
    start = normalized.rindex("foo")
    end = start + 3
    orig_start, _ = remap_span(start, end, mapping)
    assert original[orig_start:orig_start + 3] == "foo"
    assert orig_start == original.rindex("foo")


def test_turkish_casing_and_citation_removal() -> None:
    """'İstanbul   (Yılmaz, 2024) İstanbul' → last 'İstanbul' correct."""
    original = "İstanbul   (Yılmaz, 2024) İstanbul"
    from citeguard.corpus.normalize import normalize_corpus_text_with_map
    normalized, mapping = normalize_corpus_text_with_map(original)
    assert "istanbul" in normalized
    # Find last istanbul in normalized
    start = normalized.rindex("istanbul")
    end = start + len("istanbul")
    extracted = _corpus_remap(original, start, end)
    assert extracted == "İstanbul"
    orig_start, _ = remap_span(start, end, mapping)
    assert orig_start == original.rindex("İstanbul")


def test_collapsed_whitespace_and_citation_removal() -> None:
    """'abc\\t\\tdef' → 'abc def' maps correctly."""
    original = "abc\t\tdef"
    from citeguard.corpus.normalize import normalize_corpus_text_with_map
    normalized, mapping = normalize_corpus_text_with_map(original)
    assert normalized == "abc def"
    start = normalized.index("def")
    end = start + 3
    extracted = _corpus_remap(original, start, end)
    assert extracted == "def"
    orig_start, _ = remap_span(start, end, mapping)
    assert original[orig_start:orig_start + 3] == "def"


def test_turkish_casing_and_bracket_removal() -> None:
    """'ışık [2] ışık' → second 'ışık' maps to original second."""
    original = "ışık [2] ışık"
    from citeguard.corpus.normalize import normalize_corpus_text_with_map
    normalized, mapping = normalize_corpus_text_with_map(original)
    assert normalized == "ışık ışık"
    start = normalized.rindex("ışık")
    end = start + 4
    extracted = _corpus_remap(original, start, end)
    assert extracted == "ışık"
    orig_start, _ = remap_span(start, end, mapping)
    assert orig_start == original.rindex("ışık")
