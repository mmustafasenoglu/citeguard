"""Tests for persistent SimilarityIndex I/O (.ctac/ directory)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from citeguard.similarity.index_io import (
    IndexManifest,
    compute_corpus_hash,
    invalidate_index,
    is_index_valid,
    load_index,
    save_index,
)

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_roundtrip() -> None:
    m = IndexManifest(
        schema_version="1",
        corpus_hash="sha256:abc",
        entry_count=42,
        embedding_model="test-model",
        embedding_dimension=384,
    )
    d = m.to_dict()
    m2 = IndexManifest.from_dict(d)
    assert m == m2


def test_manifest_default_values() -> None:
    m = IndexManifest()
    assert m.schema_version == "1"
    assert m.corpus_hash == ""
    assert m.entry_count == 0


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def test_hash_deterministic() -> None:
    h1 = compute_corpus_hash("hello world")
    h2 = compute_corpus_hash("hello world")
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_hash_different_inputs() -> None:
    assert compute_corpus_hash("a") != compute_corpus_hash("b")


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    idx_dir = tmp_path / "test.ctac"
    entries = [{"doc_id": "d1", "text": "hello"}, {"doc_id": "d2", "text": "world"}]
    manifest = IndexManifest(
        entry_count=2,
        corpus_hash="sha256:test",
        embedding_model="test",
        embedding_dimension=64,
    )

    save_index(idx_dir, entries, manifest)

    loaded_manifest, loaded_entries, embeddings, tfidf = load_index(idx_dir)
    assert loaded_manifest.entry_count == 2
    assert loaded_manifest.corpus_hash == "sha256:test"
    assert len(loaded_entries) == 2
    assert loaded_entries[0]["doc_id"] == "d1"
    assert embeddings is None
    assert tfidf is None


def test_save_load_with_embeddings(tmp_path: Path) -> None:
    import numpy as np

    idx_dir = tmp_path / "test.ctac"
    entries = [{"doc_id": "d1"}]
    emb = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    manifest = IndexManifest(
        entry_count=1, embedding_enabled=True,
        embedding_model="m", embedding_dimension=3,
    )

    save_index(idx_dir, entries, manifest, embeddings=emb)

    _, _, loaded_emb, _ = load_index(idx_dir)
    assert loaded_emb is not None
    assert loaded_emb.shape == (1, 3)


def test_save_load_with_tfidf(tmp_path: Path) -> None:
    idx_dir = tmp_path / "test.ctac"
    entries = [{"doc_id": "d1"}]
    mock_vectorizer = {"vocabulary": ["a", "b", "c"]}
    manifest = IndexManifest(entry_count=1)

    save_index(idx_dir, entries, manifest, tfidf_vectorizer=mock_vectorizer)

    _, _, _, loaded_tfidf = load_index(idx_dir)
    assert loaded_tfidf is not None
    assert loaded_tfidf["vocabulary"] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Atomic save (overwrite existing)
# ---------------------------------------------------------------------------


def test_save_overwrites_existing(tmp_path: Path) -> None:
    idx_dir = tmp_path / "test.ctac"
    manifest1 = IndexManifest(entry_count=1)
    save_index(idx_dir, [{"doc_id": "d1"}], manifest1)

    manifest2 = IndexManifest(entry_count=2)
    save_index(idx_dir, [{"doc_id": "d1"}, {"doc_id": "d2"}], manifest2)

    _, entries, _, _ = load_index(idx_dir)
    assert len(entries) == 2


# ---------------------------------------------------------------------------
# is_index_valid
# ---------------------------------------------------------------------------


def test_is_valid_when_matching(tmp_path: Path) -> None:
    idx_dir = tmp_path / "test.ctac"
    manifest = IndexManifest(
        corpus_hash="sha256:x", entry_count=5,
        fingerprint_config_hash="sha256:fp",
        tfidf_config_hash="sha256:tf",
        passage_config_hash="sha256:pg",
        embedding_model="m", embedding_dimension=128,
    )
    save_index(idx_dir, [{"i": i} for i in range(5)], manifest)
    assert is_index_valid(idx_dir, manifest) is True


def test_is_invalid_when_corpus_hash_differs(tmp_path: Path) -> None:
    idx_dir = tmp_path / "test.ctac"
    manifest = IndexManifest(corpus_hash="sha256:old", entry_count=1)
    save_index(idx_dir, [{"i": 0}], manifest)

    expected = IndexManifest(corpus_hash="sha256:new", entry_count=1)
    assert is_index_valid(idx_dir, expected) is False


def test_is_invalid_when_missing(tmp_path: Path) -> None:
    idx_dir = tmp_path / "nonexistent.ctac"
    expected = IndexManifest()
    assert is_index_valid(idx_dir, expected) is False


def test_is_invalid_when_schema_mismatch(tmp_path: Path) -> None:
    idx_dir = tmp_path / "test.ctac"
    manifest = IndexManifest(schema_version="1", entry_count=1)
    save_index(idx_dir, [{"i": 0}], manifest)

    expected = IndexManifest(schema_version="2", entry_count=1)
    assert is_index_valid(idx_dir, expected) is False


# ---------------------------------------------------------------------------
# invalidate_index
# ---------------------------------------------------------------------------


def test_invalidate_removes_directory(tmp_path: Path) -> None:
    idx_dir = tmp_path / "test.ctac"
    save_index(idx_dir, [{"i": 0}], IndexManifest(entry_count=1))
    assert idx_dir.exists()

    invalidate_index(idx_dir)
    assert not idx_dir.exists()


def test_invalidate_nonexistent_is_noop(tmp_path: Path) -> None:
    invalidate_index(tmp_path / "nope.ctac")  # should not raise


# ---------------------------------------------------------------------------
# load_index error cases
# ---------------------------------------------------------------------------


def test_load_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "nope.ctac")


def test_load_wrong_schema_raises(tmp_path: Path) -> None:
    idx_dir = tmp_path / "test.ctac"
    idx_dir.mkdir()
    manifest_data = {"schema_version": "999", "entry_count": 0}
    (idx_dir / "manifest.json").write_text(json.dumps(manifest_data))

    with pytest.raises(ValueError, match="Schema version mismatch"):
        load_index(idx_dir)
