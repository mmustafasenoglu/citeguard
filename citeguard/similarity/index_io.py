"""Persistent SimilarityIndex I/O for .ctac/ directories.

Handles atomic save/load of corpus indexes including entries, fingerprints,
TF-IDF vectorizers, and optional embedding matrices.

Trust boundary
--------------
Serialized artifacts are loaded from **self-created** ``.ctac/`` directories
only.  Loading an index from an untrusted source is not supported and may
execute arbitrary code via pickle deserialization.

Manifest fields that trigger invalidation (artifact-producing config):
    schema_version, normalization_version, corpus_hash,
    fingerprint_config_hash, tfidf_config_hash, passage_config_hash,
    embedding_model, embedding_dimension, entry_count.

Runtime config (does NOT invalidate index):
    semantic_threshold, rrf_k, reranker weights, max_results_per_sentence,
    risk thresholds, quotation_coverage_threshold.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Manifest model
# ---------------------------------------------------------------------------

MANIFEST_VERSION = "1"
SCHEMA_VERSION = "1"


@dataclass(frozen=True, slots=True)
class IndexManifest:
    """Manifest describing a persisted ``.ctac/`` index."""

    schema_version: str = SCHEMA_VERSION
    normalization_version: str = "1"
    corpus_hash: str = ""
    fingerprint_config_hash: str = ""
    tfidf_config_hash: str = ""
    passage_config_hash: str = ""
    embedding_model: str = ""
    embedding_dimension: int = 0
    embedding_enabled: bool = False
    entry_count: int = 0
    created_at: str = ""
    citeguard_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexManifest:
        return cls(
            schema_version=data.get("schema_version", ""),
            normalization_version=data.get("normalization_version", ""),
            corpus_hash=data.get("corpus_hash", ""),
            fingerprint_config_hash=data.get("fingerprint_config_hash", ""),
            tfidf_config_hash=data.get("tfidf_config_hash", ""),
            passage_config_hash=data.get("passage_config_hash", ""),
            embedding_model=data.get("embedding_model", ""),
            embedding_dimension=data.get("embedding_dimension", 0),
            embedding_enabled=data.get("embedding_enabled", False),
            entry_count=data.get("entry_count", 0),
            created_at=data.get("created_at", ""),
            citeguard_version=data.get("citeguard_version", ""),
        )


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_config(*parts: Any) -> str:
    """Deterministic hash of config values that affect artifact generation."""
    raw = "|".join(str(p) for p in parts)
    return "sha256:" + _hash_text(raw)


def compute_corpus_hash(entries_jsonl: str) -> str:
    """SHA-256 of the canonical corpus content."""
    return "sha256:" + _hash_text(entries_jsonl)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_index(
    directory: Path,
    entries: list[dict[str, Any]],
    manifest: IndexManifest,
    embeddings: Any | None = None,
    tfidf_vectorizer: Any | None = None,
) -> None:
    """Atomically persist a ``.ctac/`` index directory.

    Writes to a temporary directory first, then renames for atomicity.
    """
    directory = Path(directory)
    tmp_dir = Path(tempfile.mkdtemp(dir=directory.parent, prefix=".ctac_tmp_"))
    try:
        # Write entries
        entries_path = tmp_dir / "entries.jsonl"
        with open(entries_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Write manifest
        manifest_path = tmp_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)

        # Write embeddings if present
        if embeddings is not None:
            import numpy as np

            np.save(str(tmp_dir / "embeddings.npy"), embeddings)

        # Write TF-IDF vectorizer if present
        if tfidf_vectorizer is not None:
            import pickle

            tfidf_path = tmp_dir / "tfidf_vectorizer.pkl"
            with open(tfidf_path, "wb") as f:
                pickle.dump(tfidf_vectorizer, f)

        # Atomic rename
        if directory.exists():
            shutil.rmtree(directory)
        shutil.move(str(tmp_dir), str(directory))
        logger.info("Index saved to %s (%d entries)", directory, manifest.entry_count)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def load_index(
    directory: Path,
) -> tuple[IndexManifest, list[dict[str, Any]], Any | None, Any | None]:
    """Load a persisted ``.ctac/`` index.

    Returns
    -------
    (manifest, entries, embeddings, tfidf_vectorizer)

    Raises
    ------
    FileNotFoundError
        If the directory or required files are missing.
    ValueError
        If the manifest is invalid or stale.
    """
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = IndexManifest.from_dict(json.load(f))

    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Schema version mismatch: expected {SCHEMA_VERSION}, "
            f"got {manifest.schema_version}. Rebuild the index."
        )

    # Load entries
    entries_path = directory / "entries.jsonl"
    entries: list[dict[str, Any]] = []
    if entries_path.exists():
        with open(entries_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

    # Load embeddings
    embeddings = None
    emb_path = directory / "embeddings.npy"
    if emb_path.exists():
        import numpy as np

        embeddings = np.load(str(emb_path))

    # Load TF-IDF vectorizer (trusted local only)
    tfidf_vectorizer = None
    tfidf_path = directory / "tfidf_vectorizer.pkl"
    if tfidf_path.exists():
        import pickle

        with open(tfidf_path, "rb") as f:
            tfidf_vectorizer = pickle.load(f)  # noqa: S301 — trusted local index

    return manifest, entries, embeddings, tfidf_vectorizer


def is_index_valid(
    directory: Path,
    expected_manifest: IndexManifest,
) -> bool:
    """Check whether a persisted index matches the expected configuration.

    Only artifact-producing config fields are compared.  Runtime config
    differences (thresholds, weights) do not invalidate the index.
    """
    try:
        actual, _, _, _ = load_index(directory)
    except (FileNotFoundError, ValueError):
        return False

    return (
        actual.schema_version == expected_manifest.schema_version
        and actual.normalization_version == expected_manifest.normalization_version
        and actual.corpus_hash == expected_manifest.corpus_hash
        and actual.fingerprint_config_hash == expected_manifest.fingerprint_config_hash
        and actual.tfidf_config_hash == expected_manifest.tfidf_config_hash
        and actual.passage_config_hash == expected_manifest.passage_config_hash
        and actual.embedding_model == expected_manifest.embedding_model
        and actual.embedding_dimension == expected_manifest.embedding_dimension
        and actual.entry_count == expected_manifest.entry_count
    )


def invalidate_index(directory: Path) -> None:
    """Remove a persisted index directory."""
    directory = Path(directory)
    if directory.exists():
        shutil.rmtree(directory)
        logger.info("Index invalidated: %s", directory)
