"""Persistent SimilarityIndex I/O for .ctac/ directories.

Handles atomic save/load of corpus indexes including entries, fingerprints,
TF-IDF vectorizers, TF-IDF matrices, and optional embedding matrices.

Trust boundary
--------------
Serialized artifacts are loaded from **self-created** ``.ctac/`` directories
only.  Loading an index from an untrusted source is not supported and may
execute arbitrary code via pickle deserialization.

Manifest fields that trigger invalidation (artifact-producing config):
    schema_version, normalization_version, corpus_hash,
    fingerprint_config_hash, tfidf_config_hash, passage_config_hash,
    embedding_model, embedding_dimension, embedding_normalized, entry_count,
    passage_count, embedding_enabled.

Runtime config (does NOT invalidate index):
    semantic_threshold, rrf_k, reranker weights, max_results_per_sentence,
    risk thresholds, quotation_coverage_threshold.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
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
    embedding_normalized: bool = False
    entry_count: int = 0
    passage_count: int = 0
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
            embedding_normalized=data.get("embedding_normalized", False),
            entry_count=data.get("entry_count", 0),
            passage_count=data.get("passage_count", 0),
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
# Temp index validation (P1-7)
# ---------------------------------------------------------------------------


def _validate_temp_index(
    tmp_dir: Path,
    manifest: IndexManifest,
) -> None:
    """Validate a newly-written temporary index directory before swap.

    Checks:
        - manifest.json is parseable and schema matches
        - entries.jsonl exists and entry count matches manifest
        - passage_count consistency
        - TF-IDF matrix row count matches indexed passages when both present
        - embeddings presence when embedding_enabled=True
        - embeddings row count matches indexed passages
        - embedding dimension matches manifest
        - corpus/config manifest fields are internally consistent

    Raises
    ------
    ValueError
        If validation fails.
    """
    # 1. Manifest parseable and schema valid
    manifest_path = tmp_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Temp index missing manifest.json")

    with open(manifest_path, encoding="utf-8") as f:
        loaded = IndexManifest.from_dict(json.load(f))

    if loaded.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Temp index schema version mismatch: {loaded.schema_version}"
        )

    if loaded.entry_count != manifest.entry_count:
        raise ValueError(
            f"Temp index entry_count mismatch: {loaded.entry_count} != {manifest.entry_count}"
        )

    if loaded.passage_count != manifest.passage_count:
        raise ValueError(
            f"Temp index passage_count mismatch: {loaded.passage_count} != {manifest.passage_count}"
        )

    # 2. entries.jsonl exists and line count matches
    entries_path = tmp_dir / "entries.jsonl"
    if not entries_path.exists():
        raise ValueError("Temp index missing entries.jsonl")

    line_count = 0
    with open(entries_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                line_count += 1

    if line_count != manifest.entry_count:
        raise ValueError(
            f"Temp index entries.jsonl line count {line_count} != manifest {manifest.entry_count}"
        )

    # 3. TF-IDF matrix row count (only validate if both manifest claims it and file exists)
    tfidf_mat_path = tmp_dir / "tfidf_matrix.npz"
    if loaded.tfidf_config_hash and tfidf_mat_path.exists():
        from scipy import sparse
        matrix = sparse.load_npz(str(tfidf_mat_path))
        if matrix.shape[0] != manifest.entry_count:
            raise ValueError(
                f"TF-IDF matrix rows {matrix.shape[0]} != entry count {manifest.entry_count}"
            )

    # 4. Embedding artifacts
    if manifest.embedding_enabled:
        emb_path = tmp_dir / "embeddings.npy"
        if not emb_path.exists():
            raise ValueError("Temp index missing embeddings.npy (embedding_enabled=True)")

        import numpy as np
        embeddings = np.load(str(emb_path))

        if embeddings.shape[0] != manifest.entry_count:
            raise ValueError(
                f"Embedding rows {embeddings.shape[0]} != entry count {manifest.entry_count}"
            )

        if manifest.embedding_dimension > 0 and embeddings.shape[1] != manifest.embedding_dimension:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} != manifest {manifest.embedding_dimension}"
            )

        # Check normalization contract: each row should have unit norm
        if manifest.embedding_normalized and embeddings.shape[0] > 0:
            norms = np.linalg.norm(embeddings, axis=1)
            non_unit = np.abs(norms - 1.0) > 0.1
            if np.any(non_unit):
                raise ValueError(
                    "Embedding normalization contract violated: "
                    f"{np.sum(non_unit)} vectors have non-unit norm"
                )


# ---------------------------------------------------------------------------
# Save / load with backup/rollback
# ---------------------------------------------------------------------------


def save_index(
    directory: Path,
    entries: list[dict[str, Any]],
    manifest: IndexManifest,
    embeddings: Any | None = None,
    tfidf_vectorizer: Any | None = None,
    tfidf_matrix: Any | None = None,
) -> None:
    """Persist a ``.ctac/`` index directory with backup/rollback safety.

    Strategy:
        1. Build new index in .ctac_tmp/
        2. Validate temp index
        3. Rename existing .ctac → .ctac.backup (if exists)
        4. Rename .ctac_tmp → .ctac
        5. Delete .ctac.backup on success
        6. Rollback .ctac.backup → .ctac on rename failure
    """
    directory = Path(directory)
    tmp_dir = Path(tempfile.mkdtemp(dir=directory.parent, prefix=".ctac_tmp_"))
    backup_dir = directory.parent / ".ctac.backup"
    try:
        # --- Write new index to temp directory ---
        entries_path = tmp_dir / "entries.jsonl"
        with open(entries_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        manifest_path = tmp_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)

        if embeddings is not None:
            import numpy as np

            np.save(str(tmp_dir / "embeddings.npy"), embeddings)

        if tfidf_vectorizer is not None:
            import pickle

            with open(tmp_dir / "tfidf_vectorizer.pkl", "wb") as f:
                pickle.dump(tfidf_vectorizer, f)

        if tfidf_matrix is not None:
            from scipy import sparse

            sparse.save_npz(str(tmp_dir / "tfidf_matrix.npz"), tfidf_matrix)

        # --- Validate temp index before swap (P1-7) ---
        _validate_temp_index(tmp_dir, manifest)

        # --- Backup/swap/rollback ---
        # Clean stale tmp from prior failed saves
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        # Rename existing .ctac → .ctac.backup
        if directory.exists():
            os.rename(str(directory), str(backup_dir))

        # Rename new tmp → .ctac
        os.rename(str(tmp_dir), str(directory))

        # Success: delete backup
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        logger.info(
            "Index saved to %s (%d entries, %d passages)",
            directory,
            manifest.entry_count,
            manifest.passage_count,
        )
    except Exception:
        # Rollback: restore .ctac.backup if rename failed
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if backup_dir.exists() and not directory.exists():
            os.rename(str(backup_dir), str(directory))
        elif backup_dir.exists():
            shutil.rmtree(backup_dir)
        raise


# ---------------------------------------------------------------------------
# Deep index validation helper (P1-8, P1-9)
# ---------------------------------------------------------------------------


def _is_directory_usable(directory: Path) -> bool:
    """Deep-check if a .ctac directory has all required artifacts and valid data.

    Used by ``recover_index_state()`` to determine whether an index is
    genuinely usable.  Validates:
        - manifest.json parseable and schema valid
        - entries.jsonl exists with correct line count
        - TF-IDF vectorizer + matrix present when tfidf_config_hash set
        - TF-IDF matrix rows == entry_count
        - embeddings present when embedding_enabled=True
        - embeddings rows == entry_count
        - embedding dimension matches manifest
        - normalization contract when embedding_normalized=True
    """
    if not directory.exists():
        return False

    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return False

    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = IndexManifest.from_dict(json.load(f))
    except (json.JSONDecodeError, ValueError, KeyError):
        return False

    if manifest.schema_version != SCHEMA_VERSION:
        return False

    entries_path = directory / "entries.jsonl"
    if not entries_path.exists():
        return False

    # Verify entry line count matches manifest
    line_count = 0
    with open(entries_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                line_count += 1

    if line_count != manifest.entry_count:
        return False

    # Check TF-IDF artifacts: vectorizer.pkl AND tfidf_matrix.npz must exist
    if manifest.tfidf_config_hash:
        if not (directory / "tfidf_vectorizer.pkl").exists():
            return False
        tfidf_mat_path = directory / "tfidf_matrix.npz"
        if not tfidf_mat_path.exists():
            return False
        # Validate TF-IDF matrix row count
        try:
            from scipy import sparse
            matrix = sparse.load_npz(str(tfidf_mat_path))
            if matrix.shape[0] != manifest.entry_count:
                return False
        except Exception:
            return False

    # Check embedding artifacts
    if manifest.embedding_enabled:
        emb_path = directory / "embeddings.npy"
        if not emb_path.exists():
            return False

        try:
            import numpy as np
            embeddings = np.load(str(emb_path))
            if embeddings.shape[0] != manifest.entry_count:
                return False
            if (manifest.embedding_dimension > 0
                    and embeddings.shape[1] != manifest.embedding_dimension):
                return False
            # Normalization contract
            if manifest.embedding_normalized and embeddings.shape[0] > 0:
                norms = np.linalg.norm(embeddings, axis=1)
                non_unit = np.abs(norms - 1.0) > 0.1
                if np.any(non_unit):
                    return False
        except Exception:
            return False

    return True


def recover_index_state(directory: Path) -> str:
    """Check and recover index state after a potential crash.

    Recovery logic:
        - valid primary → use primary and clean stale temp
        - invalid/missing primary + valid backup → restore backup
        - invalid primary + invalid backup → raise clear error
        - stale temp → remove safely

    Never deletes the only valid index during recovery.

    Returns one of:
        ``"valid"``    — .ctac is present and valid
        ``"restored"`` — backup was restored
        ``"stale_cleaned"`` — stale .tmp was cleaned
        ``"missing"``  — neither .ctac nor backup exists

    Raises ValueError when both .ctac and backup are corrupt.
    """
    directory = Path(directory)
    backup_dir = directory.parent / ".ctac.backup"
    stale_tmp_dirs = sorted(directory.parent.glob(".ctac_tmp_*"))

    primary_usable = _is_directory_usable(directory)
    backup_usable = _is_directory_usable(backup_dir)

    # Case 1: primary valid → clean stale tmp and backup
    if primary_usable:
        for tmp in stale_tmp_dirs:
            shutil.rmtree(tmp, ignore_errors=True)
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
        return "valid"

    # Case 2: primary invalid/missing + backup valid → restore
    if backup_usable:
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
        os.rename(str(backup_dir), str(directory))
        for tmp in stale_tmp_dirs:
            shutil.rmtree(tmp, ignore_errors=True)
        logger.info("Restored index from backup: %s", directory)
        return "restored"

    # Case 3: both invalid → clean stale tmp and raise
    for tmp in stale_tmp_dirs:
        shutil.rmtree(tmp, ignore_errors=True)

    if directory.exists() and backup_dir.exists():
        # Both exist but neither is usable — raise
        raise ValueError(
            f"Both primary index ({directory}) and backup ({backup_dir}) are corrupt. "
            "Rebuild the index."
        )

    if directory.exists():
        # Primary exists but is not usable, no backup → clean and report missing
        shutil.rmtree(directory, ignore_errors=True)
        return "missing"

    if backup_dir.exists():
        # Backup exists but is not usable → clean and report missing
        shutil.rmtree(backup_dir, ignore_errors=True)

    # Case 4: stale tmps only → clean
    if stale_tmp_dirs:
        return "stale_cleaned"

    # Case 5: nothing exists
    return "missing"


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


def load_tfidf_matrix(directory: Path) -> Any | None:
    """Load the persisted TF-IDF sparse matrix, or *None* if absent."""
    directory = Path(directory)
    npz_path = directory / "tfidf_matrix.npz"
    if not npz_path.exists():
        return None
    from scipy import sparse

    return sparse.load_npz(str(npz_path))


def load_index_as_similarity_index(
    directory: Path,
    config: Any | None = None,
) -> Any:
    """Load a persisted index and reconstruct a usable ``SimilarityIndex``.

    This is the recommended entry point for loading.  It handles:
        - manifest validation
        - TF-IDF vectorizer + matrix reconstruction
        - embedding loading
        - SimilarityIndex construction

    Returns
    -------
    SimilarityIndex

    Raises
    ------
    FileNotFoundError
        If the directory or required files are missing.
    ValueError
        If the manifest is invalid.
    """
    from citeguard.similarity.index import SimilarityIndex

    manifest, entries_raw, embeddings, tfidf_vectorizer = load_index(directory)
    tfidf_matrix = load_tfidf_matrix(directory)

    # Reconstruct CorpusEntryTuple list from JSONL
    entries = _rebuild_entries_from_dicts(entries_raw)

    return SimilarityIndex(
        entries=entries,
        fingerprints=[e[2] for e in entries] if entries else [],
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        embeddings=embeddings,
        embedding_model=manifest.embedding_model or None,
        embedding_dim=manifest.embedding_dimension or None,
    )


def _rebuild_entries_from_dicts(
    raw_entries: list[dict[str, Any]],
) -> list[Any]:
    """Reconstruct CorpusEntryTuple list from persisted JSONL dicts.

    Each dict has the format produced by ``save_index()``:
        {passage_id, normalized_text, fingerprint, metadata,
         source_entry_index, original_text, offset_map, ...}
    """
    from citeguard.similarity.models import Fingerprint, FingerprintPoint

    result: list[Any] = []
    for d in raw_entries:
        # Reconstruct Fingerprint
        fp_data = d.get("fingerprint", {})
        fp_points = [
            FingerprintPoint(hash=p["hash"], position=p["position"],
                             start=p.get("start", 0), end=p.get("end", 0))
            for p in fp_data.get("points", [])
        ]
        fingerprint = Fingerprint(
            points=fp_points,
            doc_id=fp_data.get("doc_id"),
        )

        # Reconstruct CorpusMetadata
        metadata_dict = d.get("metadata")
        metadata = None
        if metadata_dict:
            try:
                from citeguard.corpus.models import CorpusMetadata

                metadata = CorpusMetadata(
                    title=metadata_dict.get("title", ""),
                    authors=metadata_dict.get("authors", []),
                    year=metadata_dict.get("year"),
                    language=metadata_dict.get("language", ""),
                    license=metadata_dict.get("license", ""),
                    similarity_index_allowed=metadata_dict.get(
                        "similarity_index_allowed", True
                    ),
                    doi=metadata_dict.get("doi"),
                    url=metadata_dict.get("url"),
                )
            except Exception:
                pass

        # Build passage-level source object for span remapping
        passage_source = _PassageSource(
            original_text=d.get("original_text", ""),
            normalized_text=d.get("normalized_text", ""),
            offset_map=d.get("offset_map", []),
        )

        doc_id = d.get("passage_id", d.get("doc_id", ""))
        normalized_text = d.get("normalized_text", "")

        result.append((
            doc_id,
            normalized_text,
            fingerprint,
            metadata,
            d.get("source_entry_index", 0),
            passage_source,
        ))
    return result


class _PassageSource:
    """Minimal passage-level source for span remapping.

    Carries the same interface that engine.py expects from the
    sixth element of a CorpusEntryTuple (``corpus_entry_obj``):
        - ``.text``       → original text of this passage
        - ``.offset_map`` → normalized→original offset map
    """

    __slots__ = ("text", "offset_map")

    def __init__(self, original_text: str, normalized_text: str,
                 offset_map: list[int]) -> None:
        self.text = original_text
        self.offset_map = offset_map


def is_index_valid(
    directory: Path,
    expected_manifest: IndexManifest,
) -> bool:
    """Check whether a persisted index matches the expected configuration.

    Artifact-producing config fields are compared and deep artifact
    validation is performed.  Runtime config differences (thresholds,
    weights) do not invalidate the index.
    """
    try:
        actual, _, _, _ = load_index(directory)
    except (FileNotFoundError, ValueError):
        return False

    # Check required artifact files exist
    directory = Path(directory)
    entries_path = directory / "entries.jsonl"
    if not entries_path.exists():
        return False

    # Validate entry line count matches manifest
    line_count = 0
    with open(entries_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                line_count += 1
    if line_count != actual.entry_count:
        return False

    # Validate TF-IDF artifacts when expected
    if actual.tfidf_config_hash:
        if not (directory / "tfidf_vectorizer.pkl").exists():
            return False
        tfidf_mat_path = directory / "tfidf_matrix.npz"
        if not tfidf_mat_path.exists():
            return False
        try:
            from scipy import sparse
            matrix = sparse.load_npz(str(tfidf_mat_path))
            if matrix.shape[0] != actual.entry_count:
                return False
        except Exception:
            return False

    # Validate embedding artifacts when expected
    if actual.embedding_enabled:
        emb_path = directory / "embeddings.npy"
        if not emb_path.exists():
            return False
        try:
            import numpy as np
            embeddings = np.load(str(emb_path))
            if embeddings.shape[0] != actual.entry_count:
                return False
            if (actual.embedding_dimension > 0
                    and embeddings.shape[1] != actual.embedding_dimension):
                return False
            if actual.embedding_normalized and embeddings.shape[0] > 0:
                norms = np.linalg.norm(embeddings, axis=1)
                non_unit = np.abs(norms - 1.0) > 0.1
                if np.any(non_unit):
                    return False
        except Exception:
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
        and actual.embedding_normalized == expected_manifest.embedding_normalized
        and actual.embedding_enabled == expected_manifest.embedding_enabled
        and actual.entry_count == expected_manifest.entry_count
        and actual.passage_count == expected_manifest.passage_count
    )


def invalidate_index(directory: Path) -> None:
    """Remove a persisted index directory."""
    directory = Path(directory)
    if directory.exists():
        shutil.rmtree(directory)
        logger.info("Index invalidated: %s", directory)
