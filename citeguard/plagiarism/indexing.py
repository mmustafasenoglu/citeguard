"""Versioned persistent index support for plagiarism corpora."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from citeguard import __version__
from citeguard.similarity.index import SimilarityIndex
from citeguard.similarity.index_io import (
    IndexManifest,
    compute_corpus_hash,
    load_index_as_similarity_index,
    save_index,
)

from .models import PlagiarismConfig, PlagiarismSource

INDEX_ALGORITHM_VERSION = "plagiarism-1"


def cache_key(sources: list[PlagiarismSource], config: PlagiarismConfig) -> str:
    """Return a stable key covering content and artifact-producing settings."""
    payload = {
        "algorithm": INDEX_ALGORITHM_VERSION,
        "sources": [(source.source_id, source.content_hash) for source in sources],
        "shingle_size": 5,
        "winnow_window": 4,
        "ngram_range": [1, 3],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_directory(key: str) -> Path:
    return Path(".local/citeguard/plagiarism-index") / f"{key}.ctac"


def _metadata_dict(metadata: Any) -> dict[str, Any] | None:
    if metadata is None:
        return None
    payload = asdict(metadata)
    language = payload.get("language")
    if hasattr(language, "value"):
        payload["language"] = language.value
    return payload


def _entries(index: SimilarityIndex) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for doc_id, normalized, fingerprint, metadata, source_entry_index, source in index.entries:
        text = getattr(source, "original_text", None) or getattr(source, "text", normalized)
        serialized.append(
            {
                "passage_id": doc_id,
                "normalized_text": normalized,
                "fingerprint": {
                    "doc_id": fingerprint.doc_id,
                    "points": [
                        {
                            "hash": point.hash,
                            "position": point.position,
                            "start": point.start,
                            "end": point.end,
                        }
                        for point in fingerprint.points
                    ],
                },
                "metadata": _metadata_dict(metadata),
                "source_entry_index": source_entry_index,
                "char_offset": getattr(source, "char_offset", 0),
                "original_text": text,
                "offset_map": getattr(source, "offset_map", None) or [],
            }
        )
    return serialized


def save_cached_index(
    index: SimilarityIndex,
    sources: list[PlagiarismSource],
    config: PlagiarismConfig,
    *,
    directory: Path | None = None,
) -> Path:
    """Atomically persist a self-created lexical index."""
    key = cache_key(sources, config)
    destination = directory or cache_directory(key)
    serialized = _entries(index)
    canonical = "\n".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) for item in serialized
    )
    manifest = IndexManifest(
        normalization_version="1",
        corpus_hash=compute_corpus_hash(canonical),
        fingerprint_config_hash="sha256:plagiarism-shingle-5-window-4",
        tfidf_config_hash="sha256:plagiarism-tfidf-1-3",
        passage_config_hash="sha256:sentence-passages-v1",
        entry_count=len(serialized),
        passage_count=len(serialized),
        citeguard_version=__version__,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_index(
        destination,
        serialized,
        manifest,
        tfidf_vectorizer=index.tfidf_vectorizer,
        tfidf_matrix=index.tfidf_matrix,
    )
    return destination


def load_cached_index(
    sources: list[PlagiarismSource], config: PlagiarismConfig
) -> SimilarityIndex | None:
    """Load only the deterministic cache path generated for these sources."""
    if config.no_cache:
        return None
    directory = cache_directory(cache_key(sources, config))
    if not directory.exists():
        return None
    try:
        return load_index_as_similarity_index(directory)
    except (FileNotFoundError, OSError, ValueError):
        return None
