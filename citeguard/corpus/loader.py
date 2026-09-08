"""Corpus loading and similarity-index building.

Provides ``load_similarity_corpus``, the library-level entry point
for ingesting a corpus path and returning a ready ``SimilarityIndex``.
Both the CLI and the product/audit layer import from this module
so that core code never depends on ``citeguard.cli``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .deduplicate import deduplicate_entries
from .ingest import ingest_directory, ingest_file

if TYPE_CHECKING:
    from ..similarity.index import SimilarityIndex


def load_similarity_corpus(
    corpus_path: Path | None, license_str: str | None = None
) -> SimilarityIndex:
    """Load and fingerprint a similarity corpus.

    Returns a ``SimilarityIndex`` holding corpus entries, fingerprints,
    and TF-IDF state.  When *corpus_path* is ``None`` an empty index
    is returned.
    """
    from ..similarity.fingerprint import generate_shingles, winnow
    from ..similarity.index import SimilarityIndex
    from ..similarity.models import Fingerprint as _Fingerprint

    if not corpus_path:
        return SimilarityIndex()

    print(f"Loading corpus from {corpus_path}...", file=sys.stderr)
    if corpus_path.is_dir():
        docs = ingest_directory(corpus_path, recursive=True, license_str=license_str)
    else:
        docs = [ingest_file(corpus_path, license_str=license_str)]

    all_entries = []
    for doc in docs:
        all_entries.extend(doc.entries)

    deduped = deduplicate_entries(all_entries)

    corpus_entries = []
    for entry in deduped:
        shingles = generate_shingles(entry.normalized_text)
        points = winnow(shingles)
        fp = _Fingerprint(points=points, doc_id=entry.doc_id)
        corpus_entries.append(
            (entry.doc_id, entry.normalized_text, fp, entry.metadata, entry.entry_index, entry)
        )

    print(f"Corpus loaded: {len(docs)} docs, {len(deduped)} unique segments", file=sys.stderr)
    return SimilarityIndex.build(corpus_entries)
