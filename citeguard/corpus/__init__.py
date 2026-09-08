"""CTAC v1 corpus infrastructure for Citeguard similarity engine.

Provides ingest, normalization, deduplication, and license management
for academic text corpora used in similarity comparison.
"""

from __future__ import annotations

from citeguard.corpus.deduplicate import deduplicate_entries
from citeguard.corpus.ingest import ingest_directory, ingest_file
from citeguard.corpus.licenses import LicenseType, is_open_license
from citeguard.corpus.loader import load_similarity_corpus
from citeguard.corpus.models import CorpusDocument, CorpusEntry, CorpusMetadata
from citeguard.corpus.normalize import normalize_corpus_text

__all__ = [
    "CorpusDocument",
    "CorpusEntry",
    "CorpusMetadata",
    "LicenseType",
    "deduplicate_entries",
    "ingest_directory",
    "ingest_file",
    "is_open_license",
    "load_similarity_corpus",
    "normalize_corpus_text",
]
