"""Data models for CTAC v1 corpus entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CorpusLanguage(str, Enum):
    """Supported corpus languages."""

    TURKISH = "tr"
    ENGLISH = "en"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class CorpusMetadata:
    """Bibliographic metadata for a corpus document."""

    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    source: str | None = None
    license: str | None = None
    language: CorpusLanguage = CorpusLanguage.TURKISH
    url: str | None = None
    
    # Permissions based on license
    training_use_allowed: bool = False
    similarity_index_allowed: bool = False
    commercial_use_allowed: bool = False


@dataclass(slots=True)
class CorpusEntry:
    """A single text segment within a corpus document.

    Each entry represents a sentence or short passage that has been
    normalized and fingerprinted for similarity comparison.
    """

    text: str
    normalized_text: str
    doc_id: str
    entry_index: int
    char_offset: int = 0
    char_end: int = 0
    fingerprint_hash: int | None = None


@dataclass(slots=True)
class CorpusDocument:
    """A full document in the corpus with its entries and metadata.

    Attributes
    ----------
    doc_id:
        Unique identifier (typically filename or DOI).
    metadata:
        Bibliographic information.
    entries:
        List of normalized text segments.
    raw_text:
        Original full text before splitting/normalization.
    """

    doc_id: str
    metadata: CorpusMetadata
    entries: list[CorpusEntry] = field(default_factory=list)
    raw_text: str = ""
