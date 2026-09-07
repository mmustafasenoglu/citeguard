"""Corpus ingest: read files and produce CorpusDocument objects.

Supports plain-text (.txt), Markdown (.md), and DOCX (.docx) input.
Each document is split into sentences, normalized, and packaged as
``CorpusEntry`` objects inside a ``CorpusDocument``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from citeguard.corpus.licenses import classify_license, get_permissions
from citeguard.corpus.models import (
    CorpusDocument,
    CorpusEntry,
    CorpusLanguage,
    CorpusMetadata,
)
from citeguard.corpus.normalize import normalize_corpus_text_with_map
from citeguard.extractor import read_paragraphs
from citeguard.sentence_splitter import split_sentences

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".docx"})


def _detect_language(text: str) -> CorpusLanguage:
    """Simple heuristic language detection based on Turkish characters."""
    turkish_chars = set("çğıöşüÇĞİÖŞÜ")
    count = sum(1 for ch in text if ch in turkish_chars)
    ratio = count / max(len(text), 1)
    if ratio > 0.005:
        return CorpusLanguage.TURKISH
    return CorpusLanguage.ENGLISH


def ingest_file(
    path: str | Path,
    *,
    doc_id: str | None = None,
    license_str: str | None = None,
    metadata: CorpusMetadata | None = None,
) -> CorpusDocument:
    """Ingest a single file into a ``CorpusDocument``.

    Parameters
    ----------
    path:
        File to read (.txt, .md, or .docx).
    doc_id:
        Unique identifier. Defaults to the filename stem.
    license_str:
        Raw license string for classification.
    metadata:
        Pre-built metadata. When provided, ``license_str`` is ignored.

    Returns
    -------
    CorpusDocument
        Document with normalized, sentence-split entries.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the file does not exist.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Corpus file not found: {filepath}")
    if filepath.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {filepath.suffix}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )

    actual_id = doc_id or filepath.stem
    paragraphs = read_paragraphs(filepath)
    raw_text = "\n\n".join(paragraphs)

    # Build metadata if not provided
    if metadata is None:
        license_type = classify_license(license_str)
        language = _detect_language(raw_text)
        perms = get_permissions(license_type)
        metadata = CorpusMetadata(
            title=filepath.stem.replace("-", " ").replace("_", " ").title(),
            license=license_type.value,
            language=language,
            training_use_allowed=perms["training_use_allowed"],
            similarity_index_allowed=perms["similarity_index_allowed"],
            commercial_use_allowed=perms["commercial_use_allowed"],
        )
        
    if metadata and not metadata.similarity_index_allowed:
        # If explicitly not allowed for similarity index, log a warning and return empty entries
        logger.warning(
            "Document '%s' license (%s) restricts similarity indexing. Skipping.", 
            actual_id, metadata.license
        )
        return CorpusDocument(
            doc_id=actual_id,
            metadata=metadata,
            entries=[],
            raw_text=raw_text,
        )

    # Split into sentences and normalize
    entries: list[CorpusEntry] = []
    global_idx = 0
    char_cursor = 0
    for para_text in paragraphs:
        sentences = split_sentences(para_text)
        for sent in sentences:
            normalized, offset_map = normalize_corpus_text_with_map(sent.text)
            if not normalized or len(normalized) < 10:
                char_cursor += len(sent.text)
                continue
            entries.append(
                CorpusEntry(
                    text=sent.text,
                    normalized_text=normalized,
                    doc_id=actual_id,
                    entry_index=global_idx,
                    metadata=metadata,
                    char_offset=char_cursor + sent.start_offset,
                    char_end=char_cursor + sent.end_offset,
                    offset_map=offset_map,
                )
            )
            global_idx += 1
        # Account for paragraph separator
        char_cursor += len(para_text) + 2  # "\n\n"

    return CorpusDocument(
        doc_id=actual_id,
        metadata=metadata,
        entries=entries,
        raw_text=raw_text,
    )


def ingest_directory(
    directory: str | Path,
    *,
    license_str: str | None = None,
    recursive: bool = False,
) -> list[CorpusDocument]:
    """Ingest all supported files from a directory.

    Parameters
    ----------
    directory:
        Directory to scan.
    license_str:
        Default license for all files (can be overridden per-file).
    recursive:
        If ``True``, scan subdirectories recursively.

    Returns
    -------
    list[CorpusDocument]
        Successfully ingested documents (failures are logged and skipped).
    """
    dirpath = Path(directory)
    if not dirpath.is_dir():
        raise NotADirectoryError(f"Not a directory: {dirpath}")

    pattern = "**/*" if recursive else "*"
    documents: list[CorpusDocument] = []

    for filepath in sorted(dirpath.glob(pattern)):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            continue
        try:
            doc = ingest_file(filepath, license_str=license_str)
            documents.append(doc)
        except Exception:
            logger.warning("Failed to ingest %s", filepath, exc_info=True)
            continue

    return documents
