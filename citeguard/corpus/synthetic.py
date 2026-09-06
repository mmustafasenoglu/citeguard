"""Synthetic corpus generation for test fixtures only.

These functions produce small, deterministic test documents.
Synthetic entries are NOT part of the main CTAC corpus — they exist
solely to support offline unit and integration tests.
"""

from __future__ import annotations

from citeguard.corpus.models import (
    CorpusDocument,
    CorpusEntry,
    CorpusLanguage,
    CorpusMetadata,
)
from citeguard.corpus.normalize import normalize_corpus_text


def make_test_entry(
    text: str,
    *,
    doc_id: str = "synthetic-test",
    entry_index: int = 0,
) -> CorpusEntry:
    """Create a single corpus entry for testing."""
    normalized = normalize_corpus_text(text)
    return CorpusEntry(
        text=text,
        normalized_text=normalized,
        doc_id=doc_id,
        entry_index=entry_index,
        char_offset=0,
        char_end=len(text),
    )


def make_test_document(
    sentences: list[str],
    *,
    doc_id: str = "synthetic-test",
    title: str = "Synthetic Test Document",
    language: CorpusLanguage = CorpusLanguage.TURKISH,
) -> CorpusDocument:
    """Create a corpus document from a list of sentences for testing.

    Parameters
    ----------
    sentences:
        Raw sentence strings.
    doc_id:
        Document identifier.
    title:
        Document title for metadata.
    language:
        Language classification.

    Returns
    -------
    CorpusDocument
        A document with normalized entries, ready for fingerprinting.
    """
    entries: list[CorpusEntry] = []
    offset = 0
    for idx, text in enumerate(sentences):
        normalized = normalize_corpus_text(text)
        entries.append(
            CorpusEntry(
                text=text,
                normalized_text=normalized,
                doc_id=doc_id,
                entry_index=idx,
                char_offset=offset,
                char_end=offset + len(text),
            )
        )
        offset += len(text) + 1  # +1 for space separator

    metadata = CorpusMetadata(
        title=title,
        language=language,
        license="CC0",
    )

    return CorpusDocument(
        doc_id=doc_id,
        metadata=metadata,
        entries=entries,
        raw_text=" ".join(sentences),
    )


# -----------------------------------------------------------------------
# Pre-built fixtures for common test scenarios
# -----------------------------------------------------------------------

TURKISH_ACADEMIC_SENTENCES: list[str] = [
    "Bu çalışmada Türk edebiyatının temel dönemleri incelenmiştir.",
    "Araştırma sonuçları istatistiksel olarak anlamlı bulunmuştur.",
    "Deneysel veriler hipotezi destekler niteliktedir.",
    "Sonuç olarak, önerilen yöntem mevcut yaklaşımlardan üstündür.",
    "Literatür taraması kapsamlı bir şekilde gerçekleştirilmiştir.",
]

ENGLISH_ACADEMIC_SENTENCES: list[str] = [
    "This study examines the fundamental periods of Turkish literature.",
    "Research results were found to be statistically significant.",
    "Experimental data supports the proposed hypothesis.",
    "In conclusion, the proposed method outperforms existing approaches.",
    "A comprehensive literature review was conducted.",
]
