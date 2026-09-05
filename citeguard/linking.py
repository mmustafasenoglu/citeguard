"""Citation-to-sentence and citation-to-bibliography linking."""

from __future__ import annotations

import re

from .bibliography import citation_matches_entry
from .models import BibliographyEntry, CitationContext, ExistingCitation

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ])")


def link_citations_to_contexts(
    paragraphs: list[str],
    citations: list[ExistingCitation],
    entries: list[BibliographyEntry],
) -> list[CitationContext]:
    """Link each citation to its containing sentence and matching bibliography entries."""
    contexts: list[CitationContext] = []
    for citation in citations:
        paragraph = paragraphs[citation.paragraph_index]
        sentence = _sentence_at_offset(paragraph, citation.char_offset)
        entry_indexes = [
            index
            for index, entry in enumerate(entries)
            if citation_matches_entry(citation, entry)
        ]
        contexts.append(
            CitationContext(
                citation=citation,
                sentence=sentence,
                bibliography_entry_indexes=entry_indexes,
            )
        )
    return contexts


def _sentence_at_offset(paragraph: str, offset: int) -> str:
    if not paragraph:
        return ""
    offset = max(0, min(offset, len(paragraph) - 1))
    start = 0
    end = len(paragraph)
    for boundary in SENTENCE_BOUNDARY_RE.finditer(paragraph):
        if boundary.end() <= offset:
            start = boundary.end()
            continue
        end = boundary.start()
        break
    return paragraph[start:end].strip()
