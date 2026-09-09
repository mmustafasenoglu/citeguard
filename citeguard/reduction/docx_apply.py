"""Conservative, transactional DOCX replacement for demonstrably safe spans."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

from ..bibliography import find_bibliography_start
from ..extractor import _iter_docx_blocks
from .apply import TextReplacement

_UNSAFE_TAGS = {
    qn("w:fldChar"),
    qn("w:instrText"),
    qn("w:bookmarkStart"),
    qn("w:bookmarkEnd"),
    qn("w:footnoteReference"),
    qn("w:endnoteReference"),
    qn("w:drawing"),
    qn("w:object"),
    qn("w:ins"),
    qn("w:del"),
    qn("w:hyperlink"),
}


@dataclass(frozen=True, slots=True)
class DocxAppliedReplacement:
    """Audit record for one formatting-preserving DOCX run edit."""

    passage_id: str
    paragraph_index: int
    original: str
    replacement: str
    run_index: int
    formatting_preserved: bool = True


class UnsafeDocxStructureError(ValueError):
    """The requested edit cannot be proven safe without XML flattening."""


def _content_paragraphs(document):
    return [p for p in _iter_docx_blocks(document) if p.text.strip()]


def write_revised_docx(
    source: Path,
    output: Path,
    replacements: list[TextReplacement],
) -> list[DocxAppliedReplacement]:
    """Apply unique single-run edits and atomically write a separate DOCX."""
    if source.resolve() == output.resolve():
        raise ValueError("DOCX output must differ from the source path.")
    if source.suffix.lower() != output.suffix.lower():
        raise ValueError("DOCX output format must match the source format.")
    if output.exists():
        raise ValueError("output already exists; choose a new output path")
    document = Document(source)
    paragraphs = _content_paragraphs(document)
    texts = [paragraph.text.strip() for paragraph in paragraphs]
    bibliography_start = find_bibliography_start(texts)
    limit = bibliography_start if bibliography_start is not None else len(paragraphs)
    applied: list[DocxAppliedReplacement] = []

    for replacement in replacements:
        matches = [
            (index, paragraph)
            for index, paragraph in enumerate(paragraphs[:limit])
            if replacement.original in paragraph.text
        ]
        if len(matches) != 1:
            raise UnsafeDocxStructureError(
                f"{replacement.passage_id}: target paragraph is missing or ambiguous"
            )
        paragraph_index, paragraph = matches[0]
        if any(element.tag in _UNSAFE_TAGS for element in paragraph._p.iter()):
            raise UnsafeDocxStructureError(
                f"{replacement.passage_id}: unsafe DOCX field or structure"
            )
        run_matches = [
            (index, run)
            for index, run in enumerate(paragraph.runs)
            if replacement.original in run.text
        ]
        if len(run_matches) != 1:
            raise UnsafeDocxStructureError(f"{replacement.passage_id}: replacement crosses runs")
        run_index, run = run_matches[0]
        if run.text.count(replacement.original) != 1:
            raise UnsafeDocxStructureError(
                f"{replacement.passage_id}: target text is ambiguous within run"
            )
        run.text = run.text.replace(replacement.original, replacement.replacement, 1)
        applied.append(
            DocxAppliedReplacement(
                passage_id=replacement.passage_id,
                paragraph_index=paragraph_index,
                original=replacement.original,
                replacement=replacement.replacement,
                run_index=run_index,
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}-", suffix=".docx", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        document.save(temporary)
        reopened = Document(temporary)
        revised_text = "\n".join(p.text for p in _content_paragraphs(reopened))
        if any(item.replacement not in revised_text for item in replacements):
            raise UnsafeDocxStructureError("saved DOCX failed replacement validation")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return applied
