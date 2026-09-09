"""Safe, reversible text patch application for Markdown and plain text."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TextReplacement:
    """One exact replacement guarded by the original text."""

    original: str
    replacement: str
    passage_id: str


@dataclass(frozen=True, slots=True)
class AppliedReplacement:
    """Audit record for one applied replacement."""

    passage_id: str
    original: str
    replacement: str
    start_offset: int
    end_offset: int


def apply_text_replacements(
    document: str,
    replacements: list[TextReplacement],
) -> tuple[str, list[AppliedReplacement]]:
    """Apply unique exact replacements without touching ambiguous passages."""
    ordered = sorted(replacements, key=lambda item: document.find(item.original))
    output = document
    applied: list[AppliedReplacement] = []
    cursor = 0
    for item in ordered:
        if not item.original:
            raise ValueError(f"Replacement {item.passage_id} has empty original text.")
        start = output.find(item.original, cursor)
        if start < 0:
            raise ValueError(f"Original passage for {item.passage_id} was not found unchanged.")
        if output.find(item.original, start + 1) >= 0:
            raise ValueError(
                f"Original passage for {item.passage_id} is ambiguous; manual review is required."
            )
        end = start + len(item.original)
        output = output[:start] + item.replacement + output[end:]
        applied.append(
            AppliedReplacement(
                passage_id=item.passage_id,
                original=item.original,
                replacement=item.replacement,
                start_offset=start,
                end_offset=end,
            )
        )
        cursor = start + len(item.replacement)
    return output, applied


def write_revised_text(
    source: Path,
    output: Path,
    replacements: list[TextReplacement],
) -> list[AppliedReplacement]:
    """Write a revised Markdown/TXT document without replacing the source."""
    if source.suffix.lower() not in {".md", ".txt"}:
        raise ValueError("Automatic application supports only .md and .txt files.")
    if source.resolve() == output.resolve():
        raise ValueError("Text output must differ from the source path.")
    if source.suffix.lower() != output.suffix.lower():
        raise ValueError("Text output format must match the source format.")
    if output.exists():
        raise ValueError("output already exists; choose a new output path")
    original = source.read_text(encoding="utf-8")
    revised, applied = apply_text_replacements(original, replacements)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}-", suffix=output.suffix, dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(revised)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return applied


def restore_text(
    revised: str,
    applied: list[AppliedReplacement],
) -> str:
    """Restore replacements only when their recorded locations remain intact.

    ``AppliedReplacement.start_offset`` is relative to the document state just
    after that replacement was made.  Reversing the audit trail restores that
    same state for each record.  Do not fall back to a global text search:
    after a user edit or when replacement text is duplicated, that could
    silently restore an unrelated passage.
    """
    output = revised
    for item in reversed(applied):
        start = item.start_offset
        end = start + len(item.replacement)
        if start < 0 or output[start:end] != item.replacement:
            raise ValueError(
                f"Replacement for {item.passage_id} is missing or has changed; "
                "refusing an ambiguous restore."
            )
        output = output[:start] + item.original + output[end:]
    return output
