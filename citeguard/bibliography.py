"""Bibliography section detection, entry parsing, and consistency checks."""

from __future__ import annotations

import re
from collections import Counter

from .extractor import DOI_RE
from .models import (
    BibliographyEntry,
    BibliographyIssue,
    BibliographyIssueKind,
    ExistingCitation,
)
from .retrieval import normalize_doi

MAX_BIBLIOGRAPHY_NUMBER = 999


def resolve_numbered_citations(
    citations: list[ExistingCitation],
    bibliography: list[BibliographyEntry],
) -> dict[int, BibliographyEntry | None]:
    bib_by_number: dict[int, BibliographyEntry] = {}
    for entry in bibliography:
        if entry.numbered_ref is not None:
            bib_by_number[entry.numbered_ref] = entry
    seen_numbers: set[int] = set()
    resolved: dict[int, BibliographyEntry | None] = {}
    for citation in citations:
        if citation.numbered_ref is None:
            continue
        num = citation.numbered_ref
        if num in seen_numbers:
            continue
        seen_numbers.add(num)
        if num < 1 or num > MAX_BIBLIOGRAPHY_NUMBER:
            resolved[num] = None
        else:
            resolved[num] = bib_by_number.get(num)
    return resolved

BIBLIOGRAPHY_HEADINGS = {
    "references",
    "bibliography",
    "works cited",
    "kaynaklar",
    "kaynakça",
}
YEAR_RE = re.compile(r"\b(18|19|20|21)\d{2}[a-z]?\b", re.IGNORECASE)
NO_DATE_RE = re.compile(r"\((?:t\.?\s*y\.?|n\.?\s*d\.?)\)", re.IGNORECASE)
NUMBER_PREFIX_RE = re.compile(r"^\s*(?:\[(?P<bracket>\d+)\]|(?P<plain>\d+)[.)])\s*")


def _heading_key(text: str) -> str:
    stripped = text.strip().lstrip("#").strip().rstrip(":").lower()
    stripped = re.sub(r"^ek\s*[-.]?\s*\d+\s*:?\s*", "", stripped)
    return re.sub(r"\s+", " ", stripped)


def find_bibliography_start(paragraphs: list[str]) -> int | None:
    for index, paragraph in enumerate(paragraphs):
        if _heading_key(paragraph) in BIBLIOGRAPHY_HEADINGS:
            return index
    # Structured forms often use localized or application-specific appendix labels.
    # A short heading followed by several reference-shaped paragraphs is a safe fallback.
    for index, paragraph in enumerate(paragraphs[:-3]):
        if (
            len(paragraph) > 80
            or len(paragraph.split()) > 8
            or paragraph != paragraph.upper()
        ):
            continue
        following = paragraphs[index + 1 : index + 6]
        reference_count = sum(
            bool(YEAR_RE.search(value) or NO_DATE_RE.search(value) or DOI_RE.search(value))
            for value in following
        )
        if reference_count >= 3:
            return index
    return None


def parse_bibliography(paragraphs: list[str]) -> tuple[int | None, list[BibliographyEntry]]:
    start = find_bibliography_start(paragraphs)
    if start is None:
        return None, []

    entries: list[BibliographyEntry] = []
    for paragraph in paragraphs[start + 1 :]:
        raw = paragraph.strip()
        if not raw:
            continue
        if raw.startswith("#"):
            break
        entries.append(parse_bibliography_entry(raw))
    return start, entries


def parse_bibliography_entry(raw: str) -> BibliographyEntry:
    number_match = NUMBER_PREFIX_RE.match(raw)
    numbered_ref = None
    body = raw
    if number_match:
        numbered_ref = int(number_match.group("bracket") or number_match.group("plain"))
        body = raw[number_match.end() :].strip()

    doi_match = DOI_RE.search(body)
    doi = normalize_doi(doi_match.group(0)) if doi_match else None

    year_match = YEAR_RE.search(body)
    year = int(year_match.group(0)[:4]) if year_match else None
    no_date_match = NO_DATE_RE.search(body)
    no_date = no_date_match is not None and year_match is None

    authors: str | None = None
    title: str | None = None
    if year_match:
        before_year = body[: year_match.start()].strip(" .,(;")
        after_year = body[year_match.end() :].strip(" .);,")
        authors = before_year or None
        title = _extract_probable_title(after_year)
    elif no_date_match:
        before_date = body[: no_date_match.start()].strip(" .,(;")
        after_date = body[no_date_match.end() :].strip(" .);,")
        authors = before_date or None
        title = _extract_probable_title(after_date)
    else:
        # Best effort only; bibliography formats vary substantially.
        title = _extract_probable_title(body)

    return BibliographyEntry(
        raw_text=raw,
        authors=authors,
        year=year,
        title=title,
        doi=doi,
        numbered_ref=numbered_ref,
        no_date=no_date,
    )


def _extract_probable_title(text: str) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"https?://(?:dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    cleaned = DOI_RE.sub("", cleaned).strip(" .")
    if not cleaned:
        return None
    # Prefer the first sentence-like segment after year. It is deliberately conservative.
    first = re.split(
        r"\.\s+(?=(?:[A-Z]|\d{4}\b|arXiv:|https?://))",
        cleaned,
        maxsplit=1,
    )[0].strip(" .")
    return first or None


def bibliography_issues(
    citations: list[ExistingCitation], entries: list[BibliographyEntry]
) -> list[BibliographyIssue]:
    issues: list[BibliographyIssue] = []

    dois = [entry.doi for entry in entries if entry.doi]
    for doi, count in Counter(dois).items():
        if count > 1:
            issues.append(
                BibliographyIssue(
                    kind=BibliographyIssueKind.DUPLICATE_DOI,
                    detail=f"DOI {doi} appears {count} times in the bibliography.",
                )
            )

    # Author-date comparison is intentionally fuzzy in v0.1.
    bibliography_keys = {
        (_normalize_author_key(entry.authors), entry.year)
        for entry in entries
        if entry.authors and (entry.year is not None or entry.no_date)
    }
    citations_by_key: dict[tuple[str, int | None], list[ExistingCitation]] = {}
    for citation in citations:
        if not citation.authors or (citation.year is None and not citation.no_date):
            continue
        key = (_normalize_author_key(citation.authors), citation.year)
        citations_by_key.setdefault(key, []).append(citation)
    citation_keys = set(citations_by_key)

    for key in sorted(citation_keys - bibliography_keys, key=_author_date_sort_key):
        first_citation = citations_by_key[key][0]
        issues.append(
            BibliographyIssue(
                kind=BibliographyIssueKind.CITED_NOT_LISTED,
                detail=(
                    f"Citation {key[0]} ({_format_year(key[1])}) was found in text but not in the "
                    "bibliography."
                ),
                paragraph_index=first_citation.paragraph_index,
            )
        )

    for key in sorted(bibliography_keys - citation_keys, key=_author_date_sort_key):
        issues.append(
            BibliographyIssue(
                kind=BibliographyIssueKind.LISTED_NOT_CITED,
                detail=(
                    f"Bibliography entry {key[0]} ({_format_year(key[1])}) was not found in text."
                ),
            )
        )

    return issues


def citation_matches_entry(citation: ExistingCitation, entry: BibliographyEntry) -> bool:
    if citation.numbered_ref is not None and entry.numbered_ref is not None:
        return citation.numbered_ref == entry.numbered_ref
    if citation.doi and entry.doi:
        return citation.doi == entry.doi
    if not citation.authors or not entry.authors:
        return False
    if citation.year is None and not citation.no_date:
        return False
    if entry.year is None and not entry.no_date:
        return False
    return (
        _normalize_author_key(citation.authors) == _normalize_author_key(entry.authors)
        and citation.year == entry.year
    )


def _format_year(year: int | None) -> str:
    return str(year) if year is not None else "n.d."


def _author_date_sort_key(value: tuple[str, int | None]) -> tuple[str, int]:
    return value[0], -1 if value[1] is None else value[1]


def _normalize_author_key(authors: str | None) -> str:
    if not authors:
        return ""
    value = authors.lower().replace("&", "and")
    value = re.sub(r"\bet\s+al\.?\b", "", value)
    value = re.sub(r"[^a-zà-öø-ÿ]+", " ", value)
    tokens = [token for token in value.split() if token not in {"and"}]
    # First surname is the most stable cross-format key for v0.1.
    return tokens[0] if tokens else ""
