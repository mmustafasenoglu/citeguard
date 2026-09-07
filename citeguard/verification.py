"""Bibliography verification against academic providers."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from .models import (
    BibliographyEntry,
    MetadataScores,
    ReferenceVerification,
    SourceCandidate,
    SourceType,
    VerificationStatus,
)
from .retrieval import RetrievalEngine, normalize_doi, normalize_title

VERIFIED_METADATA_THRESHOLD = 70
PARTIAL_METADATA_THRESHOLD = 50
_DEFAULT_MAX_AGE_YEARS = 25
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_PRIMARY_SOURCE_VENUES = {
    "nature",
    "science",
    "the lancet",
    "cell",
    "ieee",
    "acm",
    "springer",
    "elsevier",
    "wiley",
    "oxford",
    "cambridge",
    "nber",
    "pnas",
    "proceedings of the national academy",
    "journal of the american chemical society",
    "physical review",
}


def verify_bibliography(
    entries: list[BibliographyEntry],
    engine: RetrievalEngine,
    *,
    max_results: int = 5,
    recency_max_age: int = _DEFAULT_MAX_AGE_YEARS,
) -> list[ReferenceVerification]:
    results: list[ReferenceVerification] = []
    for index, entry in enumerate(entries):
        query = bibliography_query(entry)
        if not query:
            results.append(
                ReferenceVerification(
                    entry_index=index,
                    entry=entry,
                    query="",
                    status=VerificationStatus.UNRESOLVED,
                    candidate=None,
                    scores=None,
                    warnings=["The entry has insufficient metadata for provider search."],
                )
            )
            continue

        retrieval = engine.search_with_result(query, max_results=max_results)
        warnings = list(retrieval.warnings)
        if not retrieval.candidates and entry.doi:
            fallback_query = bibliography_query(entry, prefer_doi=False)
            if fallback_query:
                retrieval = engine.search_with_result(fallback_query, max_results=max_results)
                warnings.extend(retrieval.warnings)
                warnings.append(
                    "The DOI was not registered in Crossref; bibliographic search was used."
                )

        ranked = sorted(
            (
                (metadata_scores(entry, candidate), candidate)
                for candidate in retrieval.candidates
            ),
            key=lambda item: (-item[0].overall, normalize_title(item[1].title)),
        )
        if not ranked:
            status = (
                VerificationStatus.NOT_FOUND
                if not warnings
                else VerificationStatus.PROVIDER_ERROR
            )
            results.append(
                ReferenceVerification(
                    entry_index=index,
                    entry=entry,
                    query=query,
                    status=status,
                    candidate=None,
                    scores=None,
                    warnings=warnings,
                )
            )
            continue

        scores, candidate = ranked[0]
        status = _determine_status(entry, candidate, scores, ranked)
        recency = _check_recency(entry, candidate, recency_max_age)
        source_type = _classify_source_type(candidate)
        conflicts = _detect_provider_conflicts(ranked)
        results.append(
            ReferenceVerification(
                entry_index=index,
                entry=entry,
                query=query,
                status=status,
                candidate=candidate,
                scores=scores,
                warnings=warnings,
                recency_warning=recency,
                source_type=source_type,
                provider_conflicts=conflicts,
            )
        )
    return results


def _determine_status(
    entry: BibliographyEntry,
    candidate: SourceCandidate,
    scores: MetadataScores,
    ranked: list[tuple[MetadataScores, SourceCandidate]],
) -> VerificationStatus:
    if entry.doi and candidate.doi:
        if normalize_doi(entry.doi) == normalize_doi(candidate.doi):
            return VerificationStatus.VERIFIED
        if scores.doi == 0:
            return VerificationStatus.METADATA_MISMATCH
    if scores.overall >= VERIFIED_METADATA_THRESHOLD:
        return VerificationStatus.VERIFIED
    if scores.overall >= PARTIAL_METADATA_THRESHOLD:
        return VerificationStatus.PARTIALLY_VERIFIED
    if len(ranked) > 1:
        second_scores = ranked[1][0]
        if second_scores.overall >= VERIFIED_METADATA_THRESHOLD:
            return VerificationStatus.UNRESOLVED
    return VerificationStatus.UNRESOLVED


def _check_recency(
    entry: BibliographyEntry,
    candidate: SourceCandidate,
    max_age: int,
) -> str | None:
    from datetime import date

    year = candidate.year or entry.year
    if year is None:
        return None
    current_year = date.today().year
    age = current_year - year
    if age > max_age:
        return (
            f"Possibly outdated: source is {age} years old "
            f"(published {year}, threshold {max_age} years)."
        )
    return None


def _classify_source_type(candidate: SourceCandidate) -> SourceType:
    venue = (candidate.venue or "").lower()
    if not venue:
        return SourceType.UNKNOWN
    for pattern in _PRIMARY_SOURCE_VENUES:
        if pattern in venue:
            return SourceType.PRIMARY
    return SourceType.SECONDARY


def _detect_provider_conflicts(
    ranked: list[tuple[MetadataScores, SourceCandidate]],
) -> list[str]:
    if len(ranked) < 2:
        return []
    conflicts: list[str] = []
    top_scores, top_candidate = ranked[0]
    for second_scores, second_candidate in ranked[1:2]:
        titles_differ = normalize_title(top_candidate.title) != normalize_title(
            second_candidate.title
        )
        dois_differ = normalize_doi(top_candidate.doi) != normalize_doi(second_candidate.doi)
        if (
            top_scores.overall >= VERIFIED_METADATA_THRESHOLD
            and second_scores.overall >= VERIFIED_METADATA_THRESHOLD
            and titles_differ
            and dois_differ
        ):
            conflicts.append(
                f"Provider conflict: '{top_candidate.title}' ({top_candidate.source_api}) "
                f"vs '{second_candidate.title}' ({second_candidate.source_api})"
            )
    return conflicts


def bibliography_query(entry: BibliographyEntry, *, prefer_doi: bool = True) -> str:
    if entry.doi and prefer_doi:
        return entry.doi
    parts = [entry.title or "", _first_author_for_query(entry.authors), str(entry.year or "")]
    return " ".join(part for part in parts if part).strip()


def metadata_scores(entry: BibliographyEntry, candidate: SourceCandidate) -> MetadataScores:
    title_score = _text_similarity(entry.title, candidate.title) if entry.title else None
    author_score = _author_similarity(entry.authors, candidate.authors)
    year_score = _year_score(entry, candidate)
    entry_doi = normalize_doi(entry.doi)
    candidate_doi = normalize_doi(candidate.doi)
    doi_score = None
    if entry_doi and candidate_doi:
        doi_score = 100 if candidate_doi == entry_doi else 0

    weighted_fields = [
        (title_score, 40),
        (author_score, 35),
        (year_score, 25),
    ]
    if doi_score is not None:
        weighted_fields = [
            (doi_score, 60),
            (title_score, 20),
            (author_score, 15),
            (year_score, 5),
        ]
    available = [(score, weight) for score, weight in weighted_fields if score is not None]
    if not available:
        return MetadataScores(
            author=author_score,
            year=year_score,
            title=title_score,
            doi=doi_score,
            overall=0,
        )
    weighted_total = sum(score * weight for score, weight in available)
    overall = round(weighted_total / sum(weight for _, weight in available))
    if doi_score == 0:
        overall = min(overall, 49)
    return MetadataScores(
        author=author_score,
        year=year_score,
        title=title_score,
        doi=doi_score,
        overall=overall,
    )


def _text_similarity(first: str | None, second: str | None) -> int | None:
    if not first or not second:
        return None
    left = normalize_title(first)
    right = normalize_title(second)
    if not left or not right:
        return None
    left_words = set(left.split())
    right_words = set(right.split())
    overlap = len(left_words & right_words) / max(len(left_words | right_words), 1)
    sequence = SequenceMatcher(None, left, right).ratio()
    return round((overlap * 0.55 + sequence * 0.45) * 100)


def _author_similarity(entry_authors: str | None, candidate_authors: list[str]) -> int | None:
    if not entry_authors or not candidate_authors:
        return None
    entry_first = _first_author_for_query(entry_authors)
    candidate_first = candidate_authors[0]
    entry_surname = _normalized_name(entry_first).split()
    candidate_parts = _normalized_name(candidate_first).split()
    if not entry_surname or not candidate_parts:
        return None
    if entry_surname[0] in candidate_parts:
        return 100
    return _text_similarity(entry_first, candidate_first)


def _year_score(entry: BibliographyEntry, candidate: SourceCandidate) -> int | None:
    if entry.no_date or entry.year is None or candidate.year is None:
        return None
    return 100 if entry.year == candidate.year else 0


def _first_author_for_query(authors: str | None) -> str:
    if not authors:
        return ""
    first = re.split(r"\s+(?:&|and|ve)\s+|;", authors, maxsplit=1, flags=re.IGNORECASE)[0]
    if "," in first:
        return first.split(",", maxsplit=1)[0].strip()
    return first.strip()


def _normalized_name(value: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(_TOKEN_RE.findall(ascii_text.lower()))
