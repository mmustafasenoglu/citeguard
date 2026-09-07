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
_PARTIAL_METADATA_THRESHOLD = 50
_DEFAULT_MAX_AGE_YEARS = 25
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_PRIMARY_WORK_TYPES = frozenset({
    "research-article",
    "original-research",
    "clinical-trial",
    "randomized-controlled-trial",
    "research-output",
})
_SECONDARY_WORK_TYPES = frozenset({
    "review",
    "editorial",
    "letter",
    "commentary",
    "erratum",
    "retraction",
    "book-review",
    "supplementary-materials",
    "meta-analysis",
    "systematic-review",
})


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
        has_provider_errors = bool(retrieval.provider_errors)
        if not retrieval.candidates and entry.doi:
            fallback_query = bibliography_query(entry, prefer_doi=False)
            if fallback_query:
                retrieval = engine.search_with_result(fallback_query, max_results=max_results)
                warnings.extend(retrieval.warnings)
                has_provider_errors = has_provider_errors or bool(retrieval.provider_errors)
                warnings.append(
                    "The DOI lookup returned no match; "
                    "bibliographic metadata search was used."
                )

        ranked = sorted(
            (
                (metadata_scores(entry, candidate), candidate)
                for candidate in retrieval.candidates
            ),
            key=lambda item: (-item[0].overall, normalize_title(item[1].title)),
        )
        if not ranked:
            if has_provider_errors:
                status = VerificationStatus.PROVIDER_ERROR
            else:
                status = VerificationStatus.NOT_FOUND
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
    if scores.overall >= _PARTIAL_METADATA_THRESHOLD:
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
    work_type = (candidate.work_type or "").lower().strip()
    if not work_type:
        return SourceType.UNKNOWN
    if work_type in _SECONDARY_WORK_TYPES:
        return SourceType.SECONDARY
    if work_type in _PRIMARY_WORK_TYPES:
        return SourceType.PRIMARY
    return SourceType.UNKNOWN


def _detect_provider_conflicts(
    ranked: list[tuple[MetadataScores, SourceCandidate]],
) -> list[str]:
    conflicts: list[str] = []
    for _scores, candidate in ranked:
        records = candidate.provider_records
        if len(records) < 2:
            continue
        providers = list(records.keys())
        for i in range(len(providers)):
            for j in range(i + 1, len(providers)):
                rec_a = records[providers[i]]
                rec_b = records[providers[j]]
                norm_title_a = normalize_title(str(rec_a.get("title", "")))
                norm_title_b = normalize_title(str(rec_b.get("title", "")))
                year_a = rec_a.get("year")
                year_b = rec_b.get("year")
                doi_a = rec_a.get("doi")
                doi_b = rec_b.get("doi")
                title_conflict = bool(
                    norm_title_a and norm_title_b
                    and norm_title_a != norm_title_b
                )
                year_conflict = bool(
                    year_a is not None and year_b is not None
                    and year_a != year_b
                )
                doi_both = normalize_doi(str(doi_a)) if doi_a else None
                doi_other = normalize_doi(str(doi_b)) if doi_b else None
                if doi_both and doi_other and doi_both == doi_other and title_conflict:
                    conflicts.append(
                        f"Metadata conflict for DOI {doi_both}: "
                        f"'{rec_a.get('title', '?')}' ({providers[i]}) "
                        f"vs '{rec_b.get('title', '?')}' ({providers[j]})"
                    )
                elif doi_both and doi_other and doi_both == doi_other and year_conflict:
                    conflicts.append(
                        f"Year conflict for DOI {doi_both}: "
                        f"{year_a} ({providers[i]}) vs {year_b} ({providers[j]})"
                    )
    if not conflicts:
        top_candidate = ranked[0][1] if ranked else None
        if top_candidate:
            for _second_scores, second_candidate in ranked[1:]:
                if _is_same_work_different_doi(top_candidate, second_candidate):
                    conflicts.append(
                        f"DOI conflict: providers describe the same work "
                        f"with different DOIs: '{top_candidate.title}' "
                        f"({top_candidate.source_api}, "
                        f"{normalize_doi(top_candidate.doi)}) vs "
                        f"'{second_candidate.title}' "
                        f"({second_candidate.source_api}, "
                        f"{normalize_doi(second_candidate.doi)})."
                    )
    return conflicts


def _is_same_work_different_doi(
    first: SourceCandidate, second: SourceCandidate
) -> bool:
    """Return True when two candidates carry different real DOIs but their
    deterministic metadata strongly suggests the same bibliographic work.

    A missing DOI is never a conflicting DOI: both sides must carry a valid
    normalized DOI identity that disagrees.
    """
    doi_a = normalize_doi(first.doi)
    doi_b = normalize_doi(second.doi)
    if not doi_a or not doi_b or doi_a == doi_b:
        return False
    title_similarity = _text_similarity(first.title, second.title)
    if title_similarity is None or title_similarity < VERIFIED_METADATA_THRESHOLD:
        return False
    if first.year is not None and second.year is not None and first.year != second.year:
        return False
    if first.authors and second.authors:
        author_score = _author_similarity(", ".join(first.authors), second.authors)
        if author_score is None or author_score < _PARTIAL_METADATA_THRESHOLD:
            return False
    return True


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
