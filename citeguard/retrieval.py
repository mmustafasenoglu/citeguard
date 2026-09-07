"""Provider orchestration, candidate normalization, deduplication, and ranking."""

from __future__ import annotations

import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, fields, replace
from difflib import SequenceMatcher
from typing import Any

from .cache import FileCache
from .config import DEFAULT_MAX_RESULTS, STRONG_MATCH_THRESHOLD
from .models import SourceCandidate
from .providers.arxiv import ArxivProvider
from .providers.base import (
    ProviderResponseError,
    SourceProvider,
    candidate_from_dict,
    candidate_to_dict,
)
from .providers.crossref import CrossrefProvider
from .providers.openalex import OpenAlexProvider
from .providers.semantic_scholar import SemanticScholarProvider

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)
_ARXIV_RE = re.compile(
    r"(?:arxiv:\s*|arxiv\.org/(?:abs|pdf)/)?"
    r"((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z-]+)?/\d{7})(?:v\d+)?)",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9]+")
_TRAILING_DOI_PUNCTUATION = ".,;:)]}>"
_PROVIDER_PRIORITY = {"semantic_scholar": 0, "crossref": 1, "openalex": 2, "arxiv": 3}


class _OfflineSkipped(Exception):
    """Internal signal: offline mode with no cached result for a provider.

    This is a policy skip, not a provider failure: the provider was never
    called and no error diagnostic should be recorded for it.
    """


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    candidates: list[SourceCandidate]
    warnings: list[str]
    queried_providers: list[str]
    early_stopped: bool = False
    provider_errors: list[str] = field(default_factory=list)
    offline: bool = False
    skipped_providers: list[str] = field(default_factory=list)
    cache_hits: list[str] = field(default_factory=list)


class _RateLimiter:
    """Simple per-provider token-bucket rate limiter."""

    def __init__(self, min_interval: float = 1.0) -> None:
        self._min_interval = min_interval
        self._last_call: dict[str, float] = {}

    def wait(self, provider_name: str) -> None:
        now = time.monotonic()
        last = self._last_call.get(provider_name, 0.0)
        elapsed = now - last
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call[provider_name] = time.monotonic()


class RetrievalEngine:
    """Search academic providers and produce a deterministic candidate list."""

    def __init__(
        self,
        providers: list[SourceProvider] | None = None,
        *,
        cache: FileCache | None = None,
        use_cache: bool = True,
        parallel: bool = True,
        rate_limit: bool = True,
        offline: bool = False,
    ) -> None:
        self.providers = providers if providers is not None else [
            SemanticScholarProvider(),
            CrossrefProvider(),
            OpenAlexProvider(),
            ArxivProvider(),
        ]
        self.cache = cache
        self.use_cache = use_cache
        self.parallel = parallel
        self._rate_limiter = _RateLimiter() if rate_limit else None
        self.offline = offline
        self.last_result: RetrievalResult | None = None
        self.remote_calls: list[str] = []
        self._call_skipped: list[str] = []
        self._call_cache_hits: list[str] = []

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[SourceCandidate]:
        """Return ranked candidates; provider failures are exposed via ``last_result``."""
        self.last_result = self.search_with_result(query, max_results=max_results)
        return self.last_result.candidates

    def search_with_result(
        self, query: str, max_results: int = DEFAULT_MAX_RESULTS
    ) -> RetrievalResult:
        """Return academic results and diagnostics for a future fallback decision layer.

        When *parallel* is True and multiple providers are configured, providers
        are queried concurrently via a thread pool. The first provider is always
        checked for a strong early-stop match before the remaining providers are
        dispatched.
        """
        query = " ".join(query.split())
        if not query or max_results <= 0:
            return RetrievalResult([], [], [], offline=self.offline)

        gathered: list[SourceCandidate] = []
        warnings: list[str] = []
        queried: list[str] = []
        provider_errors: list[str] = []
        early_stopped = False
        self._call_skipped: list[str] = []
        self._call_cache_hits: list[str] = []

        if self.parallel and len(self.providers) > 1:
            gathered, warnings, queried, early_stopped, provider_errors = (
                self._search_parallel(query, max_results)
            )
        else:
            gathered, warnings, queried, early_stopped, provider_errors = (
                self._search_sequential(query, max_results)
            )

        candidates = rank_candidates(query, deduplicate_candidates(gathered))[:max_results]
        return RetrievalResult(
            candidates,
            warnings,
            queried,
            early_stopped,
            provider_errors,
            offline=self.offline,
            skipped_providers=sorted(set(self._call_skipped)),
            cache_hits=sorted(set(self._call_cache_hits)),
        )

    def _search_sequential(
        self, query: str, max_results: int
    ) -> tuple[list[SourceCandidate], list[str], list[str], bool, list[str]]:
        gathered: list[SourceCandidate] = []
        warnings: list[str] = []
        queried: list[str] = []
        provider_errors: list[str] = []
        for index, provider in enumerate(self.providers):
            queried.append(provider.name)
            if self._rate_limiter:
                self._rate_limiter.wait(provider.name)
            try:
                candidates = self._provider_search(provider, query, max_results)
            except _OfflineSkipped:
                continue
            except Exception as exc:
                provider_errors.append(f"{provider.name}: {exc}")
                continue
            gathered.extend(candidates)
            if index == 0 and any(is_strong_match(query, item) for item in candidates):
                return gathered, warnings, queried, True, provider_errors
        return gathered, warnings, queried, False, provider_errors

    def _search_parallel(
        self, query: str, max_results: int
    ) -> tuple[list[SourceCandidate], list[str], list[str], bool, list[str]]:
        first_provider = self.providers[0]
        remaining = self.providers[1:]

        if self._rate_limiter:
            self._rate_limiter.wait(first_provider.name)
        try:
            first_candidates = self._provider_search(
                first_provider, query, max_results
            )
        except _OfflineSkipped:
            first_candidates = []
            provider_errors_first = []
        except Exception as exc:
            first_candidates = []
            provider_errors_first = [f"{first_provider.name}: {exc}"]
        else:
            provider_errors_first = []

        if any(is_strong_match(query, item) for item in first_candidates):
            return first_candidates, [], [first_provider.name], True, provider_errors_first

        remaining_queried = [p.name for p in remaining]
        remaining_gathered: list[SourceCandidate] = []
        remaining_provider_errors: list[str] = []

        if remaining:
            with ThreadPoolExecutor(max_workers=len(remaining)) as executor:
                future_to_provider = {
                    executor.submit(
                        self._safe_provider_search, p, query, max_results
                    ): p
                    for p in remaining
                }
                for future in as_completed(future_to_provider):
                    provider = future_to_provider[future]
                    result = future.result()
                    if result is None:
                        pass
                    elif isinstance(result, Exception):
                        remaining_provider_errors.append(f"{provider.name}: {result}")
                    else:
                        remaining_gathered.extend(result)

        all_gathered = first_candidates + remaining_gathered
        all_queried = [first_provider.name] + remaining_queried
        all_errors = provider_errors_first + remaining_provider_errors
        return all_gathered, [], all_queried, False, all_errors

    def _safe_provider_search(
        self, provider: SourceProvider, query: str, max_results: int
    ) -> list[SourceCandidate] | Exception:
        if self._rate_limiter:
            self._rate_limiter.wait(provider.name)
        try:
            return self._provider_search(provider, query, max_results)
        except _OfflineSkipped:
            return []
        except Exception as exc:
            return exc

    def _provider_search(
        self, provider: SourceProvider, query: str, max_results: int
    ) -> list[SourceCandidate]:
        key = None
        if self.cache is not None and self.use_cache:
            key = self.cache.make_key(
                operation="provider_search",
                provider=provider.name,
                query=f"{query}\nmax_results={max_results}",
            )
            cached = self.cache.get(key)
            if cached is not None:
                try:
                    if not isinstance(cached, list):
                        raise ProviderResponseError("Cached provider result is malformed.")
                    result = [
                        normalize_candidate(candidate_from_dict(item))
                        for item in cached
                    ]
                except ProviderResponseError:
                    pass
                else:
                    self._call_cache_hits.append(provider.name)
                    return result

        if self.offline:
            self._call_skipped.append(provider.name)
            raise _OfflineSkipped(provider.name)

        candidates = [normalize_candidate(item) for item in provider.search(query, max_results)]
        self.remote_calls.append(provider.name)
        if self.cache is not None and self.use_cache and key is not None:
            self.cache.set(key, [candidate_to_dict(item) for item in candidates])
        return candidates


def retrieve_sources(
    query: str,
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
    providers: list[SourceProvider] | None = None,
    cache: FileCache | None = None,
    use_cache: bool = True,
) -> list[SourceCandidate]:
    """Convenience entry point for higher-level claim and citation workflows."""
    return RetrievalEngine(providers, cache=cache, use_cache=use_cache).search(
        query, max_results=max_results
    )


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    match = _DOI_RE.search(value.strip())
    return match.group(0).rstrip(_TRAILING_DOI_PUNCTUATION).lower() if match else None


def normalize_arxiv_id(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().removesuffix(".pdf")
    match = _ARXIV_RE.search(cleaned)
    if not match:
        return None
    return re.sub(r"v\d+$", "", match.group(1), flags=re.IGNORECASE).lower()


def normalize_title(value: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(_WORD_RE.findall(ascii_text.lower()))


def normalize_candidate(candidate: SourceCandidate) -> SourceCandidate:
    return replace(
        candidate,
        title=" ".join(candidate.title.split()),
        authors=[" ".join(author.split()) for author in candidate.authors if author.strip()],
        doi=normalize_doi(candidate.doi),
        arxiv_id=normalize_arxiv_id(candidate.arxiv_id),
    )


def candidate_identity(candidate: SourceCandidate) -> tuple[Any, ...]:
    doi = normalize_doi(candidate.doi)
    if doi:
        return ("doi", doi)
    arxiv_id = normalize_arxiv_id(candidate.arxiv_id)
    if arxiv_id:
        return ("arxiv", arxiv_id)
    return ("title_year", normalize_title(candidate.title), candidate.year)


def deduplicate_candidates(candidates: list[SourceCandidate]) -> list[SourceCandidate]:
    deduplicated: list[SourceCandidate] = []
    identities: dict[tuple[Any, ...], int] = {}
    for raw_candidate in candidates:
        candidate = normalize_candidate(raw_candidate)
        identity = candidate_identity(candidate)
        existing_index = identities.get(identity)
        if existing_index is None:
            existing_index = len(deduplicated)
            deduplicated.append(candidate)
        else:
            deduplicated[existing_index] = _merge_candidates(
                deduplicated[existing_index], candidate
            )
        identities[identity] = existing_index
    return deduplicated


def candidate_relevance(query: str, candidate: SourceCandidate) -> int:
    query_doi = normalize_doi(query)
    if query_doi and query_doi == normalize_doi(candidate.doi):
        return 100
    query_arxiv = normalize_arxiv_id(query)
    if query_arxiv and query_arxiv == normalize_arxiv_id(candidate.arxiv_id):
        return 100
    normalized_query = normalize_title(query)
    normalized_candidate = normalize_title(candidate.title)
    if not normalized_query or not normalized_candidate:
        return 0
    query_words = set(normalized_query.split())
    candidate_words = set(normalized_candidate.split())
    overlap = len(query_words & candidate_words) / max(len(query_words), 1)
    sequence = SequenceMatcher(None, normalized_query, normalized_candidate).ratio()
    return round((overlap * 0.65 + sequence * 0.35) * 100)


def is_strong_match(query: str, candidate: SourceCandidate) -> bool:
    query_doi = normalize_doi(query)
    if query_doi and query_doi == normalize_doi(candidate.doi):
        return True
    query_arxiv = normalize_arxiv_id(query)
    if query_arxiv and query_arxiv == normalize_arxiv_id(candidate.arxiv_id):
        return True
    return (
        normalize_title(query) == normalize_title(candidate.title)
        and candidate_relevance(query, candidate) >= STRONG_MATCH_THRESHOLD
    )


def rank_candidates(query: str, candidates: list[SourceCandidate]) -> list[SourceCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate_relevance(query, candidate),
            -(candidate.year or 0),
            normalize_title(candidate.title),
            normalize_doi(candidate.doi) or "",
            _PROVIDER_PRIORITY.get(candidate.source_api, len(_PROVIDER_PRIORITY)),
        ),
    )


def _merge_candidates(first: SourceCandidate, second: SourceCandidate) -> SourceCandidate:
    values: dict[str, Any] = {}
    for fld in fields(SourceCandidate):
        if fld.name in ("source_apis", "provider_records"):
            continue
        first_value = getattr(first, fld.name)
        second_value = getattr(second, fld.name)
        values[fld.name] = first_value or second_value
    if len(second.abstract or "") > len(first.abstract or ""):
        values["abstract"] = second.abstract
    if len(second.authors) > len(first.authors):
        values["authors"] = second.authors
    merged_apis: list[str] = list(first.source_apis) if first.source_apis else [first.source_api]
    second_api = second.source_api
    if second_api and second_api not in merged_apis:
        merged_apis.append(second_api)
    values["source_apis"] = merged_apis
    values["source_api"] = first.source_api
    merged_records: dict[str, dict[str, object]] = dict(first.provider_records)
    merged_records[second.source_api] = {
        "title": second.title,
        "authors": list(second.authors),
        "year": second.year,
        "doi": second.doi,
        "work_type": second.work_type,
        "venue": second.venue,
    }
    if first.source_api not in merged_records:
        merged_records[first.source_api] = {
            "title": first.title,
            "authors": list(first.authors),
            "year": first.year,
            "doi": first.doi,
            "work_type": first.work_type,
            "venue": first.venue,
        }
    values["provider_records"] = merged_records
    return SourceCandidate(**values)
