"""Provider orchestration, candidate normalization, deduplication, and ranking."""

from __future__ import annotations

import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields, replace
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
from .providers.semantic_scholar import SemanticScholarProvider

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)
_ARXIV_RE = re.compile(
    r"(?:arxiv:\s*|arxiv\.org/(?:abs|pdf)/)?"
    r"((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z-]+)?/\d{7})(?:v\d+)?)",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9]+")
_TRAILING_DOI_PUNCTUATION = ".,;:)]}>"
_PROVIDER_PRIORITY = {"semantic_scholar": 0, "crossref": 1, "arxiv": 2}


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    candidates: list[SourceCandidate]
    warnings: list[str]
    queried_providers: list[str]
    early_stopped: bool = False


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
    ) -> None:
        self.providers = providers or [
            SemanticScholarProvider(),
            CrossrefProvider(),
            ArxivProvider(),
        ]
        self.cache = cache
        self.use_cache = use_cache
        self.parallel = parallel
        self._rate_limiter = _RateLimiter() if rate_limit else None
        self.last_result: RetrievalResult | None = None

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
            return RetrievalResult([], [], [])

        gathered: list[SourceCandidate] = []
        warnings: list[str] = []
        queried: list[str] = []
        early_stopped = False

        if self.parallel and len(self.providers) > 1:
            gathered, warnings, queried, early_stopped = self._search_parallel(
                query, max_results
            )
        else:
            gathered, warnings, queried, early_stopped = self._search_sequential(
                query, max_results
            )

        candidates = rank_candidates(query, deduplicate_candidates(gathered))[:max_results]
        return RetrievalResult(candidates, warnings, queried, early_stopped)

    def _search_sequential(
        self, query: str, max_results: int
    ) -> tuple[list[SourceCandidate], list[str], list[str], bool]:
        gathered: list[SourceCandidate] = []
        warnings: list[str] = []
        queried: list[str] = []
        for index, provider in enumerate(self.providers):
            queried.append(provider.name)
            if self._rate_limiter:
                self._rate_limiter.wait(provider.name)
            try:
                candidates = self._provider_search(provider, query, max_results)
            except Exception as exc:
                warnings.append(f"{provider.name}: {exc}")
                continue
            gathered.extend(candidates)
            if index == 0 and any(is_strong_match(query, item) for item in candidates):
                return gathered, warnings, queried, True
        return gathered, warnings, queried, False

    def _search_parallel(
        self, query: str, max_results: int
    ) -> tuple[list[SourceCandidate], list[str], list[str], bool]:
        first_provider = self.providers[0]
        remaining = self.providers[1:]

        if self._rate_limiter:
            self._rate_limiter.wait(first_provider.name)
        try:
            first_candidates = self._provider_search(
                first_provider, query, max_results
            )
        except Exception as exc:
            first_candidates = []
            warnings_first = [f"{first_provider.name}: {exc}"]
        else:
            warnings_first = []

        if any(is_strong_match(query, item) for item in first_candidates):
            return first_candidates, warnings_first, [first_provider.name], True

        remaining_queried = [p.name for p in remaining]
        remaining_gathered: list[SourceCandidate] = []
        remaining_warnings: list[str] = []

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
                        remaining_warnings.append(f"{provider.name}: {result}")
                    else:
                        remaining_gathered.extend(result)

        all_gathered = first_candidates + remaining_gathered
        all_warnings = warnings_first + remaining_warnings
        all_queried = [first_provider.name] + remaining_queried
        return all_gathered, all_warnings, all_queried, False

    def _safe_provider_search(
        self, provider: SourceProvider, query: str, max_results: int
    ) -> list[SourceCandidate] | Exception:
        if self._rate_limiter:
            self._rate_limiter.wait(provider.name)
        try:
            return self._provider_search(provider, query, max_results)
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
                    return [normalize_candidate(candidate_from_dict(item)) for item in cached]
                except ProviderResponseError:
                    pass

        candidates = [normalize_candidate(item) for item in provider.search(query, max_results)]
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
            _PROVIDER_PRIORITY.get(candidate.source_api, len(_PROVIDER_PRIORITY)),
            -(candidate.year or 0),
            normalize_title(candidate.title),
            normalize_doi(candidate.doi) or "",
        ),
    )


def _merge_candidates(first: SourceCandidate, second: SourceCandidate) -> SourceCandidate:
    values: dict[str, Any] = {}
    for field in fields(SourceCandidate):
        first_value = getattr(first, field.name)
        second_value = getattr(second, field.name)
        values[field.name] = first_value or second_value
    if len(second.abstract or "") > len(first.abstract or ""):
        values["abstract"] = second.abstract
    if len(second.authors) > len(first.authors):
        values["authors"] = second.authors
    values["source_api"] = first.source_api
    return SourceCandidate(**values)
