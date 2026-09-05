from citeguard.cache import FileCache
from citeguard.models import SourceCandidate
from citeguard.providers.base import ProviderError
from citeguard.retrieval import (
    RetrievalEngine,
    candidate_identity,
    deduplicate_candidates,
    normalize_arxiv_id,
    normalize_doi,
)


def candidate(
    title="A Paper", *, source="test", doi=None, arxiv_id=None, year=2022, abstract=None
):
    return SourceCandidate(title, [], year, None, doi, None, abstract, source, arxiv_id)


class StubProvider:
    def __init__(self, name, results=None, error=None):
        self.name = name
        self.results = results or []
        self.error = error
        self.calls = []

    def search(self, query, max_results=5):
        self.calls.append((query, max_results))
        if self.error:
            raise self.error
        return self.results


def test_doi_normalization() -> None:
    assert normalize_doi("https://doi.org/10.1000/ABC.123).") == "10.1000/abc.123"
    assert normalize_doi("doi: 10.5555/X_Y") == "10.5555/x_y"
    assert normalize_doi("not a doi") is None


def test_arxiv_normalization() -> None:
    assert normalize_arxiv_id("https://arxiv.org/abs/1706.03762v7") == "1706.03762"
    assert normalize_arxiv_id("arXiv:hep-th/9901001v2") == "hep-th/9901001"


def test_identity_uses_canonical_priority() -> None:
    item = candidate(doi="10.1/no", arxiv_id="1706.03762")
    assert candidate_identity(item) == ("arxiv", "1706.03762")
    item.doi = "10.1234/YES"
    assert candidate_identity(item) == ("doi", "10.1234/yes")


def test_deduplicates_across_providers_and_fills_metadata() -> None:
    first = candidate("Shared title", source="semantic_scholar", doi="10.1000/SAME")
    second = candidate(
        "Shared title",
        source="crossref",
        doi="https://doi.org/10.1000/same",
        abstract="More metadata",
    )
    results = deduplicate_candidates([first, second])
    assert len(results) == 1
    assert results[0].source_api == "semantic_scholar"
    assert results[0].abstract == "More metadata"


def test_provider_failure_falls_back_and_preserves_order() -> None:
    semantic = StubProvider("semantic_scholar", error=ProviderError("unavailable"))
    crossref = StubProvider("crossref", [candidate("Weakly related", source="crossref")])
    arxiv = StubProvider("arxiv", [candidate("Exact query words", source="arxiv")])
    engine = RetrievalEngine([semantic, crossref, arxiv])
    result = engine.search_with_result("Exact query words")
    assert [provider.name for provider in (semantic, crossref, arxiv) if provider.calls] == [
        "semantic_scholar",
        "crossref",
        "arxiv",
    ]
    assert result.candidates[0].source_api == "arxiv"
    assert result.warnings == ["semantic_scholar: unavailable"]


def test_strong_semantic_scholar_match_early_stops() -> None:
    semantic = StubProvider(
        "semantic_scholar", [candidate("Exact Paper Title", source="semantic_scholar")]
    )
    crossref = StubProvider("crossref", [candidate(source="crossref")])
    result = RetrievalEngine([semantic, crossref]).search_with_result("Exact Paper Title")
    assert result.early_stopped is True
    assert result.queried_providers == ["semantic_scholar"]
    assert crossref.calls == []


def test_non_exact_result_searches_every_provider() -> None:
    providers = [
        StubProvider("semantic_scholar", [candidate("Query words and additions")]),
        StubProvider("crossref"),
        StubProvider("arxiv"),
    ]
    RetrievalEngine(providers).search("Query words")
    assert all(provider.calls for provider in providers)


def test_empty_results_are_safe() -> None:
    result = RetrievalEngine([StubProvider("one"), StubProvider("two")]).search("nothing")
    assert result == []


def test_cache_and_no_cache_behavior(tmp_path) -> None:
    provider = StubProvider("crossref", [candidate(source="crossref")])
    cache = FileCache(tmp_path)
    assert len(RetrievalEngine([provider], cache=cache).search("query")) == 1
    provider.results = []
    assert len(RetrievalEngine([provider], cache=cache).search("query")) == 1
    assert len(provider.calls) == 1
    assert RetrievalEngine([provider], cache=cache, use_cache=False).search("query") == []
    assert len(provider.calls) == 2


def test_parallel_search_returns_results() -> None:
    providers = [
        StubProvider("semantic_scholar", [candidate("Paper A", source="semantic_scholar")]),
        StubProvider("crossref", [candidate("Paper B", source="crossref")]),
        StubProvider("arxiv", [candidate("Paper C", source="arxiv")]),
    ]
    engine = RetrievalEngine(providers, parallel=True, rate_limit=False)
    results = engine.search("Paper")
    assert len(results) == 3
    assert engine.last_result is not None
    assert set(engine.last_result.queried_providers) == {
        "semantic_scholar", "crossref", "arxiv"
    }


def test_parallel_search_with_provider_failure() -> None:
    providers = [
        StubProvider("semantic_scholar", error=ProviderError("down")),
        StubProvider("crossref", [candidate("Paper B", source="crossref")]),
        StubProvider("arxiv", [candidate("Paper C", source="arxiv")]),
    ]
    engine = RetrievalEngine(providers, parallel=True, rate_limit=False)
    results = engine.search("Paper")
    assert len(results) == 2
    assert engine.last_result is not None
    assert any("semantic_scholar" in w for w in engine.last_result.warnings)


def test_sequential_fallback_when_parallel_false() -> None:
    providers = [
        StubProvider("semantic_scholar", [candidate("Paper A", source="semantic_scholar")]),
        StubProvider("crossref", [candidate("Paper B", source="crossref")]),
    ]
    engine = RetrievalEngine(providers, parallel=False, rate_limit=False)
    results = engine.search("Paper")
    assert len(results) == 2
    for p in providers:
        assert len(p.calls) == 1


def test_rate_limiter_delays_between_calls() -> None:
    from citeguard.retrieval import _RateLimiter

    limiter = _RateLimiter(min_interval=0.05)
    limiter.wait("crossref")
    limiter.wait("crossref")
    limiter.wait("arxiv")
    assert len(limiter._last_call) == 2


def test_early_stop_still_works_with_parallel() -> None:
    semantic = StubProvider(
        "semantic_scholar",
        [candidate("Exact Paper Title", source="semantic_scholar")],
    )
    crossref = StubProvider("crossref", [candidate(source="crossref")])
    engine = RetrievalEngine([semantic, crossref], parallel=True, rate_limit=False)
    result = engine.search_with_result("Exact Paper Title")
    assert result.early_stopped is True
    assert crossref.calls == []


def test_single_provider_always_sequential() -> None:
    provider = StubProvider("crossref", [candidate("Unrelated stuff", source="crossref")])
    engine = RetrievalEngine([provider], parallel=True, rate_limit=False)
    result = engine.search_with_result("Paper")
    assert result.early_stopped is False
    assert len(result.candidates) == 1
