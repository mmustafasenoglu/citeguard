"""Tests for LLM router, cache, failover, and diagnostics."""

import time

import pytest

from citeguard.llm_backends import LLMCapabilities, LLMResponse
from citeguard.llm_cache import LLMCache
from citeguard.llm_router import CircuitBreaker, LLMRouter, ProviderHealth

# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------


def test_llm_response_fields() -> None:
    resp = LLMResponse(
        text="hello", provider="groq", model="test",
        latency_ms=100, status_code=200, attempts=1,
    )
    assert resp.text == "hello"
    assert resp.provider == "groq"
    assert resp.error is None
    assert resp.input_tokens is None


def test_llm_response_error() -> None:
    resp = LLMResponse(
        text=None, provider="groq", model="test",
        latency_ms=0, status_code=401, attempts=1,
        error="Unauthorized",
    )
    assert resp.text is None
    assert resp.error == "Unauthorized"


# ---------------------------------------------------------------------------
# LLMCapabilities
# ---------------------------------------------------------------------------


def test_capabilities_defaults() -> None:
    caps = LLMCapabilities()
    assert caps.structured_output is False
    assert caps.json_schema is False
    assert caps.temperature is True
    assert caps.seed is False
    assert caps.responses_api is False


def test_capabilities_custom() -> None:
    caps = LLMCapabilities(structured_output=True, json_schema=True, seed=True)
    assert caps.structured_output is True
    assert caps.json_schema is True
    assert caps.seed is True


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


def test_circuit_breaker_closed_by_default() -> None:
    cb = CircuitBreaker(threshold=3, recovery=60)
    assert cb.is_open is False


def test_circuit_breaker_opens_after_threshold() -> None:
    cb = CircuitBreaker(threshold=3, recovery=60)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open is False
    cb.record_failure()
    assert cb.is_open is True


def test_circuit_breaker_resets_on_success() -> None:
    cb = CircuitBreaker(threshold=3, recovery=60)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open is False


def test_circuit_breaker_recovers_after_timeout() -> None:
    cb = CircuitBreaker(threshold=2, recovery=0.01)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open is True
    time.sleep(0.02)
    assert cb.is_open is False


# ---------------------------------------------------------------------------
# ProviderHealth
# ---------------------------------------------------------------------------


def test_provider_health_defaults() -> None:
    h = ProviderHealth()
    assert h.total == 0
    assert h.success_rate == 1.0
    assert h.avg_latency_ms == 0.0


def test_provider_health_tracking() -> None:
    h = ProviderHealth()
    h.successes = 8
    h.failures = 2
    h.total_latency_ms = 1000
    assert h.total == 10
    assert h.success_rate == 0.8
    assert h.avg_latency_ms == 125.0


def test_provider_health_snapshot() -> None:
    h = ProviderHealth(successes=5, failures=1, total_latency_ms=600)
    snap = h.snapshot()
    assert snap["successes"] == 5
    assert snap["failures"] == 1
    assert snap["success_rate"] == "83%"


# ---------------------------------------------------------------------------
# LLMCache
# ---------------------------------------------------------------------------


def test_cache_miss() -> None:
    cache = LLMCache(prompt_version="1")
    result = cache.get("groq", "model", "sys", "usr")
    assert result is None


def test_cache_hit() -> None:
    cache = LLMCache(prompt_version="1")
    resp = LLMResponse(
        text="cached", provider="groq", model="model",
        latency_ms=100, status_code=200, attempts=1,
    )
    cache.put("groq", "model", "sys", "usr", resp)
    hit = cache.get("groq", "model", "sys", "usr")
    assert hit is not None
    assert hit.text == "cached"


def test_cache_version_invalidation() -> None:
    cache = LLMCache(prompt_version="1")
    resp = LLMResponse(
        text="v1", provider="groq", model="model",
        latency_ms=100, status_code=200, attempts=1,
    )
    cache.put("groq", "model", "sys", "usr", resp)
    cache._prompt_version = "2"
    hit = cache.get("groq", "model", "sys", "usr")
    assert hit is None


def test_cache_ttl_expiry() -> None:
    cache = LLMCache(ttl_seconds=0.01, prompt_version="1")
    resp = LLMResponse(
        text="ttl", provider="groq", model="model",
        latency_ms=100, status_code=200, attempts=1,
    )
    cache.put("groq", "model", "sys", "usr", resp)
    time.sleep(0.02)
    hit = cache.get("groq", "model", "sys", "usr")
    assert hit is None


def test_cache_does_not_store_errors() -> None:
    cache = LLMCache(prompt_version="1")
    resp = LLMResponse(
        text=None, provider="groq", model="model",
        latency_ms=0, status_code=500, attempts=3,
        error="Server error",
    )
    cache.put("groq", "model", "sys", "usr", resp)
    assert cache.size == 0


def test_cache_clear() -> None:
    cache = LLMCache(prompt_version="1")
    resp = LLMResponse(
        text="data", provider="groq", model="model",
        latency_ms=100, status_code=200, attempts=1,
    )
    cache.put("groq", "model", "sys", "usr", resp)
    assert cache.size == 1
    cache.clear()
    assert cache.size == 0


def test_cache_invalidate_stale_versions() -> None:
    cache = LLMCache(prompt_version="1")
    resp = LLMResponse(
        text="old", provider="groq", model="model",
        latency_ms=100, status_code=200, attempts=1,
    )
    cache.put("groq", "model", "sys", "usr", resp)
    cache._prompt_version = "2"
    cache.invalidate_version()
    assert cache.size == 0


def test_cache_stats() -> None:
    cache = LLMCache(prompt_version="1")
    resp = LLMResponse(
        text="data", provider="groq", model="model",
        latency_ms=100, status_code=200, attempts=1,
    )
    cache.put("groq", "model", "sys", "usr", resp)
    stats = cache.stats()
    assert stats["entries"] == 1


# ---------------------------------------------------------------------------
# LLMRouter — using mock backend
# ---------------------------------------------------------------------------


class MockBackend:
    """Minimal mock implementing LLMBackend protocol for router tests."""

    def __init__(
        self,
        responses: list[LLMResponse | None] | None = None,
        name: str = "mock",
    ) -> None:
        self.name = name
        self.capabilities = LLMCapabilities()
        self._responses = list(responses or [])
        self._call_count = 0

    def chat(
        self, system, user_message, *, model="mock",
        max_tokens=2048, timeout=30,
    ) -> LLMResponse:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            if resp is not None:
                return resp
        return LLMResponse(
            text=None, provider=self.name, model=model,
            latency_ms=0, status_code=0, attempts=0,
            error="no more responses",
        )


def test_router_success(monkeypatch) -> None:
    resp = LLMResponse(
        text="ok", provider="groq", model="test",
        latency_ms=100, status_code=200, attempts=1,
    )
    mock = MockBackend(responses=[resp])
    monkeypatch.setattr(
        "citeguard.llm_router.resolve_backend",
        lambda name=None: mock,
    )
    monkeypatch.delenv("CITEGUARD_LLM_FALLBACKS", raising=False)

    router = LLMRouter()
    result = router.call("sys", "usr")
    assert result is not None
    assert result.text == "ok"


def test_router_all_providers_fail(monkeypatch) -> None:
    fail_resp = LLMResponse(
        text=None, provider="mock", model="test",
        latency_ms=100, status_code=500, attempts=3,
        error="Server error",
    )
    mock = MockBackend(responses=[fail_resp])
    monkeypatch.setattr(
        "citeguard.llm_router.resolve_backend",
        lambda name=None: mock,
    )
    monkeypatch.delenv("CITEGUARD_LLM_FALLBACKS", raising=False)

    router = LLMRouter()
    result = router.call("sys", "usr")
    assert result is None


def test_router_auth_error_no_retry(monkeypatch) -> None:
    auth_resp = LLMResponse(
        text=None, provider="mock", model="test",
        latency_ms=50, status_code=401, attempts=1,
        error="Unauthorized",
    )
    mock = MockBackend(responses=[auth_resp])
    monkeypatch.setattr(
        "citeguard.llm_router.resolve_backend",
        lambda name=None: mock,
    )
    monkeypatch.delenv("CITEGUARD_LLM_FALLBACKS", raising=False)

    router = LLMRouter()
    result = router.call("sys", "usr")
    assert result is None
    assert mock._call_count == 1


def test_router_circuit_breaker(monkeypatch) -> None:
    fail_resp = LLMResponse(
        text=None, provider="mock", model="test",
        latency_ms=50, status_code=500, attempts=1,
        error="Server error",
    )

    def make_mock(name=None):
        m = MockBackend(responses=[fail_resp, fail_resp, fail_resp, fail_resp, fail_resp])
        return m

    monkeypatch.setattr("citeguard.llm_router.resolve_backend", make_mock)
    monkeypatch.delenv("CITEGUARD_LLM_FALLBACKS", raising=False)

    router = LLMRouter()
    for _ in range(6):
        router.call("sys", "usr")

    snap = router.health_snapshot()
    assert any(
        s.get("failures", 0) >= 5
        for s in snap.values()
    )


def test_router_health_snapshot(monkeypatch) -> None:
    resp = LLMResponse(
        text="ok", provider="groq", model="test",
        latency_ms=100, status_code=200, attempts=1,
    )
    mock = MockBackend(responses=[resp])
    monkeypatch.setattr(
        "citeguard.llm_router.resolve_backend",
        lambda name=None: mock,
    )
    monkeypatch.delenv("CITEGUARD_LLM_FALLBACKS", raising=False)

    router = LLMRouter()
    router.call("sys", "usr")
    snap = router.health_snapshot()
    assert len(snap) == 1


def test_router_active_provider(monkeypatch) -> None:
    monkeypatch.setenv("CITEGUARD_LLM_PROVIDER", "groq")
    monkeypatch.delenv("CITEGUARD_LLM_FALLBACKS", raising=False)

    router = LLMRouter()
    assert router.active_provider == "groq"


# ---------------------------------------------------------------------------
# Live integration tests
# ---------------------------------------------------------------------------


@pytest.mark.live_llm
@pytest.mark.skipif(
    not any(
        __import__("os").getenv(k)
        for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"]
    ),
    reason="No LLM API key configured",
)
def test_live_llm_basic_chat() -> None:
    """Smoke test against real provider. Run with: pytest -m live_llm."""
    from citeguard.llm_backends import resolve_backend

    backend = resolve_backend()
    assert backend is not None
    resp = backend.chat(
        "Reply with one word: ok",
        "Say ok",
        model="default",
        max_tokens=8,
        timeout=15,
    )
    assert resp.text is not None
    assert len(resp.text) > 0
    assert resp.status_code == 200


@pytest.mark.live_llm
@pytest.mark.skipif(
    not any(
        __import__("os").getenv(k)
        for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"]
    ),
    reason="No LLM API key configured",
)
def test_live_llm_router_with_failover() -> None:
    """Test router end-to-end with real providers."""
    router = LLMRouter()
    resp = router.call(
        "Reply with one word: ok",
        "Say ok",
        max_tokens=8,
    )
    if resp:
        assert resp.text is not None
