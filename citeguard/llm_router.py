"""LLM router with failover, retry, circuit breaker, and health tracking.

The router wraps individual backends with resilience patterns:

- **Retry with exponential backoff** on transient errors (429, 5xx, timeout).
  Respects ``Retry-After`` headers.
- **Circuit breaker** per provider: after *threshold* consecutive failures the
  provider is skipped for *recovery* seconds.
- **Failover chain**: primary provider → fallbacks from
  ``CITEGUARD_LLM_FALLBACKS=openai,groq,openrouter``.
- **Health tracking**: per-provider success/failure counts and latency
  statistics for the current process.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from .llm_backends import (
    LLMBackend,
    LLMResponse,
    resolve_backend,
)

log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 30.0
_CIRCUIT_BREAKER_THRESHOLD = 5
_CIRCUIT_BREAKER_RECOVERY_S = 60.0
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_AUTH_ERROR_CODES = {401, 403}


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Per-provider circuit breaker.

    After *threshold* consecutive failures the circuit opens and rejects
    requests for *recovery* seconds.  A single success resets the counter.
    """

    def __init__(
        self,
        threshold: int = _CIRCUIT_BREAKER_THRESHOLD,
        recovery: float = _CIRCUIT_BREAKER_RECOVERY_S,
    ) -> None:
        self._threshold = threshold
        self._recovery = recovery
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self._recovery:
            self._reset()
            return False
        return True

    def record_success(self) -> None:
        self._reset()

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()

    def _reset(self) -> None:
        self._failures = 0
        self._opened_at = None


# ---------------------------------------------------------------------------
# Provider health tracking
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProviderHealth:
    """Rolling health stats for a single provider."""

    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    last_error: str | None = None

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 1.0
        return self.successes / self.total

    @property
    def avg_latency_ms(self) -> float:
        if self.successes == 0:
            return 0.0
        return self.total_latency_ms / self.successes

    def snapshot(self) -> dict[str, object]:
        return {
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": f"{self.success_rate:.0%}",
            "avg_latency_ms": f"{self.avg_latency_ms:.0f}",
            "last_error": self.last_error,
        }


# ---------------------------------------------------------------------------
# LLM Router
# ---------------------------------------------------------------------------


def _parse_fallbacks() -> list[str]:
    """Parse ``CITEGUARD_LLM_FALLBACKS`` into a list of provider names."""
    raw = os.getenv("CITEGUARD_LLM_FALLBACKS", "").strip()
    if not raw:
        return []
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def _is_retryable(status_code: int) -> bool:
    return status_code in _RETRYABLE_STATUS_CODES


def _is_auth_error(status_code: int) -> bool:
    return status_code in _AUTH_ERROR_CODES


class LLMRouter:
    """Resilient LLM router with retry, circuit breaker, and failover.

    Usage::

        router = LLMRouter()
        resp = router.call("system", "user msg", model="gpt-4o")
        if resp and resp.text:
            process(resp.text)
    """

    def __init__(self, *, timeout: float = 30.0) -> None:
        self._timeout = timeout
        self._breakers: dict[str, CircuitBreaker] = {}
        self._health: dict[str, ProviderHealth] = {}
        self._primary_name: str | None = None
        self._fallback_names: list[str] = []

        self._init_providers()

    def _init_providers(self) -> None:
        primary = resolve_backend()
        if primary is not None:
            self._primary_name = primary.name
        self._fallback_names = _parse_fallbacks()

    def _get_breaker(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker()
        return self._breakers[name]

    def _get_health(self, name: str) -> ProviderHealth:
        if name not in self._health:
            self._health[name] = ProviderHealth()
        return self._health[name]

    def _build_chain(self) -> list[str]:
        """Build ordered provider chain: primary → fallbacks."""
        chain: list[str] = []
        if self._primary_name:
            chain.append(self._primary_name)
        for name in self._fallback_names:
            if name not in chain:
                chain.append(name)
        return chain

    def _try_single(
        self,
        backend: LLMBackend,
        system: str,
        user_message: str,
        *,
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Try a single provider with retry + backoff."""
        cb = self._get_breaker(backend.name)
        health = self._get_health(backend.name)
        active_model = model or "default"

        # Circuit breaker check
        if cb.is_open:
            log.debug(
                "Circuit breaker open for %s, skipping", backend.name,
            )
            return LLMResponse(
                text=None,
                provider=backend.name,
                model=active_model,
                latency_ms=0,
                status_code=0,
                attempts=0,
                error="Circuit breaker open",
            )

        last_resp: LLMResponse | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            resp = backend.chat(
                system, user_message,
                model=active_model,
                max_tokens=max_tokens,
                timeout=self._timeout,
            )
            last_resp = resp

            # Success
            if resp.text is not None:
                cb.record_success()
                health.successes += 1
                health.total_latency_ms += resp.latency_ms
                resp.attempts = attempt
                return resp

            # Auth errors: do not retry
            if _is_auth_error(resp.status_code):
                cb.record_failure()
                health.failures += 1
                health.last_error = resp.error
                resp.attempts = attempt
                return resp

            # Retryable errors (429, 5xx, timeout)
            retryable = _is_retryable(resp.status_code) or resp.status_code == 0
            if retryable and attempt < _MAX_RETRIES:
                backoff = min(
                    _INITIAL_BACKOFF_S * (2 ** (attempt - 1)),
                    _MAX_BACKOFF_S,
                )
                log.debug(
                    "Retry %d/%d for %s in %.1fs (status=%d)",
                    attempt, _MAX_RETRIES, backend.name,
                    backoff, resp.status_code,
                )
                time.sleep(backoff)
                continue

            # Non-retryable client errors or exhausted retries
            break

        # All attempts failed
        cb.record_failure()
        health.failures += 1
        health.last_error = last_resp.error if last_resp else "unknown"
        if last_resp:
            last_resp.attempts = _MAX_RETRIES
        return last_resp or LLMResponse(
            text=None,
            provider=backend.name,
            model=active_model,
            latency_ms=0,
            status_code=0,
            attempts=0,
            error="No response",
        )

    def call(
        self,
        system: str,
        user_message: str,
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        timeout: float | None = None,
    ) -> LLMResponse | None:
        """Call through the provider chain with retry and failover.

        Returns ``None`` when all providers are exhausted.
        """
        if timeout is not None:
            self._timeout = timeout

        chain = self._build_chain()
        for provider_name in chain:
            backend = resolve_backend(provider_name)
            if backend is None:
                continue

            resp = self._try_single(
                backend, system, user_message,
                model=model, max_tokens=max_tokens,
            )
            if resp.text is not None:
                return resp

            log.debug(
                "Provider %s failed: %s, trying next",
                provider_name, resp.error,
            )

        return None

    def health_snapshot(self) -> dict[str, dict[str, object]]:
        """Return per-provider health stats for the current process."""
        return {name: h.snapshot() for name, h in self._health.items()}

    @property
    def active_provider(self) -> str | None:
        """Return the name of the primary configured provider."""
        return self._primary_name

    @property
    def fallback_providers(self) -> list[str]:
        return list(self._fallback_names)
