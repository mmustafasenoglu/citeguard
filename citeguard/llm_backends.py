"""Multi-provider LLM backend abstraction with resilience.

Each provider implements the ``LLMBackend`` protocol: a ``chat()`` method
that sends a system + user message pair and returns an ``LLMResponse``.

Provider selection is driven by ``CITEGUARD_LLM_PROVIDER`` (or auto-detected
from available API keys).  The ``resolve_backend()`` factory returns the
correct implementation.

Supported providers:

- ``anthropic`` — Anthropic Messages API
- ``openai`` — OpenAI Responses API
- ``xai`` — xAI / Grok Responses API
- ``groq`` — OpenAI-compatible Chat Completions
- ``openrouter`` — OpenAI-compatible Chat Completions
- ``nvidia`` — NVIDIA NIM Chat Completions
- ``custom`` — any OpenAI-compatible Chat Completions endpoint
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

_DEFAULT_TIMEOUT: float = 30


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LLMResponse:
    """Structured response from an LLM backend."""

    text: str | None
    provider: str
    model: str
    latency_ms: float
    status_code: int
    attempts: int
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class LLMCapabilities:
    """Provider capability declaration."""

    structured_output: bool = False
    json_schema: bool = False
    temperature: bool = True
    seed: bool = False
    responses_api: bool = False


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Declarative provider specification."""

    name: str
    endpoint: str
    api_key_env: str
    protocol: str  # "chat_completions" | "responses" | "anthropic"
    capabilities: LLMCapabilities = field(default_factory=LLMCapabilities)
    extra_headers: dict[str, str] = field(default_factory=dict)
    default_model: str = ""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class LLMBackend(Protocol):
    """Protocol for LLM chat backends."""

    name: str
    capabilities: LLMCapabilities

    def chat(
        self,
        system: str,
        user_message: str,
        *,
        model: str,
        max_tokens: int = 2048,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _http_post(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any] | None:
    """POST JSON to *url* and return parsed response, or ``None`` on failure."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def _extract_text_from_openai(data: dict[str, Any]) -> str | None:
    """Extract assistant text from an OpenAI-style Chat Completions response."""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {})
        text = msg.get("content", "")
        if isinstance(text, str) and text:
            return text
    return None


def _extract_text_from_responses(data: dict[str, Any]) -> str | None:
    """Extract assistant text from an OpenAI-style Responses API response."""
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                content = item.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "output_text"
                        ):
                            text = block.get("text", "")
                            if isinstance(text, str) and text:
                                return text
    return None


def _extract_usage_openai(data: dict[str, Any]) -> dict[str, int | None]:
    """Extract token usage from an OpenAI-style response."""
    usage = data.get("usage", {})
    return {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _extract_usage_anthropic(data: dict[str, Any]) -> dict[str, int | None]:
    """Extract token usage from an Anthropic response."""
    usage = data.get("usage", {})
    inp = usage.get("input_tokens")
    out = usage.get("output_tokens")
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": (inp or 0) + (out or 0) or None,
    }


# ---------------------------------------------------------------------------
# Generic Chat Completions backend
# ---------------------------------------------------------------------------


class OpenAICompatibleBackend:
    """Generic backend for OpenAI Chat Completions-compatible providers."""

    def __init__(self, spec: ProviderSpec) -> None:
        self._spec = spec
        self.name = spec.name
        self.capabilities = spec.capabilities

    @property
    def api_key(self) -> str | None:
        return os.getenv(self._spec.api_key_env)

    def chat(
        self,
        system: str,
        user_message: str,
        *,
        model: str,
        max_tokens: int = 2048,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> LLMResponse:
        start = time.monotonic()
        api_key = self.api_key
        if not api_key:
            return LLMResponse(
                text=None,
                provider=self.name,
                model=model,
                latency_ms=0,
                status_code=0,
                attempts=0,
                error=f"No API key ({self._spec.api_key_env})",
            )

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        }
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            **self._spec.extra_headers,
        }

        data = _http_post(self._spec.endpoint, headers, payload, timeout)
        latency = (time.monotonic() - start) * 1000

        if data is None:
            return LLMResponse(
                text=None,
                provider=self.name,
                model=model,
                latency_ms=latency,
                status_code=0,
                attempts=1,
                error="Request failed",
            )

        usage = _extract_usage_openai(data)
        return LLMResponse(
            text=_extract_text_from_openai(data),
            provider=self.name,
            model=model,
            latency_ms=latency,
            status_code=200,
            attempts=1,
            **usage,
        )


# ---------------------------------------------------------------------------
# Generic Responses API backend
# ---------------------------------------------------------------------------


class OpenAIResponsesBackend:
    """Generic backend for OpenAI Responses API providers."""

    def __init__(self, spec: ProviderSpec) -> None:
        self._spec = spec
        self.name = spec.name
        self.capabilities = spec.capabilities

    @property
    def api_key(self) -> str | None:
        return os.getenv(self._spec.api_key_env)

    def chat(
        self,
        system: str,
        user_message: str,
        *,
        model: str,
        max_tokens: int = 2048,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> LLMResponse:
        start = time.monotonic()
        api_key = self.api_key
        if not api_key:
            return LLMResponse(
                text=None,
                provider=self.name,
                model=model,
                latency_ms=0,
                status_code=0,
                attempts=0,
                error=f"No API key ({self._spec.api_key_env})",
            )

        payload: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "max_output_tokens": max_tokens,
        }
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        data = _http_post(self._spec.endpoint, headers, payload, timeout)
        latency = (time.monotonic() - start) * 1000

        if data is None:
            return LLMResponse(
                text=None,
                provider=self.name,
                model=model,
                latency_ms=latency,
                status_code=0,
                attempts=1,
                error="Request failed",
            )

        usage = _extract_usage_openai(data)
        return LLMResponse(
            text=_extract_text_from_responses(data),
            provider=self.name,
            model=model,
            latency_ms=latency,
            status_code=200,
            attempts=1,
            **usage,
        )


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicBackend:
    """Anthropic Messages API backend."""

    name = "anthropic"
    capabilities = LLMCapabilities(structured_output=True, temperature=True)

    def chat(
        self,
        system: str,
        user_message: str,
        *,
        model: str,
        max_tokens: int = 2048,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> LLMResponse:
        start = time.monotonic()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return LLMResponse(
                text=None,
                provider="anthropic",
                model=model,
                latency_ms=0,
                status_code=0,
                attempts=0,
                error="No API key (ANTHROPIC_API_KEY)",
            )

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user_message}],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

        data = _http_post(ANTHROPIC_ENDPOINT, headers, payload, timeout)
        latency = (time.monotonic() - start) * 1000

        if data is None:
            return LLMResponse(
                text=None,
                provider="anthropic",
                model=model,
                latency_ms=latency,
                status_code=0,
                attempts=1,
                error="Request failed",
            )

        text = None
        content = data.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            t = content[0].get("text", "")
            if isinstance(t, str):
                text = t

        usage = _extract_usage_anthropic(data)
        return LLMResponse(
            text=text,
            provider="anthropic",
            model=model,
            latency_ms=latency,
            status_code=200,
            attempts=1,
            **usage,
        )


# ---------------------------------------------------------------------------
# Provider specifications
# ---------------------------------------------------------------------------

GROQ_SPEC = ProviderSpec(
    name="groq",
    endpoint="https://api.groq.com/openai/v1/chat/completions",
    api_key_env="GROQ_API_KEY",
    protocol="chat_completions",
    capabilities=LLMCapabilities(structured_output=True),
    default_model="llama-3.3-70b-versatile",
)

OPENROUTER_SPEC = ProviderSpec(
    name="openrouter",
    endpoint="https://openrouter.ai/api/v1/chat/completions",
    api_key_env="OPENROUTER_API_KEY",
    protocol="chat_completions",
    capabilities=LLMCapabilities(),
    extra_headers={
        "HTTP-Referer": "https://github.com/mmustafasenoglu/citeguard",
        "X-Title": "citeguard",
    },
    default_model="anthropic/claude-sonnet-4-20250514",
)

NVIDIA_SPEC = ProviderSpec(
    name="nvidia",
    endpoint="https://integrate.api.nvidia.com/v1/chat/completions",
    api_key_env="NVIDIA_API_KEY",
    protocol="chat_completions",
    capabilities=LLMCapabilities(),
    default_model="meta/llama-3.3-70b-instruct",
)

OPENAI_SPEC = ProviderSpec(
    name="openai",
    endpoint="https://api.openai.com/v1/responses",
    api_key_env="OPENAI_API_KEY",
    protocol="responses",
    capabilities=LLMCapabilities(
        structured_output=True,
        json_schema=True,
        seed=True,
        responses_api=True,
    ),
    default_model="gpt-4o",
)

XAI_SPEC = ProviderSpec(
    name="xai",
    endpoint="https://api.x.ai/v1/responses",
    api_key_env="XAI_API_KEY",
    protocol="responses",
    capabilities=LLMCapabilities(structured_output=True, responses_api=True),
    default_model="grok-3",
)


# ---------------------------------------------------------------------------
# Named wrapper classes (backward compat)
# ---------------------------------------------------------------------------


class GroqBackend(OpenAICompatibleBackend):
    """Groq OpenAI-compatible Chat Completions backend."""

    def __init__(self) -> None:
        super().__init__(GROQ_SPEC)


class OpenRouterBackend(OpenAICompatibleBackend):
    """OpenRouter OpenAI-compatible Chat Completions backend."""

    def __init__(self) -> None:
        super().__init__(OPENROUTER_SPEC)


class NvidiaBackend(OpenAICompatibleBackend):
    """NVIDIA NIM OpenAI-compatible Chat Completions backend."""

    def __init__(self) -> None:
        super().__init__(NVIDIA_SPEC)


class OpenAIBackend(OpenAIResponsesBackend):
    """OpenAI Responses API backend."""

    def __init__(self) -> None:
        super().__init__(OPENAI_SPEC)


class XAIBackend(OpenAIResponsesBackend):
    """xAI / Grok Responses API backend."""

    def __init__(self) -> None:
        super().__init__(XAI_SPEC)


class CustomBackend:
    """OpenAI-compatible Chat Completions for custom endpoints.

    Controlled by ``CITEGUARD_LLM_BASE_URL``, ``CITEGUARD_LLM_API_KEY``,
    and ``CITEGUARD_LLM_MODEL``.
    """

    name = "custom"
    capabilities = LLMCapabilities()

    def __init__(self) -> None:
        self._base_url = os.getenv("CITEGUARD_LLM_BASE_URL", "").rstrip("/")
        self._api_key = os.getenv("CITEGUARD_LLM_API_KEY", "no-key")

    def chat(
        self,
        system: str,
        user_message: str,
        *,
        model: str,
        max_tokens: int = 2048,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> LLMResponse:
        if not self._base_url:
            return LLMResponse(
                text=None,
                provider="custom",
                model=model,
                latency_ms=0,
                status_code=0,
                attempts=0,
                error="No CITEGUARD_LLM_BASE_URL configured",
            )
        endpoint = f"{self._base_url}/chat/completions"
        spec = ProviderSpec(
            name="custom",
            endpoint=endpoint,
            api_key_env="CITEGUARD_LLM_API_KEY",
            protocol="chat_completions",
        )
        backend = OpenAICompatibleBackend(spec)
        # Override api_key property directly
        original = os.environ.get

        def _patched_get(k: str, d: str = "") -> str | None:
            if k == "CITEGUARD_LLM_API_KEY":
                return self._api_key
            return original(k, d)

        os.environ.get = _patched_get
        try:
            return backend.chat(
                system, user_message,
                model=model, max_tokens=max_tokens, timeout=timeout,
            )
        finally:
            os.environ.get = original


# ---------------------------------------------------------------------------
# Provider registry and auto-detection
# ---------------------------------------------------------------------------

_PROVIDER_MAP: dict[str, type] = {
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
    "xai": XAIBackend,
    "groq": GroqBackend,
    "openrouter": OpenRouterBackend,
    "nvidia": NvidiaBackend,
    "custom": CustomBackend,
}

# Priority order for auto-detection from available API keys.
_AUTO_DETECT_KEYS: list[tuple[str, str]] = [
    ("ANTHROPIC_API_KEY", "anthropic"),
    ("OPENAI_API_KEY", "openai"),
    ("XAI_API_KEY", "xai"),
    ("GROQ_API_KEY", "groq"),
    ("OPENROUTER_API_KEY", "openrouter"),
    ("NVIDIA_API_KEY", "nvidia"),
]

DEFAULT_PROVIDER = "anthropic"


def auto_detect_provider() -> str | None:
    """Return the first provider whose API key is set, or ``None``."""
    for env_var, provider in _AUTO_DETECT_KEYS:
        if os.getenv(env_var):
            return provider
    return None


def resolve_backend(provider: str | None = None) -> LLMBackend | None:
    """Instantiate the LLM backend for *provider*.

    When *provider* is ``None``, the ``CITEGUARD_LLM_PROVIDER`` env var is
    checked, then auto-detection from API keys is attempted.  Returns
    ``None`` when no provider or API key is available.
    """
    if provider is None:
        provider = (
            os.getenv("CITEGUARD_LLM_PROVIDER", "").strip().lower() or None
        )
    if provider is None:
        provider = auto_detect_provider()
    if provider is None:
        return None

    cls = _PROVIDER_MAP.get(provider)
    if cls is None:
        return None

    if provider == "custom":
        backend = cls()
        if not backend._base_url:
            return None
        return backend

    return cls()
