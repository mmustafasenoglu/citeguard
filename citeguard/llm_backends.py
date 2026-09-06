"""Multi-provider LLM backend abstraction.

Each provider implements the ``LLMBackend`` protocol: a ``chat()`` method
that sends a system + user message pair and returns the assistant text.

Supported providers:

- ``anthropic`` — Anthropic Messages API
- ``openai`` — OpenAI Responses API (``/v1/responses``)
- ``xai`` — xAI / Grok Responses API (OpenAI-compatible ``/v1/responses``)
- ``groq`` — OpenAI-compatible Chat Completions
- ``openrouter`` — OpenAI-compatible Chat Completions
- ``nvidia`` — NVIDIA NIM Chat Completions
- ``custom`` — any OpenAI-compatible Chat Completions endpoint
  (Ollama, LM Studio, vLLM, LiteLLM, etc.)

Provider selection is driven by ``CITEGUARD_LLM_PROVIDER`` (or auto-detected
from available API keys).  The ``resolve_backend()`` factory returns the
correct implementation.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol

_DEFAULT_TIMEOUT: float = 30


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class LLMBackend(Protocol):
    """Minimal interface for an LLM chat backend."""

    name: str

    def chat(
        self,
        system: str,
        user_message: str,
        *,
        model: str,
        max_tokens: int = 2048,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> str | None: ...


# ---------------------------------------------------------------------------
# Shared HTTP helper
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
                        if isinstance(block, dict) and block.get("type") == "output_text":
                            text = block.get("text", "")
                            if isinstance(text, str) and text:
                                return text
    return None


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicBackend:
    """Anthropic Messages API backend."""

    name = "anthropic"

    def chat(
        self,
        system: str,
        user_message: str,
        *,
        model: str,
        max_tokens: int = 2048,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> str | None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
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
        if data is None:
            return None
        content = data.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            text = content[0].get("text", "")
            if isinstance(text, str):
                return text
        return None


# ---------------------------------------------------------------------------
# OpenAI backend  (Responses API)
# ---------------------------------------------------------------------------

OPENAI_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"


class OpenAIBackend:
    """OpenAI Responses API backend."""

    name = "openai"

    def chat(
        self,
        system: str,
        user_message: str,
        *,
        model: str,
        max_tokens: int = 2048,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> str | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        payload: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "max_output_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        data = _http_post(OPENAI_RESPONSES_ENDPOINT, headers, payload, timeout)
        if data is None:
            return None
        return _extract_text_from_responses(data)


# ---------------------------------------------------------------------------
# xAI / Grok backend  (Responses API, OpenAI-compatible)
# ---------------------------------------------------------------------------

XAI_RESPONSES_ENDPOINT = "https://api.x.ai/v1/responses"


class XAIBackend:
    """xAI / Grok Responses API backend."""

    name = "xai"

    def chat(
        self,
        system: str,
        user_message: str,
        *,
        model: str,
        max_tokens: int = 2048,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> str | None:
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            return None
        payload: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "max_output_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        data = _http_post(XAI_RESPONSES_ENDPOINT, headers, payload, timeout)
        if data is None:
            return None
        return _extract_text_from_responses(data)


# ---------------------------------------------------------------------------
# Groq backend  (OpenAI-compatible Chat Completions)
# ---------------------------------------------------------------------------

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


class GroqBackend:
    """Groq OpenAI-compatible Chat Completions backend."""

    name = "groq"

    def chat(
        self,
        system: str,
        user_message: str,
        *,
        model: str,
        max_tokens: int = 2048,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> str | None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        data = _http_post(GROQ_ENDPOINT, headers, payload, timeout)
        if data is None:
            return None
        return _extract_text_from_openai(data)


# ---------------------------------------------------------------------------
# OpenRouter backend  (OpenAI-compatible Chat Completions)
# ---------------------------------------------------------------------------

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterBackend:
    """OpenRouter OpenAI-compatible Chat Completions backend."""

    name = "openrouter"

    def chat(
        self,
        system: str,
        user_message: str,
        *,
        model: str,
        max_tokens: int = 2048,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> str | None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/mmustafasenoglu/citeguard",
            "X-Title": "citeguard",
        }
        data = _http_post(OPENROUTER_ENDPOINT, headers, payload, timeout)
        if data is None:
            return None
        return _extract_text_from_openai(data)


# ---------------------------------------------------------------------------
# NVIDIA NIM backend  (OpenAI-compatible Chat Completions)
# ---------------------------------------------------------------------------

NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"


class NvidiaBackend:
    """NVIDIA NIM OpenAI-compatible Chat Completions backend."""

    name = "nvidia"

    def chat(
        self,
        system: str,
        user_message: str,
        *,
        model: str,
        max_tokens: int = 2048,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> str | None:
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            return None
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        data = _http_post(NVIDIA_ENDPOINT, headers, payload, timeout)
        if data is None:
            return None
        return _extract_text_from_openai(data)


# ---------------------------------------------------------------------------
# Custom backend  (any OpenAI-compatible Chat Completions endpoint)
# ---------------------------------------------------------------------------


class CustomBackend:
    """OpenAI-compatible Chat Completions for custom endpoints.

    Controlled by ``CITEGUARD_LLM_BASE_URL``, ``CITEGUARD_LLM_API_KEY``,
    and ``CITEGUARD_LLM_MODEL``.
    """

    name = "custom"

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
    ) -> str | None:
        if not self._base_url:
            return None
        endpoint = f"{self._base_url}/chat/completions"
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        data = _http_post(endpoint, headers, payload, timeout)
        if data is None:
            return None
        return _extract_text_from_openai(data)


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
        provider = os.getenv("CITEGUARD_LLM_PROVIDER", "").strip().lower() or None
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
