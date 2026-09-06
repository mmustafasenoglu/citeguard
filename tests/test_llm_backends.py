"""Tests for multi-provider LLM backends."""

import pytest

from citeguard.llm_backends import (
    AnthropicBackend,
    CustomBackend,
    GroqBackend,
    NvidiaBackend,
    OpenAIBackend,
    OpenRouterBackend,
    XAIBackend,
    _extract_text_from_openai,
    _extract_text_from_responses,
    auto_detect_provider,
    resolve_backend,
)

# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def test_auto_detect_returns_none_without_keys(monkeypatch) -> None:
    for var in [
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY",
        "GROQ_API_KEY", "OPENROUTER_API_KEY", "NVIDIA_API_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)
    assert auto_detect_provider() is None


def test_auto_detect_prefers_anthropic(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert auto_detect_provider() == "anthropic"


def test_auto_detect_falls_to_openai(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert auto_detect_provider() == "openai"


def test_auto_detect_groq(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert auto_detect_provider() == "groq"


def test_auto_detect_xai(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    assert auto_detect_provider() == "xai"


# ---------------------------------------------------------------------------
# resolve_backend
# ---------------------------------------------------------------------------


def test_resolve_backend_returns_none_without_provider(monkeypatch) -> None:
    monkeypatch.delenv("CITEGUARD_LLM_PROVIDER", raising=False)
    for var in [
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY",
        "GROQ_API_KEY", "OPENROUTER_API_KEY", "NVIDIA_API_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)
    assert resolve_backend() is None


def test_resolve_backend_explicit_provider(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CITEGUARD_LLM_PROVIDER", "groq")
    backend = resolve_backend("groq")
    assert backend is not None
    assert backend.name == "groq"


def test_resolve_backend_custom_requires_base_url(monkeypatch) -> None:
    monkeypatch.setenv("CITEGUARD_LLM_PROVIDER", "custom")
    monkeypatch.delenv("CITEGUARD_LLM_BASE_URL", raising=False)
    assert resolve_backend("custom") is None


def test_resolve_backend_custom_with_url(monkeypatch) -> None:
    monkeypatch.setenv("CITEGUARD_LLM_PROVIDER", "custom")
    monkeypatch.setenv("CITEGUARD_LLM_BASE_URL", "http://localhost:11434/v1")
    backend = resolve_backend("custom")
    assert backend is not None
    assert backend.name == "custom"


def test_resolve_backend_invalid_provider() -> None:
    assert resolve_backend("nonexistent") is None


# ---------------------------------------------------------------------------
# Provider instances
# ---------------------------------------------------------------------------


def test_provider_names() -> None:
    assert AnthropicBackend().name == "anthropic"
    assert OpenAIBackend().name == "openai"
    assert XAIBackend().name == "xai"
    assert GroqBackend().name == "groq"
    assert OpenRouterBackend().name == "openrouter"
    assert NvidiaBackend().name == "nvidia"
    assert CustomBackend().name == "custom"


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def test_extract_text_from_openai_valid() -> None:
    data = {
        "choices": [
            {"message": {"content": "Hello world"}}
        ]
    }
    assert _extract_text_from_openai(data) == "Hello world"


def test_extract_text_from_openai_empty() -> None:
    assert _extract_text_from_openai({"choices": []}) is None
    assert _extract_text_from_openai({}) is None


def test_extract_text_from_responses_valid() -> None:
    data = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Response text"}
                ],
            }
        ]
    }
    assert _extract_text_from_responses(data) == "Response text"


def test_extract_text_from_responses_empty() -> None:
    assert _extract_text_from_responses({"output": []}) is None
    assert _extract_text_from_responses({}) is None


# ---------------------------------------------------------------------------
# Backend chat with missing keys returns None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend_cls,var",
    [
        (AnthropicBackend, "ANTHROPIC_API_KEY"),
        (OpenAIBackend, "OPENAI_API_KEY"),
        (XAIBackend, "XAI_API_KEY"),
        (GroqBackend, "GROQ_API_KEY"),
        (OpenRouterBackend, "OPENROUTER_API_KEY"),
        (NvidiaBackend, "NVIDIA_API_KEY"),
    ],
)
def test_backend_returns_none_without_key(backend_cls, var, monkeypatch) -> None:
    monkeypatch.delenv(var, raising=False)
    backend = backend_cls()
    result = backend.chat("system", "user", model="test-model")
    assert result is None


def test_custom_backend_returns_none_without_url(monkeypatch) -> None:
    monkeypatch.delenv("CITEGUARD_LLM_BASE_URL", raising=False)
    backend = CustomBackend()
    result = backend.chat("system", "user", model="test-model")
    assert result is None


# ---------------------------------------------------------------------------
# HTTP response extraction edge cases
# ---------------------------------------------------------------------------


def test_extract_text_from_openai_nested_content() -> None:
    data = {
        "choices": [
            {"message": {"content": ""}}
        ]
    }
    assert _extract_text_from_openai(data) is None


def test_extract_text_from_responses_non_message() -> None:
    data = {
        "output": [
            {"type": "function_call", "name": "test"}
        ]
    }
    assert _extract_text_from_responses(data) is None
