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
    _RawHTTPResponse,
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


def test_provider_capabilities() -> None:
    assert AnthropicBackend().capabilities.structured_output is True
    assert OpenAIBackend().capabilities.json_schema is True
    assert OpenAIBackend().capabilities.seed is True
    assert GroqBackend().capabilities.structured_output is True
    assert OpenRouterBackend().capabilities.structured_output is False
    assert NvidiaBackend().capabilities.structured_output is False


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
# Backend chat with missing keys returns LLMResponse with error
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
def test_backend_returns_error_without_key(backend_cls, var, monkeypatch) -> None:
    monkeypatch.delenv(var, raising=False)
    backend = backend_cls()
    result = backend.chat("system", "user", model="test-model")
    assert result.text is None
    assert result.error is not None
    assert result.status_code == 0


def test_custom_backend_returns_error_without_url(monkeypatch) -> None:
    monkeypatch.delenv("CITEGUARD_LLM_BASE_URL", raising=False)
    backend = CustomBackend()
    result = backend.chat("system", "user", model="test-model")
    assert result.text is None
    assert result.error is not None


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


# ---------------------------------------------------------------------------
# HTTP contract tests — verify endpoint, headers, and payload shape
# ---------------------------------------------------------------------------


def test_anthropic_contract(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    captured: dict = {}

    def fake_post_raw(url, headers, payload, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return _RawHTTPResponse(
            data={"content": [{"type": "text", "text": "ok"}]},
            status_code=200, headers={},
        )

    monkeypatch.setattr("citeguard.llm_backends._http_post_raw", fake_post_raw)
    backend = AnthropicBackend()
    result = backend.chat("sys", "usr", model="claude-3", max_tokens=100)

    assert result.text == "ok"
    assert result.provider == "anthropic"
    assert result.status_code == 200
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-ant-test-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["payload"]["model"] == "claude-3"
    assert captured["payload"]["max_tokens"] == 100
    assert captured["payload"]["system"] == "sys"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "usr"}]


def test_openai_contract(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key")
    captured: dict = {}

    def fake_post_raw(url, headers, payload, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return _RawHTTPResponse(
            data={
                "output": [
                    {"type": "message", "content": [
                        {"type": "output_text", "text": "hi"},
                    ]},
                ],
            },
            status_code=200, headers={},
        )

    monkeypatch.setattr("citeguard.llm_backends._http_post_raw", fake_post_raw)
    backend = OpenAIBackend()
    result = backend.chat("sys", "usr", model="gpt-4o")

    assert result.text == "hi"
    assert result.provider == "openai"
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer sk-openai-key"
    assert captured["payload"]["model"] == "gpt-4o"
    assert captured["payload"]["input"][0]["role"] == "system"
    assert captured["payload"]["input"][1]["role"] == "user"
    assert "max_output_tokens" in captured["payload"]


def test_xai_contract(monkeypatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    captured: dict = {}

    def fake_post_raw(url, headers, payload, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return _RawHTTPResponse(
            data={
                "output": [
                    {"type": "message", "content": [
                        {"type": "output_text", "text": "grok"},
                    ]},
                ],
            },
            status_code=200, headers={},
        )

    monkeypatch.setattr("citeguard.llm_backends._http_post_raw", fake_post_raw)
    backend = XAIBackend()
    result = backend.chat("sys", "usr", model="grok-3")

    assert result.text == "grok"
    assert captured["url"] == "https://api.x.ai/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer xai-test-key"
    assert captured["payload"]["model"] == "grok-3"


def test_groq_contract(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    captured: dict = {}

    def fake_post_raw(url, headers, payload, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return _RawHTTPResponse(
            data={"choices": [{"message": {"content": "groq-res"}}]},
            status_code=200, headers={},
        )

    monkeypatch.setattr("citeguard.llm_backends._http_post_raw", fake_post_raw)
    backend = GroqBackend()
    result = backend.chat("sys", "usr", model="llama-3.3-70b")

    assert result.text == "groq-res"
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer gsk_test_key"
    assert captured["payload"]["model"] == "llama-3.3-70b"
    assert captured["payload"]["messages"][0]["role"] == "system"
    assert captured["payload"]["messages"][1]["role"] == "user"


def test_openrouter_contract(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    captured: dict = {}

    def fake_post_raw(url, headers, payload, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return _RawHTTPResponse(
            data={"choices": [{"message": {"content": "or-res"}}]},
            status_code=200, headers={},
        )

    monkeypatch.setattr("citeguard.llm_backends._http_post_raw", fake_post_raw)
    backend = OpenRouterBackend()
    result = backend.chat("sys", "usr", model="anthropic/claude-3")

    assert result.text == "or-res"
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-or-test"
    assert captured["headers"]["HTTP-Referer"] == (
        "https://github.com/mmustafasenoglu/citeguard"
    )
    assert captured["headers"]["X-Title"] == "citeguard"


def test_nvidia_contract(monkeypatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    captured: dict = {}

    def fake_post_raw(url, headers, payload, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return _RawHTTPResponse(
            data={"choices": [{"message": {"content": "nvidia-res"}}]},
            status_code=200, headers={},
        )

    monkeypatch.setattr("citeguard.llm_backends._http_post_raw", fake_post_raw)
    backend = NvidiaBackend()
    result = backend.chat("sys", "usr", model="meta/llama-3.3-70b-instruct")

    assert result.text == "nvidia-res"
    assert captured["url"] == (
        "https://integrate.api.nvidia.com/v1/chat/completions"
    )
    assert captured["headers"]["Authorization"] == "Bearer nvapi-test"
    assert captured["payload"]["model"] == "meta/llama-3.3-70b-instruct"


def test_custom_contract(monkeypatch) -> None:
    monkeypatch.setenv("CITEGUARD_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("CITEGUARD_LLM_API_KEY", "ollama")
    captured: dict = {}

    def fake_post_raw(url, headers, payload, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return _RawHTTPResponse(
            data={"choices": [{"message": {"content": "local-res"}}]},
            status_code=200, headers={},
        )

    monkeypatch.setattr("citeguard.llm_backends._http_post_raw", fake_post_raw)
    backend = CustomBackend()
    result = backend.chat("sys", "usr", model="qwen3")

    assert result.text == "local-res"
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer ollama"
    assert captured["payload"]["model"] == "qwen3"


# ---------------------------------------------------------------------------
# Structured output in payload
# ---------------------------------------------------------------------------


def test_structured_output_in_openai_payload(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key")
    captured: dict = {}

    def fake_post_raw(url, headers, payload, timeout):
        captured["payload"] = payload
        return _RawHTTPResponse(
            data={
                "output": [
                    {"type": "message", "content": [
                        {"type": "output_text", "text": "[]"},
                    ]},
                ],
            },
            status_code=200, headers={},
        )

    monkeypatch.setattr("citeguard.llm_backends._http_post_raw", fake_post_raw)
    schema = {"type": "array", "items": {"type": "object"}}
    backend = OpenAIBackend()
    backend.chat("sys", "usr", model="gpt-4o", json_schema=schema)

    assert "text" in captured["payload"]
    fmt = captured["payload"]["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"] == schema


def test_structured_output_not_added_when_unsupported(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    captured: dict = {}

    def fake_post_raw(url, headers, payload, timeout):
        captured["payload"] = payload
        return _RawHTTPResponse(
            data={"choices": [{"message": {"content": "ok"}}]},
            status_code=200, headers={},
        )

    monkeypatch.setattr("citeguard.llm_backends._http_post_raw", fake_post_raw)
    schema = {"type": "array", "items": {"type": "object"}}
    backend = GroqBackend()
    backend.chat("sys", "usr", model="llama-3.3-70b", json_schema=schema)

    assert "response_format" not in captured["payload"]


# ---------------------------------------------------------------------------
# HTTP error propagation
# ---------------------------------------------------------------------------


def test_http_error_propagates_status_code(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

    def fake_post_raw(url, headers, payload, timeout):
        return _RawHTTPResponse(
            data=None, status_code=429,
            headers={"Retry-After": "5"},
            error="HTTP 429: rate limited",
        )

    monkeypatch.setattr("citeguard.llm_backends._http_post_raw", fake_post_raw)
    backend = GroqBackend()
    result = backend.chat("sys", "usr", model="llama-3")

    assert result.text is None
    assert result.status_code == 429
    assert result.retry_after == 5.0
    assert result.error is not None


def test_http_auth_error_propagates_status_code(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def fake_post_raw(url, headers, payload, timeout):
        return _RawHTTPResponse(
            data=None, status_code=401,
            headers={},
            error="HTTP 401: invalid key",
        )

    monkeypatch.setattr("citeguard.llm_backends._http_post_raw", fake_post_raw)
    backend = OpenAIBackend()
    result = backend.chat("sys", "usr", model="gpt-4o")

    assert result.text is None
    assert result.status_code == 401
    assert result.error is not None


# ---------------------------------------------------------------------------
# api_key_override for CustomBackend
# ---------------------------------------------------------------------------


def test_custom_backend_uses_override_key(monkeypatch) -> None:
    monkeypatch.setenv("CITEGUARD_LLM_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("CITEGUARD_LLM_API_KEY", "ollama")
    captured: dict = {}

    def fake_post_raw(url, headers, payload, timeout):
        captured["headers"] = headers
        return _RawHTTPResponse(
            data={"choices": [{"message": {"content": "hi"}}]},
            status_code=200, headers={},
        )

    monkeypatch.setattr("citeguard.llm_backends._http_post_raw", fake_post_raw)
    backend = CustomBackend()
    result = backend.chat("sys", "usr", model="test")

    assert result.text == "hi"
    assert captured["headers"]["Authorization"] == "Bearer ollama"


# ---------------------------------------------------------------------------
# _parse_retry_after helper
# ---------------------------------------------------------------------------


def test_parse_retry_after_valid() -> None:
    from citeguard.llm_backends import _parse_retry_after
    assert _parse_retry_after({"Retry-After": "10"}) == 10.0
    assert _parse_retry_after({"retry-after": "2.5"}) == 2.5


def test_parse_retry_after_missing() -> None:
    from citeguard.llm_backends import _parse_retry_after
    assert _parse_retry_after({}) is None
    assert _parse_retry_after({"Retry-After": "invalid"}) is None
