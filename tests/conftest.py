"""Shared pytest fixtures keeping the suite offline and deterministic.

The repository root may contain a real ``.env`` file, and CLI tests invoke
``load_dotenv()``. Without isolation, provider credentials leak into
``os.environ`` for the rest of the session and later tests (e.g. matcher
entailment with ``offline=False``) make live network calls. This autouse
fixture strips all LLM/provider configuration, neutralizes ``load_dotenv``
inside the CLI, and resets the global router and cache before every test.
"""

from __future__ import annotations

import pytest

_LLM_ENV_VARS = (
    # Provider API keys (auto-detection sources).
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "XAI_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "NVIDIA_API_KEY",
    # Global LLM selection.
    "CITEGUARD_LLM_PROVIDER",
    "CITEGUARD_LLM_MODEL",
    "CITEGUARD_LLM_FALLBACKS",
    "CITEGUARD_LLM_BASE_URL",
    "CITEGUARD_LLM_API_KEY",
    "CITEGUARD_LLM_TIMEOUT",
    # Task-specific overrides (claim extraction, entailment).
    "CITEGUARD_CLAIM_PROVIDER",
    "CITEGUARD_CLAIM_MODEL",
    "CITEGUARD_ENTAILMENT_PROVIDER",
    "CITEGUARD_ENTAILMENT_MODEL",
    # Rewrite provider selection.
    "CITEGUARD_REWRITE_ENABLED",
    "CITEGUARD_REWRITE_MAX_CONTEXT",
    "CITEGUARD_REWRITE_MODEL",
    "CITEGUARD_REWRITE_PROVIDER",
    "CITEGUARD_REWRITE_TIMEOUT",
)


@pytest.fixture(autouse=True)
def _isolate_llm_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _LLM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    import citeguard.cli as cli_module
    import citeguard.llm as llm_module

    # Never load ambient .env files during tests: the repo root may hold
    # real credentials and the suite must stay offline and deterministic.
    monkeypatch.setattr(cli_module, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(llm_module, "_router", None)
    monkeypatch.setattr(llm_module, "_cache", None)
