"""Global defaults and environment-backed settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CACHE_SCHEMA_VERSION = "2"
CLAIM_PROMPT_VERSION = "1"
MATCHER_PROMPT_VERSION = "1"
LINKER_PROMPT_VERSION = "1"
LLM_CACHE_PROMPT_VERSION = "1"

DEFAULT_THRESHOLD = 60
STRONG_MATCH_THRESHOLD = 80
DEFAULT_MAX_RESULTS = 5
DEFAULT_CACHE_DIR = Path(".citeguard_cache")

# Default models per provider
_PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "xai": "grok-3",
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "anthropic/claude-sonnet-4-20250514",
    "nvidia": "meta/llama-3.3-70b-instruct",
}

# Provider-specific keys mapping
_PROVIDER_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "xai": "XAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "custom": "CITEGUARD_LLM_API_KEY",
}


@dataclass(frozen=True, slots=True)
class LLMProviderSettings:
    """Resolved LLM provider configuration."""

    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None

    @classmethod
    def from_env(cls) -> LLMProviderSettings:
        """Build provider settings from environment variables.

        Resolution order for provider:
        1. ``CITEGUARD_LLM_PROVIDER``
        2. Auto-detect from available API keys

        Resolution order for model:
        1. ``CITEGUARD_LLM_MODEL``
        2. Provider-specific default
        """
        from .llm_backends import auto_detect_provider

        provider = (
            os.getenv("CITEGUARD_LLM_PROVIDER", "").strip().lower() or None
        )
        if provider is None:
            provider = auto_detect_provider()

        if provider is None:
            provider = "anthropic"

        model = os.getenv("CITEGUARD_LLM_MODEL", "").strip() or None
        if model is None:
            model = _PROVIDER_DEFAULT_MODELS.get(
                provider, "claude-sonnet-4-20250514",
            )

        api_key = _resolve_api_key(provider)
        base_url = (
            os.getenv("CITEGUARD_LLM_BASE_URL", "").strip() or None
        )

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )


def _resolve_api_key(provider: str) -> str | None:
    """Return the API key for *provider*, checking env vars in priority order."""
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY")
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    if provider == "xai":
        return os.getenv("XAI_API_KEY")
    if provider == "groq":
        return os.getenv("GROQ_API_KEY")
    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY")
    if provider == "nvidia":
        return os.getenv("NVIDIA_API_KEY")
    if provider == "custom":
        return os.getenv("CITEGUARD_LLM_API_KEY", "no-key")
    return None


@dataclass(frozen=True, slots=True)
class TaskModelSettings:
    """Task-specific LLM model configuration."""

    provider: str | None = None
    model: str | None = None

    @classmethod
    def from_env(cls, task: str) -> TaskModelSettings:
        """Load task-specific settings.

        Supported tasks: ``claim``, ``entailment``.
        """
        prefix = f"CITEGUARD_{task.upper()}"
        provider = (
            os.getenv(f"{prefix}_PROVIDER", "").strip().lower() or None
        )
        model = os.getenv(f"{prefix}_MODEL", "").strip() or None
        return cls(provider=provider, model=model)


@dataclass(frozen=True, slots=True)
class Settings:
    anthropic_api_key: str | None
    semantic_scholar_api_key: str | None
    cache_dir: Path = DEFAULT_CACHE_DIR
    threshold: int = DEFAULT_THRESHOLD
    llm: LLMProviderSettings = field(default_factory=LLMProviderSettings.from_env)

    @classmethod
    def from_env(cls, cache_dir: Path | None = None) -> Settings:
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            cache_dir=cache_dir or DEFAULT_CACHE_DIR,
            llm=LLMProviderSettings.from_env(),
        )

    def require_llm_api_key(self) -> str:
        """Return the LLM API key or raise if unavailable."""
        if not self.llm.api_key:
            raise RuntimeError(
                f"No API key found for provider '{self.llm.provider}'. "
                f"Set the appropriate environment variable "
                f"(e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY). "
                f"Run `citeguard init` for a template."
            )
        return self.llm.api_key

    def require_anthropic(self) -> str:
        """Backward-compatible: require Anthropic key specifically."""
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required for claim extraction, matching, "
                "and web-search fallback. "
                "Run `citeguard init` or set the environment variable."
            )
        return self.anthropic_api_key
