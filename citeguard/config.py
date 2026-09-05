"""Global defaults and environment-backed settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CACHE_SCHEMA_VERSION = "2"
CLAIM_PROMPT_VERSION = "1"
MATCHER_PROMPT_VERSION = "1"
LINKER_PROMPT_VERSION = "1"

DEFAULT_THRESHOLD = 60
STRONG_MATCH_THRESHOLD = 80
DEFAULT_MAX_RESULTS = 5
DEFAULT_CACHE_DIR = Path(".citeguard_cache")


@dataclass(frozen=True, slots=True)
class Settings:
    anthropic_api_key: str | None
    semantic_scholar_api_key: str | None
    cache_dir: Path = DEFAULT_CACHE_DIR
    threshold: int = DEFAULT_THRESHOLD

    @classmethod
    def from_env(cls, cache_dir: Path | None = None) -> Settings:
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            cache_dir=cache_dir or DEFAULT_CACHE_DIR,
        )

    def require_anthropic(self) -> str:
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required for claim extraction, matching, "
                "and web-search fallback. Run `citeguard init` or set the environment variable."
            )
        return self.anthropic_api_key
