"""LLM response cache with prompt-version-aware invalidation.

Caches LLM responses keyed by provider + model + prompt_version +
hash(system + user_message).  Prevents redundant API calls for identical
inputs and automatically invalidates when prompt versions change.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from .llm_backends import LLMResponse


@dataclass(slots=True)
class _CacheEntry:
    """A single cached LLM response."""

    response: LLMResponse
    created_at: float
    prompt_version: str


class LLMCache:
    """In-process LLM response cache.

    Parameters
    ----------
    ttl_seconds:
        Entries older than this are evicted.  Default 1 hour.
    prompt_version:
        Automatically included in the cache key.  When the prompt
        changes (version bumped), old entries are ignored.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 3600.0,
        prompt_version: str = "1",
    ) -> None:
        self._ttl = ttl_seconds
        self._prompt_version = prompt_version
        self._store: dict[str, _CacheEntry] = {}

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _make_key(
        self,
        provider: str,
        model: str,
        system: str,
        user_message: str,
    ) -> str:
        prompt_hash = self._hash(system + "\n\n" + user_message)
        return f"{provider}:{model}:{self._prompt_version}:{prompt_hash}"

    def get(
        self,
        provider: str,
        model: str,
        system: str,
        user_message: str,
    ) -> LLMResponse | None:
        """Return cached response or ``None`` if miss / expired / version mismatch."""
        key = self._make_key(provider, model, system, user_message)
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.prompt_version != self._prompt_version:
            del self._store[key]
            return None
        if time.monotonic() - entry.created_at > self._ttl:
            del self._store[key]
            return None
        return entry.response

    def put(
        self,
        provider: str,
        model: str,
        system: str,
        user_message: str,
        response: LLMResponse,
    ) -> None:
        """Store a response in the cache."""
        if response.text is None:
            return
        key = self._make_key(provider, model, system, user_message)
        self._store[key] = _CacheEntry(
            response=response,
            created_at=time.monotonic(),
            prompt_version=self._prompt_version,
        )

    def has(
        self,
        provider: str,
        model: str,
        system: str,
        user_message: str,
    ) -> bool:
        return self.get(provider, model, system, user_message) is not None

    def clear(self) -> None:
        self._store.clear()

    def invalidate_version(self) -> None:
        """Remove all entries whose prompt_version does not match current."""
        stale = [
            k for k, v in self._store.items()
            if v.prompt_version != self._prompt_version
        ]
        for k in stale:
            del self._store[k]

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict[str, int]:
        return {
            "entries": self.size,
            "prompt_version": int(self._prompt_version),
        }
