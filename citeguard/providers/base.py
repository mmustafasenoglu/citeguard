from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Protocol

from ..models import SourceCandidate

MAX_HTTP_ATTEMPTS = 3
RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}


class ProviderError(RuntimeError):
    """Base exception for an academic metadata provider failure."""


class ProviderHTTPError(ProviderError):
    """Raised when a provider request cannot be completed."""


class ProviderResponseError(ProviderError):
    """Raised when a provider returns an unusable response."""


class SourceProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int = 5) -> list[SourceCandidate]:
        """Return candidates for a query. Provider failures should degrade gracefully."""
        ...


def fetch_bytes(
    request: urllib.request.Request,
    *,
    timeout: float,
    opener: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = MAX_HTTP_ATTEMPTS,
) -> bytes:
    """Fetch a response, retrying rate limits and transient server failures."""
    open_request = opener or urllib.request.urlopen
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            with open_request(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in RETRYABLE_HTTP_STATUSES or attempt == attempts - 1:
                raise ProviderHTTPError(
                    f"Provider request failed with HTTP {exc.code} "
                    f"(after {attempt + 1} attempt(s))."
                ) from exc
            sleep(2**attempt)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt == attempts - 1:
                raise ProviderHTTPError(
                    f"Provider request failed: {exc}"
                ) from exc
            sleep(2**attempt)
    raise ProviderHTTPError("Provider request failed.") from last_exc


def fetch_json(
    request: urllib.request.Request,
    *,
    timeout: float,
    opener: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    try:
        return json.loads(
            fetch_bytes(request, timeout=timeout, opener=opener, sleep=sleep).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderResponseError("Provider returned malformed JSON.") from exc


def candidate_to_dict(candidate: SourceCandidate) -> dict[str, Any]:
    return asdict(candidate)


def candidate_from_dict(value: Any) -> SourceCandidate:
    if not isinstance(value, dict):
        raise ProviderResponseError("Cached provider result is malformed.")
    try:
        return SourceCandidate(
            title=str(value["title"]),
            authors=[str(author) for author in value.get("authors", [])],
            year=int(value["year"]) if value.get("year") is not None else None,
            venue=str(value["venue"]) if value.get("venue") else None,
            doi=str(value["doi"]) if value.get("doi") else None,
            url=str(value["url"]) if value.get("url") else None,
            abstract=str(value["abstract"]) if value.get("abstract") else None,
            source_api=str(value["source_api"]),
            arxiv_id=str(value["arxiv_id"]) if value.get("arxiv_id") else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderResponseError("Cached provider result is malformed.") from exc
