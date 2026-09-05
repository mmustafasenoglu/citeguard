from __future__ import annotations

import os
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from .. import __version__
from ..models import SourceCandidate
from .base import ProviderResponseError, fetch_json


class SemanticScholarProvider:
    name = "semantic_scholar"
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 10,
        opener: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.timeout = timeout
        self.opener = opener
        self.sleep = sleep

    def search(self, query: str, max_results: int = 5) -> list[SourceCandidate]:
        params = urllib.parse.urlencode(
            {
                "query": query,
                "limit": max_results,
                "fields": "title,authors,year,venue,externalIds,url,abstract",
            }
        )
        headers = {"Accept": "application/json", "User-Agent": f"citeguard/{__version__}"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        request = urllib.request.Request(f"{self.endpoint}?{params}", headers=headers)
        kwargs: dict[str, Any] = {"timeout": self.timeout, "opener": self.opener}
        if self.sleep is not None:
            kwargs["sleep"] = self.sleep
        payload = fetch_json(request, **kwargs)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ProviderResponseError("Semantic Scholar returned malformed data.")
        candidates: list[SourceCandidate] = []
        for item in payload["data"]:
            if not isinstance(item, dict) or not isinstance(item.get("title"), str):
                continue
            external_ids = item.get("externalIds") or {}
            if not isinstance(external_ids, dict):
                external_ids = {}
            raw_authors = item.get("authors") or []
            authors = [
                author["name"]
                for author in raw_authors
                if isinstance(author, dict) and isinstance(author.get("name"), str)
            ]
            candidates.append(
                SourceCandidate(
                    title=item["title"].strip(),
                    authors=authors,
                    year=item.get("year") if isinstance(item.get("year"), int) else None,
                    venue=item.get("venue") or None,
                    doi=external_ids.get("DOI"),
                    url=item.get("url") or None,
                    abstract=item.get("abstract") or None,
                    source_api=self.name,
                    arxiv_id=external_ids.get("ArXiv"),
                )
            )
        return candidates
