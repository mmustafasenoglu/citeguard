from __future__ import annotations

import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from .. import __version__
from ..models import SourceCandidate
from .base import ProviderHTTPError, ProviderResponseError, fetch_json

_TAG_RE = re.compile(r"<[^>]+>")
_DOI_QUERY_RE = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)


class CrossrefProvider:
    name = "crossref"
    endpoint = "https://api.crossref.org/works"

    def __init__(
        self,
        *,
        timeout: float = 10,
        opener: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.timeout = timeout
        self.opener = opener
        self.sleep = sleep

    def search(self, query: str, max_results: int = 5) -> list[SourceCandidate]:
        doi_match = _DOI_QUERY_RE.search(query)
        if doi_match:
            doi = doi_match.group(0).rstrip(".,;)")
            url = f"{self.endpoint}/{urllib.parse.quote(doi, safe='')}"
        else:
            params = urllib.parse.urlencode({"query.bibliographic": query, "rows": max_results})
            url = f"{self.endpoint}?{params}"
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": f"citeguard/{__version__}"},
        )
        kwargs: dict[str, Any] = {"timeout": self.timeout, "opener": self.opener}
        if self.sleep is not None:
            kwargs["sleep"] = self.sleep
        try:
            payload = fetch_json(request, **kwargs)
        except ProviderHTTPError as exc:
            if doi_match and "HTTP status 404" in str(exc):
                return []
            raise
        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, dict):
            raise ProviderResponseError("Crossref returned malformed data.")
        items = [message] if doi_match else message.get("items")
        if not isinstance(items, list):
            raise ProviderResponseError("Crossref returned malformed data.")
        candidates: list[SourceCandidate] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            titles = item.get("title")
            if not isinstance(titles, list) or not titles or not isinstance(titles[0], str):
                continue
            authors = []
            for author in item.get("author") or []:
                if not isinstance(author, dict):
                    continue
                name = " ".join(
                    part for part in (author.get("given"), author.get("family")) if part
                )
                if name:
                    authors.append(name)
            year = _publication_year(item)
            container = item.get("container-title")
            venue = container[0] if isinstance(container, list) and container else None
            abstract = item.get("abstract")
            if isinstance(abstract, str):
                abstract = _TAG_RE.sub(" ", abstract).strip()
            candidates.append(
                SourceCandidate(
                    title=titles[0].strip(),
                    authors=authors,
                    year=year,
                    venue=venue,
                    doi=item.get("DOI") or None,
                    url=item.get("URL") or None,
                    abstract=abstract or None,
                    source_api=self.name,
                )
            )
        return candidates


def _publication_year(item: dict[str, Any]) -> int | None:
    for field in ("published-print", "published-online", "issued", "created"):
        value = item.get(field)
        parts = value.get("date-parts") if isinstance(value, dict) else None
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            year = parts[0][0]
            if isinstance(year, int):
                return year
    return None
