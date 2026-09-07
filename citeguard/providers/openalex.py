from __future__ import annotations

import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from .. import __version__
from ..models import SourceCandidate
from .base import ProviderHTTPError, ProviderResponseError, fetch_json

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)


class OpenAlexProvider:
    name = "openalex"
    endpoint = "https://api.openalex.org/works"

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
        doi_match = _DOI_RE.search(query)
        if doi_match:
            doi = doi_match.group(0).rstrip(".,;)")
            url = f"{self.endpoint}/https://doi.org/{doi}"
        else:
            params = urllib.parse.urlencode({
                "search": query,
                "per_page": max_results,
            })
            url = f"{self.endpoint}?{params}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": f"citeguard/{__version__}",
            },
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
        if not isinstance(payload, dict):
            raise ProviderResponseError("OpenAlex returned malformed data.")
        results = payload.get("results")
        if doi_match:
            if not isinstance(payload, dict) or "id" not in payload:
                return []
            items = [payload]
        else:
            items = results if isinstance(results, list) else []
        candidates: list[SourceCandidate] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            if not isinstance(title, str) or not title.strip():
                continue
            authors: list[str] = []
            for authorship in item.get("authorships") or []:
                if not isinstance(authorship, dict):
                    continue
                author_obj = authorship.get("author")
                if isinstance(author_obj, dict):
                    display_name = author_obj.get("display_name")
                    if isinstance(display_name, str) and display_name.strip():
                        authors.append(display_name.strip())
            year = None
            primary_loc = item.get("primary_location")
            if isinstance(primary_loc, dict):
                source = primary_loc.get("source")
                if isinstance(source, dict):
                    pub_year = source.get("publication_year")
                    if isinstance(pub_year, int):
                        year = pub_year
            if year is None:
                biblio = item.get("biblio")
                if isinstance(biblio, dict):
                    by = biblio.get("year")
                    if isinstance(by, int):
                        year = by
            venue = None
            if isinstance(primary_loc, dict):
                source = primary_loc.get("source")
                if isinstance(source, dict):
                    venue_name = source.get("display_name")
                    if isinstance(venue_name, str) and venue_name.strip():
                        venue = venue_name.strip()
            doi_val = item.get("doi")
            if isinstance(doi_val, str):
                doi_val = doi_val.strip()
                if doi_val.startswith("https://doi.org/"):
                    doi_val = doi_val[len("https://doi.org/"):]
                if not _DOI_RE.search(doi_val):
                    doi_val = None
            else:
                doi_val = None
            url_val = item.get("id")
            if not isinstance(url_val, str):
                url_val = None
            abstract = None
            inv_index = item.get("abstract_inverted_index")
            if isinstance(inv_index, dict):
                abstract = _reconstruct_abstract(inv_index)
            candidates.append(
                SourceCandidate(
                    title=title.strip(),
                    authors=authors,
                    year=year,
                    venue=venue,
                    doi=doi_val,
                    url=url_val,
                    abstract=abstract,
                    source_api=self.name,
                )
            )
        return candidates


def _reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
    if not inverted_index:
        return ""
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)
