from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any

from .. import __version__
from ..models import SourceCandidate
from .base import ProviderResponseError, fetch_bytes

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"


class ArxivProvider:
    name = "arxiv"
    endpoint = "https://export.arxiv.org/api/query"

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
        params = urllib.parse.urlencode(
            {"search_query": f'all:"{query}"', "start": 0, "max_results": max_results}
        )
        request = urllib.request.Request(
            f"{self.endpoint}?{params}", headers={"User-Agent": f"citeguard/{__version__}"}
        )
        kwargs: dict[str, Any] = {"timeout": self.timeout, "opener": self.opener}
        if self.sleep is not None:
            kwargs["sleep"] = self.sleep
        try:
            root = ET.fromstring(fetch_bytes(request, **kwargs))
        except ET.ParseError as exc:
            raise ProviderResponseError("arXiv returned malformed XML.") from exc
        candidates: list[SourceCandidate] = []
        for entry in root.findall(f"{_ATOM}entry"):
            title = _text(entry, f"{_ATOM}title")
            if not title:
                continue
            identifier_url = _text(entry, f"{_ATOM}id")
            arxiv_id = identifier_url.rstrip("/").rsplit("/", 1)[-1] if identifier_url else None
            published = _text(entry, f"{_ATOM}published")
            year = int(published[:4]) if published and published[:4].isdigit() else None
            candidates.append(
                SourceCandidate(
                    title=" ".join(title.split()),
                    authors=[
                        name
                        for author in entry.findall(f"{_ATOM}author")
                        if (name := _text(author, f"{_ATOM}name"))
                    ],
                    year=year,
                    venue=_text(entry, f"{_ARXIV}journal_ref"),
                    doi=_text(entry, f"{_ARXIV}doi"),
                    url=identifier_url,
                    abstract=" ".join((_text(entry, f"{_ATOM}summary") or "").split()) or None,
                    source_api=self.name,
                    arxiv_id=arxiv_id,
                )
            )
        return candidates


def _text(element: ET.Element, path: str) -> str | None:
    child = element.find(path)
    if child is None or child.text is None:
        return None
    return child.text.strip() or None
