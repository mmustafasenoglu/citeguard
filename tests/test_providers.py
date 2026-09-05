import io
import json
import urllib.error

import pytest

from citeguard.providers.arxiv import ArxivProvider
from citeguard.providers.base import ProviderHTTPError, ProviderResponseError
from citeguard.providers.crossref import CrossrefProvider
from citeguard.providers.semantic_scholar import SemanticScholarProvider


class Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return self.body


def json_response(value) -> Response:
    return Response(json.dumps(value).encode())


def test_semantic_scholar_normalizes_response() -> None:
    requests = []

    def opener(request, **_kwargs):
        requests.append(request)
        return json_response(
            {
                "data": [
                    {
                        "title": " Attention Is All You Need ",
                        "authors": [{"name": "Ashish Vaswani"}],
                        "year": 2017,
                        "venue": "NeurIPS",
                        "externalIds": {"DOI": "10.1234/TEST", "ArXiv": "1706.03762"},
                        "url": "https://example.test/paper",
                        "abstract": "An abstract.",
                    }
                ]
            }
        )

    candidate = SemanticScholarProvider(api_key="secret", opener=opener).search("attention")[0]
    assert candidate.title == "Attention Is All You Need"
    assert candidate.authors == ["Ashish Vaswani"]
    assert candidate.doi == "10.1234/TEST"
    assert candidate.arxiv_id == "1706.03762"
    assert requests[0].get_header("X-api-key") == "secret"
    assert "secret" not in requests[0].full_url


def test_crossref_normalizes_response() -> None:
    provider = CrossrefProvider(
        opener=lambda *_args, **_kwargs: json_response(
            {
                "message": {
                    "items": [
                        {
                            "title": ["A useful paper"],
                            "author": [{"given": "Ada", "family": "Lovelace"}],
                            "published-online": {"date-parts": [[2020, 2, 1]]},
                            "container-title": ["Journal of Tests"],
                            "DOI": "10.1000/XYZ",
                            "URL": "https://doi.org/10.1000/XYZ",
                            "abstract": "<jats:p>Useful evidence.</jats:p>",
                        }
                    ]
                }
            }
        )
    )
    candidate = provider.search("useful")[0]
    assert candidate.authors == ["Ada Lovelace"]
    assert candidate.year == 2020
    assert candidate.venue == "Journal of Tests"
    assert candidate.abstract == "Useful evidence."


def test_crossref_uses_direct_lookup_for_doi() -> None:
    requests = []

    def opener(request, **_kwargs):
        requests.append(request)
        return json_response(
            {
                "message": {
                    "title": ["Exact DOI paper"],
                    "author": [{"given": "Ada", "family": "Lovelace"}],
                    "issued": {"date-parts": [[2020]]},
                    "DOI": "10.1000/EXACT",
                }
            }
        )

    candidate = CrossrefProvider(opener=opener).search("10.1000/exact")[0]

    assert candidate.title == "Exact DOI paper"
    assert requests[0].full_url.endswith("/10.1000%2Fexact")


def test_arxiv_parses_atom_xml() -> None:
    xml = b"""<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry><id>https://arxiv.org/abs/1706.03762v7</id>
      <published>2017-06-12T00:00:00Z</published><title>Attention Is All You Need</title>
      <summary>Transformer research.</summary><author><name>Ashish Vaswani</name></author>
      <arxiv:doi>10.5555/3295222.3295349</arxiv:doi>
      <arxiv:journal_ref>NeurIPS 2017</arxiv:journal_ref></entry></feed>"""
    candidate = ArxivProvider(opener=lambda *_args, **_kwargs: Response(xml)).search("attention")[0]
    assert candidate.arxiv_id == "1706.03762v7"
    assert candidate.year == 2017
    assert candidate.authors == ["Ashish Vaswani"]
    assert candidate.venue == "NeurIPS 2017"


def test_http_429_retries_with_exponential_backoff() -> None:
    calls = 0
    delays = []

    def opener(request, **_kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise urllib.error.HTTPError(request.full_url, 429, "limited", {}, io.BytesIO())
        return json_response({"data": []})

    assert SemanticScholarProvider(opener=opener, sleep=delays.append).search("query") == []
    assert calls == 3
    assert delays == [1, 2]


def test_http_retry_stops_after_three_attempts() -> None:
    calls = 0

    def opener(request, **_kwargs):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(request.full_url, 503, "down", {}, io.BytesIO())

    with pytest.raises(ProviderHTTPError):
        CrossrefProvider(opener=opener, sleep=lambda _delay: None).search("query")
    assert calls == 3


@pytest.mark.parametrize(
    ("provider", "response"),
    [
        (SemanticScholarProvider, json_response({"data": {}})),
        (CrossrefProvider, json_response({"message": {"items": {}}})),
        (ArxivProvider, Response(b"not xml")),
    ],
)
def test_malformed_provider_response_raises_provider_error(provider, response) -> None:
    with pytest.raises(ProviderResponseError):
        provider(opener=lambda *_args, **_kwargs: response).search("query")
