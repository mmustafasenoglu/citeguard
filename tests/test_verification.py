from citeguard.bibliography import parse_bibliography_entry
from citeguard.models import SourceCandidate, VerificationStatus
from citeguard.retrieval import RetrievalEngine
from citeguard.verification import metadata_scores, verify_bibliography


class StubProvider:
    name = "crossref"

    def __init__(self, candidates):
        self.candidates = candidates

    def search(self, _query, max_results=5):
        return self.candidates[:max_results]


class DoiFallbackProvider:
    name = "crossref"

    def __init__(self, candidates):
        self.candidates = candidates
        self.queries = []

    def search(self, query, max_results=5):
        self.queries.append(query)
        if query.startswith("10."):
            return []
        return self.candidates[:max_results]


def candidate(**overrides) -> SourceCandidate:
    values = {
        "title": "A Useful Paper",
        "authors": ["Jane Smith"],
        "year": 2020,
        "venue": "Journal",
        "doi": "10.1000/example",
        "url": "https://doi.org/10.1000/example",
        "abstract": None,
        "source_api": "crossref",
    }
    values.update(overrides)
    return SourceCandidate(**values)


def test_exact_metadata_receives_full_score() -> None:
    entry = parse_bibliography_entry(
        "Smith, J. (2020). A Useful Paper. Journal. https://doi.org/10.1000/example"
    )

    scores = metadata_scores(entry, candidate())

    assert scores.author == 100
    assert scores.year == 100
    assert scores.title == 100
    assert scores.doi == 100
    assert scores.overall == 100


def test_doi_mismatch_caps_metadata_score() -> None:
    entry = parse_bibliography_entry(
        "Smith, J. (2020). A Useful Paper. Journal. https://doi.org/10.1000/example"
    )

    scores = metadata_scores(entry, candidate(doi="10.1000/different"))

    assert scores.doi == 0
    assert scores.overall == 40


def test_verification_statuses_are_deterministic() -> None:
    entry = parse_bibliography_entry("Smith, J. (2020). A Useful Paper.")
    verified = verify_bibliography(
        [entry], RetrievalEngine([StubProvider([candidate(doi=None)])])
    )
    missing = verify_bibliography([entry], RetrievalEngine([StubProvider([])]))

    assert verified[0].status == VerificationStatus.VERIFIED
    assert missing[0].status == VerificationStatus.NOT_FOUND


def test_missing_crossref_doi_falls_back_to_bibliographic_search() -> None:
    entry = parse_bibliography_entry(
        "Smith, J. (2020). A Useful Paper. https://doi.org/10.1000/missing"
    )
    provider = DoiFallbackProvider([candidate(doi=None)])

    result = verify_bibliography([entry], RetrievalEngine([provider]))[0]

    assert len(provider.queries) == 2
    assert provider.queries[0] == "10.1000/missing"
    assert provider.queries[1] == "A Useful Paper Smith 2020"
    assert result.status == VerificationStatus.VERIFIED
    assert "bibliographic search was used" in result.warnings[0]
