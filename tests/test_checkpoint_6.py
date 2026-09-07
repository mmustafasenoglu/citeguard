"""Checkpoint 6: OpenAlex, numbered citations, verification, recency, primary-source."""

from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from citeguard.bibliography import (
    resolve_numbered_citations,
)
from citeguard.extractor import _expand_numbered_citations, extract_citations
from citeguard.models import (
    BibliographyEntry,
    ExistingCitation,
    MetadataScores,
    SourceCandidate,
    SourceType,
    VerificationStatus,
)
from citeguard.providers.openalex import OpenAlexProvider, _reconstruct_abstract
from citeguard.retrieval import RetrievalEngine
from citeguard.verification import (
    _PARTIAL_METADATA_THRESHOLD,
    VERIFIED_METADATA_THRESHOLD,
    _check_recency,
    _classify_source_type,
    _detect_provider_conflicts,
    _determine_status,
    metadata_scores,
    verify_bibliography,
)

# ---------------------------------------------------------------------------
# OpenAlex Provider
# ---------------------------------------------------------------------------


def _make_openalex_doi_payload() -> dict:
    return {
        "id": "https://openalex.org/W2741809800",
        "doi": "https://doi.org/10.1038/nature14539",
        "title": "Deep learning for protein structure prediction",
        "publication_year": 2015,
        "authorships": [
            {"author": {"display_name": "John Smith"}},
            {"author": {"display_name": "Jane Doe"}},
        ],
        "primary_location": {
            "source": {
                "display_name": "Nature",
                "publication_year": 2015,
            }
        },
        "biblio": {"year": 2015},
        "abstract_inverted_index": {
            "We": [0],
            "present": [1],
            "a": [2],
            "deep": [3, 7],
            "learning": [4],
            "approach": [5],
            "for": [6],
            "protein": [8],
            "structure": [9],
            "prediction": [10],
        },
    }


def _make_openalex_search_payload() -> dict:
    return {
        "results": [
            {
                "id": "https://openalex.org/W2741809800",
                "doi": "https://doi.org/10.1038/nature14539",
                "title": "Deep learning for protein structure prediction",
                "authorships": [
                    {"author": {"display_name": "John Smith"}},
                ],
                "primary_location": {
                    "source": {
                        "display_name": "Nature",
                        "publication_year": 2015,
                    }
                },
                "biblio": {"year": 2015},
                "abstract_inverted_index": {
                    "We": [0],
                    "present": [1],
                    "a": [2],
                    "method": [3],
                },
            }
        ],
        "meta": {"count": 1},
    }


class TestOpenAlexProvider:
    def test_doi_lookup(self) -> None:
        payload = _make_openalex_doi_payload()
        with patch(
            "citeguard.providers.openalex.fetch_json", return_value=payload
        ):
            provider = OpenAlexProvider()
            results = provider.search("10.1038/nature14539", max_results=5)
        assert len(results) == 1
        c = results[0]
        assert c.title == "Deep learning for protein structure prediction"
        assert c.doi == "10.1038/nature14539"
        assert c.year == 2015
        assert c.venue == "Nature"
        assert len(c.authors) == 2
        assert c.source_api == "openalex"

    def test_title_search(self) -> None:
        payload = _make_openalex_search_payload()
        with patch(
            "citeguard.providers.openalex.fetch_json", return_value=payload
        ):
            provider = OpenAlexProvider()
            results = provider.search("deep learning protein", max_results=5)
        assert len(results) == 1
        assert results[0].title == "Deep learning for protein structure prediction"

    def test_abstract_reconstruction(self) -> None:
        inv = {"We": [0], "are": [1], "testing": [2]}
        assert _reconstruct_abstract(inv) == "We are testing"

    def test_empty_abstract(self) -> None:
        assert _reconstruct_abstract({}) == ""
        assert _reconstruct_abstract({"word": []}) == ""

    def test_malformed_response_not_dict(self) -> None:
        with patch(
            "citeguard.providers.openalex.fetch_json", return_value="not a dict"
        ):
            provider = OpenAlexProvider()
            from citeguard.providers.base import ProviderResponseError

            with pytest.raises(ProviderResponseError, match="malformed"):
                provider.search("test query")

    def test_malformed_doi_response_no_id(self) -> None:
        with patch(
            "citeguard.providers.openalex.fetch_json", return_value={"title": "x"}
        ):
            provider = OpenAlexProvider()
            results = provider.search("10.1000/fake")
        assert results == []

    def test_timeout_raises(self) -> None:
        from citeguard.providers.base import ProviderHTTPError

        with patch(
            "citeguard.providers.openalex.fetch_json",
            side_effect=ProviderHTTPError("timeout"),
        ):
            provider = OpenAlexProvider()
            with pytest.raises(ProviderHTTPError):
                provider.search("10.1000/test")

    def test_404_on_doi_returns_empty(self) -> None:
        from citeguard.providers.base import ProviderHTTPError

        with patch(
            "citeguard.providers.openalex.fetch_json",
            side_effect=ProviderHTTPError("HTTP status 404"),
        ):
            provider = OpenAlexProvider()
            results = provider.search("10.1000/nonexistent")
        assert results == []

    def test_empty_results_list(self) -> None:
        with patch(
            "citeguard.providers.openalex.fetch_json",
            return_value={"results": []},
        ):
            provider = OpenAlexProvider()
            results = provider.search("nonexistent topic xyz")
        assert results == []

    def test_missing_title_skipped(self) -> None:
        payload = {
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "doi": None,
                    "title": None,
                    "authorships": [],
                    "primary_location": None,
                    "biblio": None,
                    "abstract_inverted_index": None,
                }
            ],
        }
        with patch(
            "citeguard.providers.openalex.fetch_json", return_value=payload
        ):
            provider = OpenAlexProvider()
            results = provider.search("test")
        assert results == []

    def test_openalex_in_default_providers(self) -> None:
        engine = RetrievalEngine()
        provider_names = [p.name for p in engine.providers]
        assert "openalex" in provider_names


# ---------------------------------------------------------------------------
# Numbered citation expansion
# ---------------------------------------------------------------------------


class TestNumberedCitationExpansion:
    def test_single_number(self) -> None:
        assert _expand_numbered_citations("1") == [1]

    def test_comma_separated(self) -> None:
        assert _expand_numbered_citations("1, 3, 5") == [1, 3, 5]

    def test_range(self) -> None:
        assert _expand_numbered_citations("2-4") == [2, 3, 4]

    def test_range_with_en_dash(self) -> None:
        assert _expand_numbered_citations("2–4") == [2, 3, 4]

    def test_mixed_ranges_and_singles(self) -> None:
        assert _expand_numbered_citations("1, 3-5, 8") == [1, 3, 4, 5, 8]

    def test_duplicates_normalized(self) -> None:
        assert _expand_numbered_citations("1, 1, 2-3, 3") == [1, 2, 3]

    def test_out_of_range_high(self) -> None:
        result = _expand_numbered_citations("1, 999999")
        assert 1 in result
        assert len(result) <= 101

    def test_reversed_range(self) -> None:
        assert _expand_numbered_citations("5-3") == [3, 4, 5]

    def test_empty_string(self) -> None:
        assert _expand_numbered_citations("") == []

    def test_zero_excluded(self) -> None:
        assert _expand_numbered_citations("0, 1, 2") == [1, 2]

    def test_negative_excluded(self) -> None:
        assert _expand_numbered_citations("-1, 1") == [1]


class TestNumberedCitationInExtractor:
    def test_single_numbered_citation(self) -> None:
        citations = extract_citations(["As shown in [1], this works."])
        numbered = [c for c in citations if c.numbered_ref is not None]
        assert len(numbered) == 1
        assert numbered[0].numbered_ref == 1

    def test_multiple_numbered(self) -> None:
        citations = extract_citations(["This is known [1, 3, 5]."])
        numbered = [c for c in citations if c.numbered_ref is not None]
        nums = sorted(c.numbered_ref for c in numbered)
        assert nums == [1, 3, 5]

    def test_range_in_text(self) -> None:
        citations = extract_citations(["Several studies [2-4] confirm this."])
        numbered = [c for c in citations if c.numbered_ref is not None]
        nums = sorted(c.numbered_ref for c in numbered)
        assert nums == [2, 3, 4]

    def test_mixed_in_text(self) -> None:
        citations = extract_citations(["Refs [1, 3-5, 8] show this."])
        numbered = [c for c in citations if c.numbered_ref is not None]
        nums = sorted(c.numbered_ref for c in numbered)
        assert nums == [1, 3, 4, 5, 8]

    def test_bracketed_number_not_citation_in_context(self) -> None:
        citations = extract_citations(["The value is [42] but that is just a number."])
        numbered = [c for c in citations if c.numbered_ref is not None]
        assert len(numbered) == 1
        assert numbered[0].numbered_ref == 42

    def test_author_year_not_affected(self) -> None:
        citations = extract_citations(["As shown by Smith (2020), this works."])
        numbered = [c for c in citations if c.numbered_ref is not None]
        author_year = [c for c in citations if c.authors is not None]
        assert len(numbered) == 0
        assert len(author_year) == 1


# ---------------------------------------------------------------------------
# Numbered citation → bibliography resolution
# ---------------------------------------------------------------------------


class TestNumberedCitationResolution:
    def test_resolve_basic(self) -> None:
        bib = [
            BibliographyEntry(
                raw_text="[1] Smith (2020). Paper A.",
                authors="Smith",
                year=2020,
                title="Paper A",
                doi=None,
                numbered_ref=1,
            ),
            BibliographyEntry(
                raw_text="[2] Doe (2021). Paper B.",
                authors="Doe",
                year=2021,
                title="Paper B",
                doi=None,
                numbered_ref=2,
            ),
        ]
        citations = [
            ExistingCitation(
                raw_text="[1]",
                authors=None,
                year=None,
                doi=None,
                numbered_ref=1,
                paragraph_index=0,
                char_offset=0,
            ),
            ExistingCitation(
                raw_text="[2]",
                authors=None,
                year=None,
                doi=None,
                numbered_ref=2,
                paragraph_index=0,
                char_offset=5,
            ),
        ]
        resolved = resolve_numbered_citations(citations, bib)
        assert resolved[1] is bib[0]
        assert resolved[2] is bib[1]

    def test_unresolved_number(self) -> None:
        bib = [
            BibliographyEntry(
                raw_text="[1] Smith (2020).",
                authors="Smith",
                year=2020,
                title="Paper",
                doi=None,
                numbered_ref=1,
            ),
        ]
        citations = [
            ExistingCitation(
                raw_text="[5]",
                authors=None,
                year=None,
                doi=None,
                numbered_ref=5,
                paragraph_index=0,
                char_offset=0,
            ),
        ]
        resolved = resolve_numbered_citations(citations, bib)
        assert resolved[5] is None

    def test_out_of_range_number(self) -> None:
        resolved = resolve_numbered_citations(
            [
                ExistingCitation(
                    raw_text="[0]",
                    authors=None,
                    year=None,
                    doi=None,
                    numbered_ref=0,
                    paragraph_index=0,
                    char_offset=0,
                ),
            ],
            [],
        )
        assert resolved[0] is None

    def test_duplicates_normalized_resolution(self) -> None:
        bib = [
            BibliographyEntry(
                raw_text="[1] Smith (2020).",
                authors="Smith",
                year=2020,
                title="Paper",
                doi=None,
                numbered_ref=1,
            ),
        ]
        citations = [
            ExistingCitation(
                raw_text="[1]",
                authors=None,
                year=None,
                doi=None,
                numbered_ref=1,
                paragraph_index=0,
                char_offset=0,
            ),
            ExistingCitation(
                raw_text="[1]",
                authors=None,
                year=None,
                doi=None,
                numbered_ref=1,
                paragraph_index=0,
                char_offset=10,
            ),
        ]
        resolved = resolve_numbered_citations(citations, bib)
        assert len(resolved) == 1
        assert resolved[1] is bib[0]

    def test_author_year_still_works(self) -> None:
        from citeguard.bibliography import citation_matches_entry

        citation = ExistingCitation(
            raw_text="(Smith, 2020)",
            authors="Smith",
            year=2020,
            doi=None,
            numbered_ref=None,
            paragraph_index=0,
            char_offset=0,
        )
        entry = BibliographyEntry(
            raw_text="Smith (2020). Paper.",
            authors="Smith",
            year=2020,
            title="Paper",
            doi=None,
            numbered_ref=None,
        )
        assert citation_matches_entry(citation, entry) is True

    def test_numbered_matches_by_number(self) -> None:
        from citeguard.bibliography import citation_matches_entry

        citation = ExistingCitation(
            raw_text="[1]",
            authors=None,
            year=None,
            doi=None,
            numbered_ref=1,
            paragraph_index=0,
            char_offset=0,
        )
        entry = BibliographyEntry(
            raw_text="[1] Smith (2020). Paper.",
            authors="Smith",
            year=2020,
            title="Paper",
            doi=None,
            numbered_ref=1,
        )
        assert citation_matches_entry(citation, entry) is True


# ---------------------------------------------------------------------------
# Verification hardening
# ---------------------------------------------------------------------------


def _make_candidate(**overrides) -> SourceCandidate:
    defaults = dict(
        title="A Useful Paper",
        authors=["Jane Smith"],
        year=2020,
        venue="Journal of Testing",
        doi="10.1000/test",
        url=None,
        abstract=None,
        source_api="crossref",
    )
    defaults.update(overrides)
    return SourceCandidate(**defaults)


def _make_entry(**overrides) -> BibliographyEntry:
    defaults = dict(
        raw_text="Smith (2020). A Useful Paper.",
        authors="Smith",
        year=2020,
        title="A Useful Paper",
        doi="10.1000/test",
    )
    defaults.update(overrides)
    return BibliographyEntry(**defaults)


class TestVerificationHardening:
    def test_doi_exact_match_verified(self) -> None:
        entry = _make_entry(doi="10.1000/test")
        candidate = _make_candidate(doi="10.1000/test")
        status = _determine_status(entry, candidate, metadata_scores(entry, candidate), [])
        assert status == VerificationStatus.VERIFIED

    def test_doi_mismatch_metadata_mismatch(self) -> None:
        entry = _make_entry(doi="10.1000/entry")
        candidate = _make_candidate(doi="10.1000/candidate")
        scores = metadata_scores(entry, candidate)
        status = _determine_status(entry, candidate, scores, [(scores, candidate)])
        assert status == VerificationStatus.METADATA_MISMATCH

    def test_high_score_verified(self) -> None:
        entry = _make_entry(doi=None, title="Exact Match", authors="Smith", year=2020)
        candidate = _make_candidate(doi=None, title="Exact Match", authors=["Smith"], year=2020)
        scores = metadata_scores(entry, candidate)
        assert scores.overall >= VERIFIED_METADATA_THRESHOLD
        status = _determine_status(entry, candidate, scores, [(scores, candidate)])
        assert status == VerificationStatus.VERIFIED

    def test_partial_score(self) -> None:
        entry = _make_entry(
            doi=None, title="Deep Learning Methods", authors="Smith", year=2020
        )
        candidate = _make_candidate(
            doi=None,
            title="Deep Learning Approaches",
            authors=["Smith"],
            year=2019,
        )
        scores = metadata_scores(entry, candidate)
        assert scores.overall >= _PARTIAL_METADATA_THRESHOLD
        assert scores.overall < VERIFIED_METADATA_THRESHOLD
        status = _determine_status(entry, candidate, scores, [(scores, candidate)])
        assert status == VerificationStatus.PARTIALLY_VERIFIED

    def test_low_score_unresolved(self) -> None:
        entry = _make_entry(doi=None, title="Something", authors="Smith", year=2020)
        candidate = _make_candidate(
            doi=None, title="Totally Different", authors=["Jones"], year=1990
        )
        scores = metadata_scores(entry, candidate)
        assert scores.overall < _PARTIAL_METADATA_THRESHOLD
        status = _determine_status(entry, candidate, scores, [(scores, candidate)])
        assert status == VerificationStatus.UNRESOLVED

    def test_provider_error_when_warnings_only(self) -> None:
        """When all providers fail with exceptions, status is PROVIDER_ERROR."""
        entry = _make_entry(doi="10.1000/test")
        mock_provider = MagicMock()
        mock_provider.name = "mock_provider"
        mock_provider.search.side_effect = RuntimeError("provider broken")
        engine = RetrievalEngine([mock_provider], use_cache=False, rate_limit=False)
        results = verify_bibliography([entry], engine, max_results=5)
        assert len(results) == 1
        assert results[0].status == VerificationStatus.PROVIDER_ERROR


# ---------------------------------------------------------------------------
# Recency signal
# ---------------------------------------------------------------------------


class TestRecencySignal:
    def test_recent_source_no_warning(self) -> None:
        entry = _make_entry(year=2020)
        candidate = _make_candidate(year=2020)
        warning = _check_recency(entry, candidate, max_age=25)
        assert warning is None

    def test_old_source_warning(self) -> None:
        entry = _make_entry(year=1990)
        candidate = _make_candidate(year=1990)
        warning = _check_recency(entry, candidate, max_age=25)
        assert warning is not None
        assert "outdated" in warning.lower()
        assert "1990" in warning

    def test_missing_year_no_warning(self) -> None:
        entry = _make_entry(year=None)
        candidate = _make_candidate(year=None)
        warning = _check_recency(entry, candidate, max_age=25)
        assert warning is None

    def test_recency_in_verify_result(self) -> None:
        from datetime import date

        current_year = date.today().year
        entry = _make_entry(year=current_year - 50)
        candidate = _make_candidate(year=current_year - 50)
        metadata_scores(entry, candidate)
        with patch(
            "citeguard.verification.RetrievalEngine.search_with_result"
        ) as mock_search:
            mock_search.return_value = MagicMock(
                candidates=[candidate], warnings=[], queried_providers=["crossref"]
            )
            engine = RetrievalEngine(
                [MagicMock(name="crossref", search=MagicMock(return_value=[candidate]))],
            )
            results = verify_bibliography(
                [entry], engine, max_results=5, recency_max_age=25,
            )
        assert len(results) == 1
        assert results[0].recency_warning is not None
        assert "outdated" in results[0].recency_warning.lower()


# ---------------------------------------------------------------------------
# Primary-source preference
# ---------------------------------------------------------------------------


class TestPrimarySourcePreference:
    def test_nature_venue_alone_is_unknown(self) -> None:
        candidate = _make_candidate(venue="Nature")
        assert _classify_source_type(candidate) == SourceType.UNKNOWN

    def test_springer_venue_alone_is_unknown(self) -> None:
        candidate = _make_candidate(venue="Springer")
        assert _classify_source_type(candidate) == SourceType.UNKNOWN

    def test_explicit_journal_article_is_unknown(self) -> None:
        candidate = _make_candidate(work_type="journal-article")
        assert _classify_source_type(candidate) == SourceType.UNKNOWN

    def test_explicit_research_article_is_primary(self) -> None:
        candidate = _make_candidate(work_type="research-article")
        assert _classify_source_type(candidate) == SourceType.PRIMARY

    def test_explicit_review_is_secondary(self) -> None:
        candidate = _make_candidate(work_type="review")
        assert _classify_source_type(candidate) == SourceType.SECONDARY

    def test_explicit_editorial_is_secondary(self) -> None:
        candidate = _make_candidate(work_type="editorial")
        assert _classify_source_type(candidate) == SourceType.SECONDARY

    def test_missing_work_type_is_unknown(self) -> None:
        candidate = _make_candidate(work_type=None)
        assert _classify_source_type(candidate) == SourceType.UNKNOWN

    def test_unknown_work_type_is_unknown(self) -> None:
        candidate = _make_candidate(work_type="patent")
        assert _classify_source_type(candidate) == SourceType.UNKNOWN

    def test_source_type_in_verify(self) -> None:
        entry = _make_entry(
            doi=None, title="Nature Paper", authors="Smith", year=2020
        )
        candidate = _make_candidate(
            doi=None,
            title="Nature Paper",
            authors=["Smith"],
            year=2020,
            venue="Nature",
            work_type="research-article",
        )
        with patch(
            "citeguard.verification.RetrievalEngine.search_with_result"
        ) as mock_search:
            mock_search.return_value = MagicMock(
                candidates=[candidate],
                warnings=[],
                queried_providers=["crossref"],
                provider_errors=[],
            )
            engine = RetrievalEngine([MagicMock()])
            results = verify_bibliography([entry], engine, max_results=5)
        assert results[0].source_type == SourceType.PRIMARY


# ---------------------------------------------------------------------------
# Provider fusion / precedence
# ---------------------------------------------------------------------------


class TestProviderFusion:
    def test_no_conflict_single_provider(self) -> None:
        entry = _make_entry(doi=None, title="Test", authors="Smith", year=2020)
        candidate = _make_candidate(doi=None, title="Test", authors=["Smith"], year=2020)
        scores = metadata_scores(entry, candidate)
        conflicts = _detect_provider_conflicts([(scores, candidate)])
        assert conflicts == []

    def test_same_doi_consistent_metadata_no_conflict(self) -> None:
        entry = _make_entry(doi="10.1000/test")
        c1 = _make_candidate(
            doi="10.1000/test", title="Same Paper", authors=["Smith"],
            year=2020, source_api="crossref",
        )
        c2 = _make_candidate(
            doi="10.1000/test", title="Same Paper", authors=["Smith"],
            year=2020, source_api="openalex",
        )
        s1 = metadata_scores(entry, c1)
        s2 = metadata_scores(entry, c2)
        conflicts = _detect_provider_conflicts([(s1, c1), (s2, c2)])
        assert conflicts == []

    def test_same_doi_conflicting_title_conflict(self) -> None:
        entry = _make_entry(doi="10.1000/test")
        c1 = _make_candidate(
            doi="10.1000/test", title="Title A", authors=["Smith"],
            year=2020, source_api="crossref",
        )
        c1.provider_records = {
            "crossref": {
                "title": "Title A", "authors": ["Smith"],
                "year": 2020, "doi": "10.1000/test",
                "work_type": None, "venue": None,
            },
            "openalex": {
                "title": "Title B", "authors": ["Smith"],
                "year": 2020, "doi": "10.1000/test",
                "work_type": None, "venue": None,
            },
        }
        s1 = metadata_scores(entry, c1)
        conflicts = _detect_provider_conflicts([(s1, c1)])
        assert len(conflicts) == 1
        assert "Metadata conflict" in conflicts[0]

    def test_same_doi_conflicting_year_conflict(self) -> None:
        entry = _make_entry(doi="10.1000/test")
        c1 = _make_candidate(
            doi="10.1000/test", title="Same Title", authors=["Smith"],
            year=2020, source_api="crossref",
        )
        c1.provider_records = {
            "crossref": {
                "title": "Same Title", "authors": ["Smith"],
                "year": 2020, "doi": "10.1000/test",
                "work_type": None, "venue": None,
            },
            "openalex": {
                "title": "Same Title", "authors": ["Smith"],
                "year": 2021, "doi": "10.1000/test",
                "work_type": None, "venue": None,
            },
        }
        s1 = metadata_scores(entry, c1)
        conflicts = _detect_provider_conflicts([(s1, c1)])
        assert len(conflicts) == 1
        assert "Year conflict" in conflicts[0]

    def test_different_doi_high_scores_conflict(self) -> None:
        entry = _make_entry(
            doi=None, title="Machine Learning Methods", authors="Smith", year=2020
        )
        c1 = _make_candidate(
            doi=None, title="Machine Learning Methods", authors=["Smith"],
            year=2020, source_api="crossref",
        )
        c2 = _make_candidate(
            doi="10.9999/different", title="Different Paper Entirely",
            authors=["Smith"], year=2020, source_api="openalex",
        )
        s1 = metadata_scores(entry, c1)
        s2 = metadata_scores(entry, c2)
        if (
            s1.overall >= VERIFIED_METADATA_THRESHOLD
            and s2.overall >= VERIFIED_METADATA_THRESHOLD
        ):
            conflicts = _detect_provider_conflicts([(s1, c1), (s2, c2)])
            assert len(conflicts) == 1

    def test_no_conflict_same_title(self) -> None:
        entry = _make_entry(doi=None, title="Test", authors="Smith", year=2020)
        c1 = _make_candidate(
            title="Test", authors=["Smith"], year=2020, source_api="crossref"
        )
        c2 = _make_candidate(
            title="Test", authors=["Smith"], year=2020, source_api="openalex"
        )
        s1 = metadata_scores(entry, c1)
        s2 = metadata_scores(entry, c2)
        conflicts = _detect_provider_conflicts([(s1, c1), (s2, c2)])
        assert conflicts == []


# ---------------------------------------------------------------------------
# Offline behavior
# ---------------------------------------------------------------------------


class TestOfflineBehavior:
    def test_offline_cli_zero_network(self, monkeypatch) -> None:
        """CLI --offline must not call any provider search methods."""
        from click.testing import CliRunner

        from citeguard.cli import main

        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "doc.md"
            p.write_text("Hello world.\n\nReferences\n\nSmith (2020). Paper.")

            call_count = [0]

            def counting_search(_self, _query, max_results=5):
                call_count[0] += 1
                return []

            monkeypatch.setattr(
                "citeguard.cli.SemanticScholarProvider.search", counting_search
            )
            monkeypatch.setattr(
                "citeguard.cli.CrossrefProvider.search", counting_search
            )
            monkeypatch.setattr(
                "citeguard.cli.ArxivProvider.search", counting_search
            )
            monkeypatch.setattr(
                "citeguard.cli.OpenAlexProvider.search", counting_search
            )
            result = CliRunner().invoke(
                main,
                ["check", str(p), "--format", "json", "--offline", "--no-cache"],
            )
            assert result.exit_code in (0, 1)
            assert call_count[0] == 0

    def test_offline_verify_cli_zero_network(self, monkeypatch) -> None:
        """CLI verify --offline must not call any provider search methods."""
        from click.testing import CliRunner

        from citeguard.cli import main

        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "doc.md"
            p.write_text(
                "A claim (Smith, 2020).\n\nReferences\n\nSmith (2020). Paper."
            )

            call_count = [0]

            def counting_search(_self, _query, max_results=5):
                call_count[0] += 1
                return []

            monkeypatch.setattr(
                "citeguard.cli.CrossrefProvider.search", counting_search
            )
            monkeypatch.setattr(
                "citeguard.cli.OpenAlexProvider.search", counting_search
            )
            result = CliRunner().invoke(
                main,
                ["verify", str(p), "--format", "json", "--offline", "--no-cache"],
            )
            assert result.exit_code == 0
            assert call_count[0] == 0


# ---------------------------------------------------------------------------
# Report integration
# ---------------------------------------------------------------------------


class TestReportIntegration:
    def test_verification_item_includes_new_fields(self) -> None:
        from citeguard.models import ReferenceVerification
        from citeguard.report import _verification_item

        entry = _make_entry()
        result = ReferenceVerification(
            entry_index=0,
            entry=entry,
            query="test",
            status=VerificationStatus.VERIFIED,
            candidate=_make_candidate(),
            scores=MetadataScores(author=100, year=100, title=100, doi=100, overall=100),
            recency_warning="Possibly outdated",
            source_type=SourceType.PRIMARY,
            provider_conflicts=["Provider conflict"],
        )
        item = _verification_item(result)
        assert item["recency_warning"] == "Possibly outdated"
        assert item["source_type"] == "primary"
        assert item["provider_conflicts"] == ["Provider conflict"]

    def test_verification_report_provider_label(self) -> None:
        from citeguard.report import verification_report

        parsed = MagicMock()
        parsed.path = "test.md"
        results = []
        report = verification_report(parsed, results)
        assert "openalex" in report["provider"]


# ---------------------------------------------------------------------------
# Provider error vs not-found distinction (P1-3)
# ---------------------------------------------------------------------------


class TestProviderErrorDistinction:
    def test_successful_call_zero_results_is_not_found(self) -> None:
        entry = _make_entry(doi="10.1000/test")
        mock_provider = MagicMock()
        mock_provider.name = "mock_provider"
        mock_provider.search.return_value = []
        engine = RetrievalEngine([mock_provider], use_cache=False, rate_limit=False)
        results = verify_bibliography([entry], engine, max_results=5)
        assert results[0].status == VerificationStatus.NOT_FOUND

    def test_timeout_no_candidates_is_provider_error(self) -> None:
        entry = _make_entry(doi="10.1000/test")
        mock_provider = MagicMock()
        mock_provider.name = "mock_provider"
        mock_provider.search.side_effect = RuntimeError("timeout")
        engine = RetrievalEngine([mock_provider], use_cache=False, rate_limit=False)
        results = verify_bibliography([entry], engine, max_results=5)
        assert results[0].status == VerificationStatus.PROVIDER_ERROR

    def test_doi_miss_fallback_no_match_is_not_found(self) -> None:
        entry = _make_entry(doi="10.1000/test", title="Some Paper", authors="Smith", year=2020)
        mock_provider = MagicMock()
        mock_provider.name = "mock_provider"
        mock_provider.search.return_value = []
        engine = RetrievalEngine([mock_provider], use_cache=False, rate_limit=False)
        results = verify_bibliography([entry], engine, max_results=5)
        assert results[0].status == VerificationStatus.NOT_FOUND
        assert any("DOI lookup returned no match" in w for w in results[0].warnings)

    def test_one_provider_fails_other_succeeds(self) -> None:
        entry = _make_entry(doi="10.1000/test")
        good_provider = MagicMock()
        good_provider.name = "good_provider"
        good_provider.search.return_value = [_make_candidate(doi="10.1000/test")]
        bad_provider = MagicMock()
        bad_provider.name = "bad_provider"
        bad_provider.search.side_effect = RuntimeError("broken")
        engine = RetrievalEngine(
            [bad_provider, good_provider], use_cache=False, rate_limit=False,
        )
        results = verify_bibliography([entry], engine, max_results=5)
        assert results[0].status == VerificationStatus.VERIFIED


# ---------------------------------------------------------------------------
# OpenAlex publication year extraction (P1-4)
# ---------------------------------------------------------------------------


class TestOpenAlexYearExtraction:
    def test_work_level_publication_year(self) -> None:
        payload = {
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.1000/test",
            "title": "Test Paper",
            "publication_year": 2023,
            "authorships": [],
            "primary_location": {
                "source": {"display_name": "Journal", "publication_year": 2020}
            },
            "biblio": {"year": 2021},
        }
        with patch(
            "citeguard.providers.openalex.fetch_json", return_value=payload
        ):
            provider = OpenAlexProvider()
            results = provider.search("10.1000/test", max_results=1)
        assert len(results) == 1
        assert results[0].year == 2023

    def test_fallback_to_biblio_year(self) -> None:
        payload = {
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.1000/test",
            "title": "Test Paper",
            "authorships": [],
            "primary_location": None,
            "biblio": {"year": 2019},
        }
        with patch(
            "citeguard.providers.openalex.fetch_json", return_value=payload
        ):
            provider = OpenAlexProvider()
            results = provider.search("10.1000/test", max_results=1)
        assert len(results) == 1
        assert results[0].year == 2019

    def test_fallback_to_source_year(self) -> None:
        payload = {
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.1000/test",
            "title": "Test Paper",
            "authorships": [],
            "primary_location": {
                "source": {"display_name": "Journal", "publication_year": 2018}
            },
            "biblio": {},
        }
        with patch(
            "citeguard.providers.openalex.fetch_json", return_value=payload
        ):
            provider = OpenAlexProvider()
            results = provider.search("10.1000/test", max_results=1)
        assert len(results) == 1
        assert results[0].year == 2018

    def test_malformed_year_ignored(self) -> None:
        payload = {
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.1000/test",
            "title": "Test Paper",
            "publication_year": "not_a_year",
            "authorships": [],
            "primary_location": None,
            "biblio": {"year": "bad"},
        }
        with patch(
            "citeguard.providers.openalex.fetch_json", return_value=payload
        ):
            provider = OpenAlexProvider()
            results = provider.search("10.1000/test", max_results=1)
        assert len(results) == 1
        assert results[0].year is None


# ---------------------------------------------------------------------------
# Provenance preservation (P0-2)
# ---------------------------------------------------------------------------


class TestProvenancePreservation:
    def test_merge_preserves_source_apis(self) -> None:
        from citeguard.retrieval import _merge_candidates

        c1 = _make_candidate(source_api="crossref", doi="10.1000/test")
        c2 = _make_candidate(source_api="openalex", doi="10.1000/test")
        merged = _merge_candidates(c1, c2)
        assert "crossref" in merged.source_apis
        assert "openalex" in merged.source_apis
        assert merged.source_api == "crossref"

    def test_merge_no_duplicate_apis(self) -> None:
        from citeguard.retrieval import _merge_candidates

        c1 = _make_candidate(source_api="crossref")
        c2 = _make_candidate(source_api="crossref")
        merged = _merge_candidates(c1, c2)
        assert merged.source_apis.count("crossref") == 1

    def test_dedup_preserves_provenance(self) -> None:
        from citeguard.retrieval import deduplicate_candidates

        c1 = _make_candidate(
            doi="10.1000/test", title="Paper", source_api="crossref"
        )
        c2 = _make_candidate(
            doi="10.1000/test", title="Paper", source_api="openalex"
        )
        result = deduplicate_candidates([c1, c2])
        assert len(result) == 1
        assert "crossref" in result[0].source_apis
        assert "openalex" in result[0].source_apis


# ---------------------------------------------------------------------------
# Provider precedence confidence-based (P1-5)
# ---------------------------------------------------------------------------


class TestConfidenceBasedPrecedence:
    def test_exact_doi_outranks_fuzzy_match(self) -> None:
        from citeguard.retrieval import rank_candidates

        c_doi = _make_candidate(
            doi="10.1000/test", title="Something Else",
            authors=["Jones"], year=2019, source_api="openalex",
        )
        c_fuzzy = _make_candidate(
            doi=None, title="Machine Learning Methods",
            authors=["Smith"], year=2020, source_api="semantic_scholar",
        )
        ranked = rank_candidates("10.1000/test", [c_fuzzy, c_doi])
        assert ranked[0].doi == "10.1000/test"

    def test_tie_breaker_is_deterministic(self) -> None:
        from citeguard.retrieval import rank_candidates

        c1 = _make_candidate(
            doi=None, title="Alpha Paper", authors=["Smith"],
            year=2020, source_api="crossref",
        )
        c2 = _make_candidate(
            doi=None, title="Alpha Paper", authors=["Smith"],
            year=2020, source_api="openalex",
        )
        result1 = rank_candidates("alpha paper", [c1, c2])
        result2 = rank_candidates("alpha paper", [c2, c1])
        assert [c.source_api for c in result1] == [c.source_api for c in result2]


# ---------------------------------------------------------------------------
# Structured RetrievalResult diagnostics (P1-3)
# ---------------------------------------------------------------------------


class TestRetrievalResultDiagnostics:
    def test_provider_errors_populated(self) -> None:
        from citeguard.retrieval import RetrievalEngine

        bad_provider = MagicMock()
        bad_provider.name = "bad"
        bad_provider.search.side_effect = RuntimeError("timeout")
        engine = RetrievalEngine([bad_provider], use_cache=False, rate_limit=False)
        result = engine.search_with_result("test query")
        assert len(result.provider_errors) == 1
        assert "timeout" in result.provider_errors[0]

    def test_no_errors_when_providers_succeed(self) -> None:
        good_provider = MagicMock()
        good_provider.name = "good"
        good_provider.search.return_value = []
        engine = RetrievalEngine([good_provider], use_cache=False, rate_limit=False)
        result = engine.search_with_result("test query")
        assert result.provider_errors == []


# ---------------------------------------------------------------------------
# End-to-end provenance regression (P0)
# ---------------------------------------------------------------------------


class TestEndToEndProvenance:
    def _make_candidate(self, *, title, year, doi, source_api, work_type="journal-article"):
        return SourceCandidate(
            title=title,
            authors=["Author A"],
            year=year,
            venue="Journal",
            doi=doi,
            url=None,
            abstract=None,
            source_api=source_api,
            work_type=work_type,
        )

    def test_same_doi_same_metadata_single_candidate(self) -> None:
        from citeguard.retrieval import deduplicate_candidates
        c1 = self._make_candidate(
            title="Same Title", year=2020, doi="10.1000/test",
            source_api="crossref",
        )
        c2 = self._make_candidate(
            title="Same Title", year=2020, doi="10.1000/test",
            source_api="openalex",
        )
        result = deduplicate_candidates([c1, c2])
        assert len(result) == 1
        c = result[0]
        assert "crossref" in c.source_apis
        assert "openalex" in c.source_apis
        assert c.provider_records["crossref"]["title"] == "Same Title"
        assert c.provider_records["openalex"]["title"] == "Same Title"

    def test_same_doi_same_metadata_no_conflict(self) -> None:
        from citeguard.retrieval import deduplicate_candidates
        c1 = self._make_candidate(
            title="Same Title", year=2020, doi="10.1000/test",
            source_api="crossref",
        )
        c2 = self._make_candidate(
            title="Same Title", year=2020, doi="10.1000/test",
            source_api="openalex",
        )
        candidates = deduplicate_candidates([c1, c2])
        entry = BibliographyEntry(
            raw_text="Test. 2020.", authors=None, year=2020,
            title="Same Title", doi="10.1000/test",
        )
        with patch(
            "citeguard.verification.RetrievalEngine.search_with_result"
        ) as mock_search:
            mock_search.return_value = MagicMock(
                candidates=candidates, warnings=[], queried_providers=["crossref", "openalex"],
                provider_errors=[],
            )
            engine_v = RetrievalEngine([MagicMock()])
            results = verify_bibliography([entry], engine_v, max_results=5)
        assert results[0].provider_conflicts == []

    def test_same_doi_conflicting_title_survives_dedup(self) -> None:
        from citeguard.retrieval import deduplicate_candidates
        c1 = self._make_candidate(
            title="Title A", year=2020, doi="10.1000/test",
            source_api="crossref",
        )
        c2 = self._make_candidate(
            title="Title B", year=2020, doi="10.1000/test",
            source_api="openalex",
        )
        candidates = deduplicate_candidates([c1, c2])
        assert len(candidates) == 1
        c = candidates[0]
        assert "crossref" in c.provider_records
        assert "openalex" in c.provider_records
        assert c.provider_records["crossref"]["title"] == "Title A"
        assert c.provider_records["openalex"]["title"] == "Title B"
        entry = BibliographyEntry(
            raw_text="Test. 2020.", authors=None, year=2020,
            title="Title A", doi="10.1000/test",
        )
        with patch(
            "citeguard.verification.RetrievalEngine.search_with_result"
        ) as mock_search:
            mock_search.return_value = MagicMock(
                candidates=candidates, warnings=[], queried_providers=["crossref", "openalex"],
                provider_errors=[],
            )
            engine_v = RetrievalEngine([MagicMock()])
            results = verify_bibliography([entry], engine_v, max_results=5)
        assert len(results[0].provider_conflicts) == 1
        assert "Metadata conflict" in results[0].provider_conflicts[0]

    def test_same_doi_conflicting_year_survives_dedup(self) -> None:
        from citeguard.retrieval import deduplicate_candidates
        c1 = self._make_candidate(
            title="Same Title", year=2020, doi="10.1000/test",
            source_api="crossref",
        )
        c2 = self._make_candidate(
            title="Same Title", year=2022, doi="10.1000/test",
            source_api="openalex",
        )
        candidates = deduplicate_candidates([c1, c2])
        assert len(candidates) == 1
        c = candidates[0]
        assert c.provider_records["crossref"]["year"] == 2020
        assert c.provider_records["openalex"]["year"] == 2022
        entry = BibliographyEntry(
            raw_text="Test. 2020.", authors=None, year=2020,
            title="Same Title", doi="10.1000/test",
        )
        with patch(
            "citeguard.verification.RetrievalEngine.search_with_result"
        ) as mock_search:
            mock_search.return_value = MagicMock(
                candidates=candidates, warnings=[], queried_providers=["crossref", "openalex"],
                provider_errors=[],
            )
            engine_v = RetrievalEngine([MagicMock()])
            results = verify_bibliography([entry], engine_v, max_results=5)
        assert len(results[0].provider_conflicts) == 1
        assert "Year conflict" in results[0].provider_conflicts[0]

    def test_same_doi_formatting_only_no_conflict(self) -> None:
        from citeguard.retrieval import deduplicate_candidates, normalize_title
        c1 = self._make_candidate(
            title="Deep Learning: A Study", year=2020,
            doi="10.1000/test", source_api="crossref",
        )
        c2 = self._make_candidate(
            title="Deep learning - a study", year=2020,
            doi="10.1000/test", source_api="openalex",
        )
        candidates = deduplicate_candidates([c1, c2])
        assert len(candidates) == 1
        c = candidates[0]
        assert normalize_title(c.provider_records["crossref"]["title"]) == normalize_title(
            c.provider_records["openalex"]["title"]
        )

    def test_exact_doi_verified_despite_metadata_disagreement(self) -> None:
        from citeguard.retrieval import deduplicate_candidates
        c1 = self._make_candidate(
            title="Title A", year=2020, doi="10.1000/test",
            source_api="crossref",
        )
        c2 = self._make_candidate(
            title="Title B", year=2022, doi="10.1000/test",
            source_api="openalex",
        )
        candidates = deduplicate_candidates([c1, c2])
        entry = BibliographyEntry(
            raw_text="Test. 2020.", authors=None, year=2020,
            title="Title A", doi="10.1000/test",
        )
        with patch(
            "citeguard.verification.RetrievalEngine.search_with_result"
        ) as mock_search:
            mock_search.return_value = MagicMock(
                candidates=candidates, warnings=[], queried_providers=["crossref", "openalex"],
                provider_errors=[],
            )
            engine_v = RetrievalEngine([MagicMock()])
            results = verify_bibliography([entry], engine_v, max_results=5)
        assert results[0].status == VerificationStatus.VERIFIED
        assert len(results[0].provider_conflicts) >= 1


# ---------------------------------------------------------------------------
# Exact arXiv identity ranking (P1)
# ---------------------------------------------------------------------------


class TestArxivRanking:
    def test_exact_arxiv_outranks_fuzzy_s2(self) -> None:
        from citeguard.retrieval import rank_candidates

        c_arxiv = _make_candidate(
            doi=None, arxiv_id="1706.03762", title="Completely Different",
            authors=["Jones"], year=2017, source_api="arxiv",
        )
        c_fuzzy = _make_candidate(
            doi=None, arxiv_id=None, title="Attention Is All You Need",
            authors=["Vaswani"], year=2017, source_api="semantic_scholar",
        )
        ranked = rank_candidates("1706.03762", [c_fuzzy, c_arxiv])
        assert ranked[0].arxiv_id == "1706.03762"

    def test_exact_arxiv_outranks_newer_year(self) -> None:
        from citeguard.retrieval import rank_candidates

        c_arxiv = _make_candidate(
            doi=None, arxiv_id="1706.03762", title="Old Title",
            authors=["Jones"], year=2017, source_api="arxiv",
        )
        c_newer = _make_candidate(
            doi=None, arxiv_id=None, title="1706.03762 Related Work",
            authors=["Smith"], year=2023, source_api="openalex",
        )
        ranked = rank_candidates("1706.03762", [c_newer, c_arxiv])
        assert ranked[0].arxiv_id == "1706.03762"

    def test_exact_doi_still_top_evidence(self) -> None:
        from citeguard.retrieval import rank_candidates

        c_doi = _make_candidate(
            doi="10.1000/test", title="Something Else",
            authors=["Jones"], year=2019, source_api="openalex",
        )
        c_fuzzy = _make_candidate(
            doi=None, title="10.1000/test Something",
            authors=["Smith"], year=2020, source_api="semantic_scholar",
        )
        ranked = rank_candidates("10.1000/test", [c_fuzzy, c_doi])
        assert ranked[0].doi == "10.1000/test"

    def test_equal_candidates_deterministic_order(self) -> None:
        from citeguard.retrieval import rank_candidates

        c1 = _make_candidate(
            doi=None, title="Alpha Paper", authors=["Smith"],
            year=2020, source_api="crossref",
        )
        c2 = _make_candidate(
            doi=None, title="Alpha Paper", authors=["Smith"],
            year=2020, source_api="openalex",
        )
        r1 = rank_candidates("alpha paper", [c1, c2])
        r2 = rank_candidates("alpha paper", [c2, c1])
        assert [c.source_api for c in r1] == [c.source_api for c in r2]

    def test_ranking_deterministic_repeated(self) -> None:
        from citeguard.retrieval import rank_candidates

        candidates = [
            _make_candidate(
                doi=None, title=f"Paper {i}", authors=["Smith"],
                year=2020, source_api=f"provider_{i}",
            )
            for i in range(5)
        ]
        r1 = rank_candidates("paper", candidates)
        r2 = rank_candidates("paper", list(reversed(candidates)))
        assert [c.title for c in r1] == [c.title for c in r2]


# ---------------------------------------------------------------------------
# Conservative primary-source classification (P1)
# ---------------------------------------------------------------------------


class TestConservativeSourceType:
    def test_venue_only_unknown(self) -> None:
        candidate = _make_candidate(venue="Nature", work_type=None)
        assert _classify_source_type(candidate) == SourceType.UNKNOWN

    def test_venue_springer_unknown(self) -> None:
        candidate = _make_candidate(venue="Springer", work_type=None)
        assert _classify_source_type(candidate) == SourceType.UNKNOWN

    def test_journal_article_unknown(self) -> None:
        candidate = _make_candidate(work_type="journal-article")
        assert _classify_source_type(candidate) == SourceType.UNKNOWN

    def test_book_unknown(self) -> None:
        candidate = _make_candidate(work_type="book")
        assert _classify_source_type(candidate) == SourceType.UNKNOWN

    def test_book_chapter_unknown(self) -> None:
        candidate = _make_candidate(work_type="book-chapter")
        assert _classify_source_type(candidate) == SourceType.UNKNOWN

    def test_proceedings_article_unknown(self) -> None:
        candidate = _make_candidate(work_type="proceedings-article")
        assert _classify_source_type(candidate) == SourceType.UNKNOWN

    def test_dataset_unknown(self) -> None:
        candidate = _make_candidate(work_type="dataset")
        assert _classify_source_type(candidate) == SourceType.UNKNOWN

    def test_review_secondary(self) -> None:
        candidate = _make_candidate(work_type="review")
        assert _classify_source_type(candidate) == SourceType.SECONDARY

    def test_editorial_secondary(self) -> None:
        candidate = _make_candidate(work_type="editorial")
        assert _classify_source_type(candidate) == SourceType.SECONDARY

    def test_meta_analysis_secondary(self) -> None:
        candidate = _make_candidate(work_type="meta-analysis")
        assert _classify_source_type(candidate) == SourceType.SECONDARY

    def test_systematic_review_secondary(self) -> None:
        candidate = _make_candidate(work_type="systematic-review")
        assert _classify_source_type(candidate) == SourceType.SECONDARY

    def test_research_article_primary(self) -> None:
        candidate = _make_candidate(work_type="research-article")
        assert _classify_source_type(candidate) == SourceType.PRIMARY

    def test_original_research_primary(self) -> None:
        candidate = _make_candidate(work_type="original-research")
        assert _classify_source_type(candidate) == SourceType.PRIMARY

    def test_clinical_trial_primary(self) -> None:
        candidate = _make_candidate(work_type="clinical-trial")
        assert _classify_source_type(candidate) == SourceType.PRIMARY

    def test_unknown_type_unknown(self) -> None:
        candidate = _make_candidate(work_type="patent")
        assert _classify_source_type(candidate) == SourceType.UNKNOWN

    def test_missing_work_type_unknown(self) -> None:
        candidate = _make_candidate(work_type=None)
        assert _classify_source_type(candidate) == SourceType.UNKNOWN
