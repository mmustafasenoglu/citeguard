from citeguard.bibliography import bibliography_issues, parse_bibliography
from citeguard.extractor import extract_citations
from citeguard.models import BibliographyIssueKind


def test_parse_bibliography() -> None:
    paragraphs = [
        "A claim (Smith, 2020).",
        "References",
        "Smith, J. (2020). A useful paper. Journal. 10.1000/abc.1",
    ]
    start, entries = parse_bibliography(paragraphs)
    assert start == 1
    assert len(entries) == 1
    assert entries[0].year == 2020
    assert entries[0].doi == "10.1000/abc.1"


def test_duplicate_doi_issue() -> None:
    paragraphs = [
        "References",
        "Smith, J. (2020). Paper A. 10.1000/abc.1",
        "Smith, J. (2020). Paper A duplicate. 10.1000/abc.1",
    ]
    _, entries = parse_bibliography(paragraphs)
    issues = bibliography_issues([], entries)
    assert any(issue.kind == BibliographyIssueKind.DUPLICATE_DOI for issue in issues)


def test_cited_not_listed() -> None:
    paragraphs = ["A claim (Jones, 2021).", "References", "Smith, J. (2020). Paper."]
    citations = extract_citations(paragraphs[:1])
    _, entries = parse_bibliography(paragraphs)
    issues = bibliography_issues(citations, entries)
    assert any(issue.kind == BibliographyIssueKind.CITED_NOT_LISTED for issue in issues)


def test_detects_reference_appendix_from_entry_shape() -> None:
    paragraphs = [
        "Main text",
        "APPENDIX 1",
        "Smith, J. (2020). First paper. 10.1000/first",
        "Jones, A. (2021). Second paper. 10.1000/second",
        "Lee, B. (2022). Third paper. 10.1000/third",
    ]
    start, entries = parse_bibliography(paragraphs)
    assert start == 1
    assert len(entries) == 3


def test_detects_prefixed_turkish_bibliography_heading() -> None:
    paragraphs = [
        "6. EKLER",
        "EK-1:  KAYNAKLAR",
        "Smith, J. (2020). First paper.",
    ]
    start, entries = parse_bibliography(paragraphs)

    assert start == 1
    assert len(entries) == 1
    assert entries[0].authors == "Smith, J"


def test_parses_and_matches_no_date_entry() -> None:
    paragraphs = [
        "Dataset details (Google Earth Engine Data Catalog, t.y.).",
        "Kaynakça",
        "Google Earth Engine Data Catalog. (t.y.). Sentinel-2 MSI dataset.",
    ]
    citations = extract_citations(paragraphs[:1])
    _, entries = parse_bibliography(paragraphs)
    issues = bibliography_issues(citations, entries)

    assert entries[0].authors == "Google Earth Engine Data Catalog"
    assert entries[0].year is None
    assert entries[0].no_date is True
    assert issues == []


def test_cited_not_listed_issue_includes_paragraph_location() -> None:
    citations = extract_citations(["Context.", "A claim (Jones, 2021)."])
    _, entries = parse_bibliography(["References", "Smith, J. (2020). Paper."])
    issues = bibliography_issues(citations, entries)

    issue = next(issue for issue in issues if issue.kind == BibliographyIssueKind.CITED_NOT_LISTED)
    assert issue.paragraph_index == 1


def test_title_excludes_venue_and_doi_url() -> None:
    _, entries = parse_bibliography(
        [
            "References",
            "Milletari, F. (2016). V-Net: Segmentation. 2016 Conference. "
            "https://doi.org/10.1000/example",
        ]
    )

    assert entries[0].title == "V-Net: Segmentation"


def test_listed_not_cited_issue() -> None:
    paragraphs = [
        "A claim about transformers.",
        "References",
        "Smith, J. (2020). A useful paper. Journal. 10.1000/abc.1",
        "Jones, A. (2021). Another paper. Journal. 10.1000/xyz.2",
    ]
    citations = extract_citations(paragraphs[:1])
    _, entries = parse_bibliography(paragraphs)
    issues = bibliography_issues(citations, entries)
    listed_not_cited = [
        i for i in issues if i.kind == BibliographyIssueKind.LISTED_NOT_CITED
    ]
    assert len(listed_not_cited) == 2
