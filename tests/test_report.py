from citeguard.extractor import parse_document
from citeguard.models import (
    Claim,
    ClaimType,
    Severity,
)
from citeguard.report import (
    check_report,
    inspection_report,
    markdown_check_report,
    markdown_suggest_report,
    markdown_verification_report,
    suggest_report,
    verification_report,
)
from citeguard.scoring import AuditMetrics


def _parsed(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text(
        "A claim (Smith, 2020).\n\nReferences\n\nSmith, J. (2020). Paper.",
        encoding="utf-8",
    )
    return parse_document(path)


def _claim():
    return Claim(
        text="A factual claim about science",
        search_query="factual claim science",
        claim_type=ClaimType.GENERAL_FACT,
        severity=Severity.MEDIUM,
        paragraph_index=0,
        has_existing_citation=True,
    )


def _metrics():
    return AuditMetrics(
        total_claims=1,
        claims_requiring_citations=0,
        cited_claims=1,
        verified_citations=1,
        weak_matches=0,
        unresolved_citations=0,
        uncited_high_severity_claims=0,
        contradictions=0,
        bibliography_issues=0,
        citation_coverage=1.0,
        verification_ratio=1.0,
        support_ratio=1.0,
        bibliography_consistency=1.0,
        evidence_coverage=1.0,
        health_score=100,
    )


def test_inspection_report_structure(tmp_path) -> None:
    parsed = _parsed(tmp_path)
    report = inspection_report(parsed)
    assert report["schema_version"] == "3"
    assert "document" in report
    assert "summary" in report
    assert "citations" in report
    assert "bibliography" in report
    assert "issues" in report


def test_verification_report_structure(tmp_path) -> None:
    parsed = _parsed(tmp_path)
    report = verification_report(parsed, [])
    assert report["schema_version"] == "3"
    assert "provider" in report
    assert "privacy" in report
    assert "summary" in report
    assert "results" in report


def test_check_report_structure(tmp_path) -> None:
    parsed = _parsed(tmp_path)
    report = check_report(
        parsed,
        verification_results=[],
        claims=[_claim()],
        verification_for_claims=[],
        metrics=_metrics(),
        bib_issues=[],
    )
    assert report["schema_version"] == "3"
    assert "privacy" in report
    assert "summary" in report
    assert "health_score" in report["summary"]
    assert "claims" in report
    assert "verification_results" in report
    assert "priority_review" in report
    assert "bibliography_issues" in report


def test_suggest_report_structure(tmp_path) -> None:
    parsed = _parsed(tmp_path)
    report = suggest_report(parsed, [_claim()], [])
    assert report["schema_version"] == "3"
    assert "summary" in report
    assert "claims" in report
    assert "suggestions" in report
    assert "privacy" in report


def test_markdown_check_report_content(tmp_path) -> None:
    parsed = _parsed(tmp_path)
    report = markdown_check_report(
        parsed,
        verification_results=[],
        claims=[_claim()],
        verification_for_claims=[],
        metrics=_metrics(),
        bib_issues=[],
    )
    assert "# citeguard Check Report" in report
    assert "Citation Health Score" in report
    assert "100/100" in report
    assert "Detected Claims" in report
    assert "factual claim about science" in report
    assert "review-prioritization heuristic" in report


def test_markdown_suggest_report_content(tmp_path) -> None:
    parsed = _parsed(tmp_path)
    report = markdown_suggest_report(parsed, [_claim()], [])
    assert "# citeguard Suggest Report" in report
    assert "human review" in report
    assert "claims" in report.lower()


def test_markdown_verification_report_content(tmp_path) -> None:
    parsed = _parsed(tmp_path)
    report = markdown_verification_report(parsed, [])
    assert "# citeguard Verification Report" in report
    assert "Crossref" in report
    assert "does not establish claim support" in report


def test_check_report_includes_bib_issues(tmp_path) -> None:
    parsed = _parsed(tmp_path)
    from citeguard.bibliography import bibliography_issues

    issues = bibliography_issues(parsed.citations, parsed.bibliography_entries)
    report = check_report(
        parsed,
        verification_results=[],
        claims=[],
        verification_for_claims=[],
        metrics=_metrics(),
        bib_issues=issues,
    )
    assert isinstance(report["bibliography_issues"], list)


def test_check_report_schema_version(tmp_path) -> None:
    parsed = _parsed(tmp_path)
    report = check_report(
        parsed,
        verification_results=[],
        claims=[],
        verification_for_claims=[],
        metrics=_metrics(),
        bib_issues=[],
    )
    assert report["schema_version"] == "3"
