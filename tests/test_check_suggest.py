import json

from click.testing import CliRunner

from citeguard.cli import main


def _doc(tmp_path, text=""):
    path = tmp_path / "paper.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_check_runs_end_to_end(tmp_path, monkeypatch) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout (Smith, 2020).\n\n"
        "References\n\nSmith, J. (2020). Paper.",
    )

    def fake_search(_self, _query, max_results=5):
        return []

    monkeypatch.setattr("citeguard.cli.CrossrefProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.SemanticScholarProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.ArxivProvider.search", fake_search)

    result = CliRunner().invoke(main, ["check", str(path), "--format", "json"])
    payload = json.loads(result.output)

    assert "health_score" in payload["summary"]
    assert "total_claims" in payload["summary"]
    assert "claims" in payload
    assert "verification_results" in payload
    assert "priority_review" in payload


def test_check_terminal_output(tmp_path, monkeypatch) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\n"
        "References\n\nSmith, J. (2020). Paper.",
    )

    def fake_search(_self, _query, max_results=5):
        return []

    monkeypatch.setattr("citeguard.cli.CrossrefProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.SemanticScholarProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.ArxivProvider.search", fake_search)

    result = CliRunner().invoke(main, ["check", str(path)])
    assert "Citation Health Score" in result.output
    assert "Audit Summary" in result.output


def test_check_markdown_output(tmp_path, monkeypatch) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\n"
        "References\n\nSmith, J. (2020). Paper.",
    )

    def fake_search(_self, _query, max_results=5):
        return []

    monkeypatch.setattr("citeguard.cli.CrossrefProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.SemanticScholarProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.ArxivProvider.search", fake_search)

    output = tmp_path / "report.md"
    result = CliRunner().invoke(
        main, ["check", str(path), "--format", "md", "--output", str(output)]
    )
    # exit_code 1 means findings were detected (expected for a claim-heavy doc)
    assert result.exit_code in (0, 1)
    content = output.read_text(encoding="utf-8")
    assert "# citeguard Check Report" in content
    assert "Citation Health Score" in content


def test_suggest_runs_end_to_end(tmp_path, monkeypatch) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout regularization techniques.",
    )

    def fake_search(_self, _query, max_results=5):
        return []

    monkeypatch.setattr("citeguard.cli.SemanticScholarProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.CrossrefProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.ArxivProvider.search", fake_search)

    result = CliRunner().invoke(main, ["suggest", str(path), "--format", "json"])
    payload = json.loads(result.output)

    assert "summary" in payload
    assert "claims" in payload
    assert "suggestions" in payload
    assert payload["summary"]["total_claims"] >= 1


def test_suggest_terminal_output(tmp_path, monkeypatch) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout regularization.",
    )

    def fake_search(_self, _query, max_results=5):
        return []

    monkeypatch.setattr("citeguard.cli.SemanticScholarProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.CrossrefProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.ArxivProvider.search", fake_search)

    result = CliRunner().invoke(main, ["suggest", str(path)])
    assert "Source Suggestions" in result.output


def test_check_respects_severity_filter(tmp_path, monkeypatch) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\n"
        "Some general observation about the field.",
    )

    def fake_search(_self, _query, max_results=5):
        return []

    monkeypatch.setattr("citeguard.cli.CrossrefProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.SemanticScholarProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.ArxivProvider.search", fake_search)

    result = CliRunner().invoke(
        main, ["check", str(path), "--severity", "high", "--format", "json"]
    )
    payload = json.loads(result.output)
    for claim in payload["claims"]:
        assert claim["severity"] == "high"


def test_check_respects_max_claims(tmp_path, monkeypatch) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\n"
        "Research shows 80% accuracy on benchmarks.\n\n"
        "Experiments demonstrate 90% improvement.",
    )

    def fake_search(_self, _query, max_results=5):
        return []

    monkeypatch.setattr("citeguard.cli.CrossrefProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.SemanticScholarProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.ArxivProvider.search", fake_search)

    result = CliRunner().invoke(
        main, ["check", str(path), "--max-claims", "1", "--format", "json"]
    )
    payload = json.loads(result.output)
    assert len(payload["claims"]) <= 1


def test_inspect_still_works(tmp_path) -> None:
    path = _doc(
        tmp_path,
        "A claim (Smith, 2020).\n\nReferences\n\nSmith, J. (2020). Paper.",
    )
    result = CliRunner().invoke(main, ["inspect", str(path)])
    assert result.exit_code == 0
    assert "Detected citations" in result.output


def test_verify_still_works(tmp_path, monkeypatch) -> None:
    from citeguard.models import SourceCandidate

    path = _doc(
        tmp_path,
        "A claim (Smith, 2020).\n\nReferences\n\n"
        "Smith, J. (2020). A Useful Paper. 10.1000/example",
    )

    def fake_search(_self, _query, max_results=5):
        return [
            SourceCandidate(
                title="A Useful Paper",
                authors=["Jane Smith"],
                year=2020,
                venue="Journal",
                doi="10.1000/example",
                url="https://doi.org/10.1000/example",
                abstract=None,
                source_api="crossref",
            )
        ][:max_results]

    monkeypatch.setattr("citeguard.cli.CrossrefProvider.search", fake_search)
    result = CliRunner().invoke(main, ["verify", str(path)])
    assert result.exit_code == 0
    assert "verified" in result.output


def test_check_format_both(tmp_path, monkeypatch) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\n"
        "References\n\nSmith, J. (2020). Paper.",
    )

    def fake_search(_self, _query, max_results=5):
        return []

    monkeypatch.setattr("citeguard.cli.CrossrefProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.SemanticScholarProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.ArxivProvider.search", fake_search)

    result = CliRunner().invoke(main, ["check", str(path), "--format", "both"])
    assert result.exit_code in (0, 1)

    json_path = tmp_path / "paper.citeguard.json"
    md_path = tmp_path / "paper.citeguard.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "health_score" in payload["summary"]

    md_content = md_path.read_text(encoding="utf-8")
    assert "# citeguard Check Report" in md_content


def test_check_format_both_with_output(tmp_path, monkeypatch) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\n"
        "References\n\nSmith, J. (2020). Paper.",
    )

    def fake_search(_self, _query, max_results=5):
        return []

    monkeypatch.setattr("citeguard.cli.CrossrefProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.SemanticScholarProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.ArxivProvider.search", fake_search)

    out = tmp_path / "report"
    result = CliRunner().invoke(
        main, ["check", str(path), "--format", "both", "--output", str(out)]
    )
    assert result.exit_code in (0, 1)
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()


def test_suggest_format_both(tmp_path, monkeypatch) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout regularization techniques.",
    )

    def fake_search(_self, _query, max_results=5):
        return []

    monkeypatch.setattr("citeguard.cli.SemanticScholarProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.CrossrefProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.ArxivProvider.search", fake_search)

    result = CliRunner().invoke(main, ["suggest", str(path), "--format", "both"])
    assert result.exit_code == 0

    json_path = tmp_path / "paper.citeguard.json"
    md_path = tmp_path / "paper.citeguard.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "suggestions" in payload

    md_content = md_path.read_text(encoding="utf-8")
    assert "# citeguard Suggest Report" in md_content


def test_inspect_format_both(tmp_path) -> None:
    path = _doc(
        tmp_path,
        "A claim (Smith, 2020).\n\nReferences\n\nSmith, J. (2020). Paper.",
    )
    result = CliRunner().invoke(main, ["inspect", str(path), "--format", "both"])
    assert result.exit_code == 0

    json_path = tmp_path / "paper.citeguard.json"
    md_path = tmp_path / "paper.citeguard.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["detected_citations"] == 1

    md_content = md_path.read_text(encoding="utf-8")
    assert "# citeguard Inspection Report" in md_content


def test_verify_format_both(tmp_path, monkeypatch) -> None:
    from citeguard.models import SourceCandidate

    path = _doc(
        tmp_path,
        "A claim (Smith, 2020).\n\nReferences\n\n"
        "Smith, J. (2020). A Useful Paper. 10.1000/example",
    )

    def fake_search(_self, _query, max_results=5):
        return [
            SourceCandidate(
                title="A Useful Paper",
                authors=["Jane Smith"],
                year=2020,
                venue="Journal",
                doi="10.1000/example",
                url="https://doi.org/10.1000/example",
                abstract=None,
                source_api="crossref",
            )
        ][:max_results]

    monkeypatch.setattr("citeguard.cli.CrossrefProvider.search", fake_search)
    result = CliRunner().invoke(main, ["verify", str(path), "--format", "both"])
    assert result.exit_code == 0

    json_path = tmp_path / "paper.citeguard.json"
    md_path = tmp_path / "paper.citeguard.md"
    assert json_path.exists()
    assert md_path.exists()


def test_inspect_empty_file_rejected(tmp_path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    result = CliRunner().invoke(main, ["inspect", str(path)])
    assert result.exit_code != 0
    assert "empty" in result.output.lower() or "whitespace" in result.output.lower()


def test_check_empty_file_rejected(tmp_path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    result = CliRunner().invoke(main, ["check", str(path)])
    assert result.exit_code != 0
    assert "empty" in result.output.lower() or "whitespace" in result.output.lower()


def test_suggest_empty_file_rejected(tmp_path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    result = CliRunner().invoke(main, ["suggest", str(path)])
    assert result.exit_code != 0
    assert "empty" in result.output.lower() or "whitespace" in result.output.lower()


def test_verify_empty_file_rejected(tmp_path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    result = CliRunner().invoke(main, ["verify", str(path)])
    assert result.exit_code != 0
    assert "empty" in result.output.lower() or "whitespace" in result.output.lower()


def test_check_exits_1_on_findings(tmp_path, monkeypatch) -> None:
    """Health score < 80 should produce exit code 1."""
    path = _doc(
        tmp_path,
        "Large language models generate realistic citations.\n\n"
        "References\n\nSmith, J. (2020). Paper.",
    )

    def fake_search(_self, _query, max_results=5):
        return []

    monkeypatch.setattr("citeguard.cli.CrossrefProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.SemanticScholarProvider.search", fake_search)
    monkeypatch.setattr("citeguard.cli.ArxivProvider.search", fake_search)

    result = CliRunner().invoke(main, ["check", str(path)])
    assert result.exit_code == 1
