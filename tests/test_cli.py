import json

from click.testing import CliRunner

from citeguard.cli import main
from citeguard.models import SourceCandidate


def test_inspect_shows_citation_locations_and_match_status(tmp_path) -> None:
    document = tmp_path / "paper.txt"
    document.write_text(
        "A claim (Smith, 2020).\n\n"
        "Another claim (Jones, 2021).\n\n"
        "References\n\n"
        "Smith, J. (2020). Paper.",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["inspect", str(document)])

    assert result.exit_code == 0
    assert "Detected citations" in result.output
    assert "(Smith, 2020)" in result.output
    assert "matched" in result.output
    assert "(Jones, 2021)" in result.output
    assert "not listed" in result.output
    assert "paragraph 2" in result.output


def test_inspect_json_is_machine_readable(tmp_path) -> None:
    document = tmp_path / "paper.txt"
    document.write_text(
        "A claim (Smith, 2020).\n\nReferences\n\nSmith, J. (2020). Paper.",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["inspect", str(document), "--format", "json"])
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["schema_version"] == "2"
    assert payload["citations"][0]["sentence"] == "A claim (Smith, 2020)."
    assert payload["citations"][0]["bibliography_entry_indexes"] == [1]


def test_verify_uses_crossref_and_can_write_json(tmp_path, monkeypatch) -> None:
    document = tmp_path / "paper.txt"
    output = tmp_path / "verification.json"
    document.write_text(
        "A claim (Smith, 2020).\n\nReferences\n\n"
        "Smith, J. (2020). A Useful Paper. 10.1000/example",
        encoding="utf-8",
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
    monkeypatch.setattr("citeguard.cli.OpenAlexProvider.search", fake_search)
    result = CliRunner().invoke(
        main,
        [
            "verify",
            str(document),
            "--format",
            "json",
            "--output",
            str(output),
            "--no-cache",
        ],
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert payload["provider"] == "crossref+openalex"
    assert payload["results"][0]["status"] == "verified"
    assert payload["results"][0]["scores"]["overall"] == 100
