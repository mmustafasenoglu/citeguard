import json
import socket

import pytest
from click.testing import CliRunner

from citeguard.cli import main

_SS = "citeguard.providers.semantic_scholar.SemanticScholarProvider.search"
_CR = "citeguard.providers.crossref.CrossrefProvider.search"
_AR = "citeguard.providers.arxiv.ArxivProvider.search"
_OA = "citeguard.providers.openalex.OpenAlexProvider.search"

_ACADEMIC_PROVIDER_SEARCHES = {
    "semantic_scholar": _SS,
    "crossref": _CR,
    "arxiv": _AR,
    "openalex": _OA,
}


@pytest.fixture(autouse=True)
def stub_academic_providers(monkeypatch):
    """Keep CLI tests hermetic while allowing shared or provider-specific responses."""

    def empty_search(_self, _query, max_results=5):
        return []

    def apply(shared_handler=None, **overrides):
        unknown = set(overrides) - set(_ACADEMIC_PROVIDER_SEARCHES)
        if unknown:
            raise ValueError(f"Unknown academic providers: {sorted(unknown)}")
        default_handler = shared_handler or empty_search
        for name, target in _ACADEMIC_PROVIDER_SEARCHES.items():
            monkeypatch.setattr(target, overrides.get(name, default_handler))

    apply()
    return apply


@pytest.fixture
def network_attempts(monkeypatch):
    """Count and reject socket connections in tests that must remain hermetic."""
    attempts = []

    def fail_on_network(*args, **kwargs):
        attempts.append((args, kwargs))
        pytest.fail("Hermetic CLI test attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", fail_on_network)
    monkeypatch.setattr(socket.socket, "connect", fail_on_network)
    return attempts


def _doc(tmp_path, text=""):
    path = tmp_path / "paper.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_check_runs_end_to_end(tmp_path) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout (Smith, 2020).\n\n"
        "References\n\nSmith, J. (2020). Paper.",
    )

    result = CliRunner().invoke(main, ["check", str(path), "--format", "json"])
    payload = json.loads(result.output)

    assert "health_score" in payload["summary"]
    assert "total_claims" in payload["summary"]
    assert "claims" in payload
    assert "verification_results" in payload
    assert "priority_review" in payload


def test_check_terminal_output(tmp_path) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\n"
        "References\n\nSmith, J. (2020). Paper.",
    )

    result = CliRunner().invoke(main, ["check", str(path)])
    assert "Citation Health Score" in result.output
    assert "Audit Summary" in result.output


def test_check_markdown_output(tmp_path) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\n"
        "References\n\nSmith, J. (2020). Paper.",
    )

    output = tmp_path / "report.md"
    result = CliRunner().invoke(
        main, ["check", str(path), "--format", "md", "--output", str(output)]
    )
    # exit_code 1 means findings were detected (expected for a claim-heavy doc)
    assert result.exit_code in (0, 1)
    content = output.read_text(encoding="utf-8")
    assert "# citeguard Check Report" in content
    assert "Citation Health Score" in content


def test_suggest_runs_end_to_end(tmp_path) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout regularization techniques.",
    )

    result = CliRunner().invoke(main, ["suggest", str(path), "--format", "json"])
    payload = json.loads(result.output)

    assert "summary" in payload
    assert "claims" in payload
    assert "suggestions" in payload
    assert payload["summary"]["total_claims"] >= 1


def test_suggest_terminal_output(tmp_path) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout regularization.",
    )

    result = CliRunner().invoke(main, ["suggest", str(path)])
    assert "Source Suggestions" in result.output


def test_check_respects_severity_filter(tmp_path) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\n"
        "Some general observation about the field.",
    )

    result = CliRunner().invoke(
        main, ["check", str(path), "--severity", "high", "--format", "json"]
    )
    payload = json.loads(result.output)
    for claim in payload["claims"]:
        assert claim["severity"] == "high"


def test_check_respects_max_claims(tmp_path) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\n"
        "Research shows 80% accuracy on benchmarks.\n\n"
        "Experiments demonstrate 90% improvement.",
    )

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


def test_verify_still_works(tmp_path, stub_academic_providers) -> None:
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

    stub_academic_providers(crossref=fake_search)
    result = CliRunner().invoke(main, ["verify", str(path)])
    assert result.exit_code == 0
    assert "verified" in result.output


def test_check_format_both(tmp_path) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\n"
        "References\n\nSmith, J. (2020). Paper.",
    )

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


def test_check_format_both_with_output(tmp_path) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\n"
        "References\n\nSmith, J. (2020). Paper.",
    )

    out = tmp_path / "report"
    result = CliRunner().invoke(
        main, ["check", str(path), "--format", "both", "--output", str(out)]
    )
    assert result.exit_code in (0, 1)
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()


def test_suggest_format_both(tmp_path) -> None:
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout regularization techniques.",
    )

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


def test_verify_format_both(tmp_path, stub_academic_providers) -> None:
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

    stub_academic_providers(crossref=fake_search)
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


def test_check_missing_file_exits_2(tmp_path) -> None:
    result = CliRunner().invoke(main, ["check", str(tmp_path / "missing.txt")])
    assert result.exit_code == 2


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


def test_check_exits_1_on_findings(tmp_path, network_attempts) -> None:
    """Health score < 80 should produce exit code 1."""
    path = _doc(
        tmp_path,
        "Large language models generate realistic citations.\n\n"
        "References\n\nSmith, J. (2020). Paper.",
    )

    result = CliRunner().invoke(main, ["check", str(path), "--no-cache"])
    assert result.exit_code == 1
    assert network_attempts == []


def test_check_evidence_pipeline_integration(tmp_path, stub_academic_providers) -> None:
    """Integration test: full pipeline with evidence extraction and entailment."""
    from citeguard.models import SourceCandidate

    path = _doc(
        tmp_path,
        "Transformers were introduced in 2017. "
        "They achieve state-of-the-art results in NLP.",
    )

    def fake_search(_self, _query, max_results=5):
        return [
            SourceCandidate(
                title="Attention Is All You Need",
                authors=["Vaswani", "Shazeer", "Parmar"],
                year=2017,
                venue="NIPS",
                doi="10.1000/transformers",
                url="https://arxiv.org/abs/1706.03762",
                abstract=(
                    "We propose a new simple network architecture, the Transformer, "
                    "based solely on attention mechanisms. The Transformer achieves "
                    "state-of-the-art results in machine translation tasks."
                ),
                source_api="semantic_scholar",
            )
        ][:max_results]

    stub_academic_providers(fake_search)

    result = CliRunner().invoke(
        main, ["check", str(path), "--format", "json", "--threshold", "0"]
    )
    assert result.exit_code in (0, 1)

    payload = json.loads(result.output)
    assert "priority_review" in payload
    assert len(payload["priority_review"]) >= 1

    matched_items = [
        item for item in payload["priority_review"] if item.get("matched") is not None
    ]
    assert len(matched_items) >= 1
    pr = matched_items[0]
    assert "evidence" in pr["matched"]
    evidence = pr["matched"]["evidence"]
    assert isinstance(evidence, list)
    assert len(evidence) >= 1
    assert "text" in evidence[0]
    assert "relevance_score" in evidence[0]
    assert "verdict" in evidence[0]


def test_check_require_evidence_filter(tmp_path, stub_academic_providers) -> None:
    """--require-evidence filters out claims without evidence."""
    from citeguard.models import SourceCandidate

    path = _doc(
        tmp_path,
        "Transformers were introduced in 2017. "
        "The sky appears blue during daytime.",
    )

    def fake_search_transformers(_self, query, max_results=5):
        if "transformer" in query.lower() or "2017" in query.lower():
            return [
                SourceCandidate(
                    title="Attention Is All You Need",
                    authors=["Vaswani", "Shazeer", "Parmar"],
                    year=2017,
                    venue="NIPS",
                    doi="10.1000/transformers",
                    url="https://arxiv.org/abs/1706.03762",
                    abstract=(
                        "We propose a new simple network architecture, the Transformer, "
                        "based solely on attention mechanisms. The Transformer achieves "
                        "state-of-the-art results in machine translation tasks."
                    ),
                    source_api="semantic_scholar",
                )
            ][:max_results]
        return []

    stub_academic_providers(fake_search_transformers)

    # Without --require-evidence: both claims should appear
    result_all = CliRunner().invoke(
        main, ["check", str(path), "--format", "json", "--no-cache", "--threshold", "0"]
    )
    payload_all = json.loads(result_all.output)
    assert len(payload_all["priority_review"]) >= 2

    # With --require-evidence: only claim with evidence should appear
    result_filtered = CliRunner().invoke(
        main,
        [
            "check", str(path), "--format", "json",
            "--require-evidence", "--no-cache", "--threshold", "0",
        ],
    )
    payload_filtered = json.loads(result_filtered.output)
    assert len(payload_filtered["priority_review"]) == 1
    assert "Transformers" in payload_filtered["priority_review"][0]["claim_text"]
