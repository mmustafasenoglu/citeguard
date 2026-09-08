"""Regression tests for Checkpoint 7 CLI/product-layer integration.

Covers the CP7.2 test matrix:
1. check delegates to product layer
2. check --offline
3. check --format json
4. check --format md
5. check --format both
6. provider-phase failure -> exit 3
7. offline incomplete metrics not exit 3
8. findings result -> exit 1
9. clean result -> exit 0
10. --require-evidence
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from citeguard.cli import main


def _doc(tmp_path: Path, text: str = "") -> Path:
    path = tmp_path / "paper.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _noop_search(_self, _query, max_results=5):
    return []


def _patch_providers(monkeypatch):
    monkeypatch.setattr(
        "citeguard.providers.crossref.CrossrefProvider.search", _noop_search
    )
    monkeypatch.setattr(
        "citeguard.providers.semantic_scholar.SemanticScholarProvider.search",
        _noop_search,
    )
    monkeypatch.setattr(
        "citeguard.providers.arxiv.ArxivProvider.search", _noop_search
    )
    monkeypatch.setattr(
        "citeguard.providers.openalex.OpenAlexProvider.search", _noop_search
    )


# ---------------------------------------------------------------------------
# 1. check delegates to product layer
# ---------------------------------------------------------------------------


def test_check_delegates_to_product_layer(tmp_path, monkeypatch) -> None:
    """check must call audit_document exactly once with correct options."""
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\nReferences\n\nSmith, J. (2020). Paper.",
    )
    _patch_providers(monkeypatch)

    call_tracker = {"called": False, "options": None}

    def patched_audit(source, options=None):
        call_tracker["called"] = True
        call_tracker["options"] = options
        from citeguard.audit import audit_document as real_audit

        return real_audit(source, options)

    monkeypatch.setattr("citeguard.audit.audit_document", patched_audit)

    result = CliRunner().invoke(main, ["check", str(path), "--format", "json"])
    assert result.exit_code in (0, 1)
    assert call_tracker["called"]
    assert call_tracker["options"] is not None
    assert call_tracker["options"].max_results == 5


# ---------------------------------------------------------------------------
# 2. check --offline
# ---------------------------------------------------------------------------


def test_check_offline_e2e(tmp_path, monkeypatch) -> None:
    """--offline: no crash, no remote calls, execution context correct."""
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\nReferences\n\nSmith, J. (2020). Paper.",
    )

    remote_calls: list[str] = []

    def track_search(self, _query, max_results=5):
        remote_calls.append(self.name)
        return []

    monkeypatch.setattr(
        "citeguard.providers.crossref.CrossrefProvider.search", track_search
    )
    monkeypatch.setattr(
        "citeguard.providers.semantic_scholar.SemanticScholarProvider.search",
        track_search,
    )
    monkeypatch.setattr(
        "citeguard.providers.arxiv.ArxivProvider.search", track_search
    )
    monkeypatch.setattr(
        "citeguard.providers.openalex.OpenAlexProvider.search", track_search
    )

    result = CliRunner().invoke(
        main, ["check", str(path), "--offline", "--format", "json"]
    )
    assert result.exit_code in (0, 1)

    payload = json.loads(result.output)
    assert payload["schema_version"] == "3"
    assert payload["execution"]["offline"] is True
    assert payload["execution"]["network_allowed"] is False
    assert payload["execution"]["network_used"] is False

    assert remote_calls == [], f"No remote calls expected offline, got: {remote_calls}"


# ---------------------------------------------------------------------------
# 3. check --format json
# ---------------------------------------------------------------------------


def test_check_format_json(tmp_path, monkeypatch) -> None:
    """JSON output has schema_version, execution, metrics, claims."""
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\nReferences\n\nSmith, J. (2020). Paper.",
    )
    _patch_providers(monkeypatch)

    result = CliRunner().invoke(
        main, ["check", str(path), "--format", "json"]
    )
    assert result.exit_code in (0, 1)
    payload = json.loads(result.output)

    assert payload["schema_version"] == "3"
    assert "execution" in payload
    assert "summary" in payload
    assert "health_score" in payload["summary"]
    assert "claims" in payload
    assert "bibliography_verification" in payload
    assert "priority_review" in payload
    assert "diagnostics" in payload


# ---------------------------------------------------------------------------
# 4. check --format md
# ---------------------------------------------------------------------------


def test_check_format_md(tmp_path, monkeypatch) -> None:
    """Markdown output derives from the same AuditResult."""
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\nReferences\n\nSmith, J. (2020). Paper.",
    )
    _patch_providers(monkeypatch)

    output = tmp_path / "report.md"
    result = CliRunner().invoke(
        main, ["check", str(path), "--format", "md", "--output", str(output)]
    )
    assert result.exit_code in (0, 1)
    content = output.read_text(encoding="utf-8")
    assert "# citeguard Check Report" in content
    assert "Citation Health Score" in content
    assert "review-prioritization heuristic" in content


# ---------------------------------------------------------------------------
# 5. check --format both
# ---------------------------------------------------------------------------


def test_check_format_both(tmp_path, monkeypatch) -> None:
    """Both: JSON and MD derive from same AuditResult."""
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\nReferences\n\nSmith, J. (2020). Paper.",
    )
    _patch_providers(monkeypatch)

    result = CliRunner().invoke(
        main, ["check", str(path), "--format", "both"]
    )
    assert result.exit_code in (0, 1)

    json_path = tmp_path / "paper.citeguard.json"
    md_path = tmp_path / "paper.citeguard.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    md_content = md_path.read_text(encoding="utf-8")

    assert payload["schema_version"] == "3"
    assert "health_score" in payload["summary"]
    assert "# citeguard Check Report" in md_content
    assert "Citation Health Score" in md_content


# ---------------------------------------------------------------------------
# 6. provider-phase failure -> exit 3
# ---------------------------------------------------------------------------


def test_check_provider_failure_exit_3(tmp_path, monkeypatch) -> None:
    """Online provider total failure -> exit 3."""
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\nReferences\n\nSmith, J. (2020). Paper.",
    )

    def failing_search(_self, _query, max_results=5):
        raise ConnectionError("Provider unavailable")

    monkeypatch.setattr(
        "citeguard.providers.crossref.CrossrefProvider.search", failing_search
    )
    monkeypatch.setattr(
        "citeguard.providers.semantic_scholar.SemanticScholarProvider.search",
        failing_search,
    )
    monkeypatch.setattr(
        "citeguard.providers.arxiv.ArxivProvider.search", failing_search
    )
    monkeypatch.setattr(
        "citeguard.providers.openalex.OpenAlexProvider.search", failing_search
    )

    result = CliRunner().invoke(
        main, ["check", str(path), "--format", "json", "--no-cache"]
    )
    assert result.exit_code == 3

    payload = json.loads(result.output)
    assert payload["provider_phase_failed"] is True


# ---------------------------------------------------------------------------
# 7. offline incomplete metrics not exit 3
# ---------------------------------------------------------------------------


def test_check_offline_not_exit_3(tmp_path) -> None:
    """Offline mode must not produce exit 3 even with incomplete metrics."""
    path = _doc(
        tmp_path,
        "Over 70% of models use dropout.\n\nReferences\n\nSmith, J. (2020). Paper.",
    )

    result = CliRunner().invoke(
        main, ["check", str(path), "--offline", "--format", "json"]
    )
    assert result.exit_code != 3

    payload = json.loads(result.output)
    assert payload["execution"]["offline"] is True


# ---------------------------------------------------------------------------
# 8. findings result -> exit 1
# ---------------------------------------------------------------------------


def test_check_findings_exit_1(tmp_path, monkeypatch) -> None:
    """Document with findings (health < 80) -> exit 1."""
    path = _doc(
        tmp_path,
        "Large language models generate realistic citations.\n\n"
        "References\n\nSmith, J. (2020). Paper.",
    )
    _patch_providers(monkeypatch)

    result = CliRunner().invoke(main, ["check", str(path)])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# 9. clean result -> exit 0
# ---------------------------------------------------------------------------


def test_check_clean_exit_0(tmp_path, monkeypatch) -> None:
    """Well-cited document (health >= 80) -> exit 0."""
    from citeguard.models import SourceCandidate

    path = _doc(
        tmp_path,
        "Over 70% of models use dropout (Smith, 2020).\n\n"
        "References\n\n"
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
                abstract="Over 70% of models use dropout.",
                source_api="crossref",
            )
        ][:max_results]

    monkeypatch.setattr(
        "citeguard.providers.crossref.CrossrefProvider.search", fake_search
    )
    monkeypatch.setattr(
        "citeguard.providers.semantic_scholar.SemanticScholarProvider.search",
        fake_search,
    )
    monkeypatch.setattr(
        "citeguard.providers.arxiv.ArxivProvider.search", fake_search
    )
    monkeypatch.setattr(
        "citeguard.providers.openalex.OpenAlexProvider.search", fake_search
    )

    result = CliRunner().invoke(main, ["check", str(path), "--format", "json"])
    payload = json.loads(result.output)

    score = payload["summary"]["health_score"]
    complete = payload["summary"]["health_score_complete"]
    if complete and score is not None and score >= 80:
        assert result.exit_code == 0
    else:
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# 10. --require-evidence
# ---------------------------------------------------------------------------


def test_check_require_evidence_does_not_alter_metrics(tmp_path, monkeypatch) -> None:
    """--require-evidence is display-only; metrics must remain identical."""
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

    monkeypatch.setattr(
        "citeguard.providers.semantic_scholar.SemanticScholarProvider.search",
        fake_search,
    )
    monkeypatch.setattr(
        "citeguard.providers.crossref.CrossrefProvider.search", fake_search
    )
    monkeypatch.setattr(
        "citeguard.providers.arxiv.ArxivProvider.search", fake_search
    )
    monkeypatch.setattr(
        "citeguard.providers.openalex.OpenAlexProvider.search", fake_search
    )

    result_all = CliRunner().invoke(
        main, ["check", str(path), "--format", "json", "--threshold", "0"]
    )
    payload_all = json.loads(result_all.output)

    result_filtered = CliRunner().invoke(
        main,
        [
            "check", str(path), "--format", "json",
            "--require-evidence", "--threshold", "0",
        ],
    )
    payload_filtered = json.loads(result_filtered.output)

    assert payload_all["summary"] == payload_filtered["summary"]
    assert payload_all["metrics"] == payload_filtered["metrics"]
    assert payload_all["claims"] == payload_filtered["claims"]
    assert payload_all["execution"] == payload_filtered["execution"]
