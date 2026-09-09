"""End-to-end plagiarism review regression tests."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
from click.testing import CliRunner
from docx import Document

from citeguard.cli import main
from citeguard.models import SourceCandidate
from citeguard.plagiarism.models import AttributionStatus, PlagiarismConfig
from citeguard.plagiarism.pipeline import scan_document
from citeguard.plagiarism.report import plagiarism_markdown, plagiarism_report
from citeguard.plagiarism.sources import fetch_url_text
from citeguard.reduction.models import FixAction, FixPlan, RewriteCandidate
from citeguard.reduction.validator import validate_candidate

SOURCE = (
    "Large language models frequently produce plausible but nonexistent citations. "
    "Independent verification is therefore necessary before scholarly use."
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _scan(tmp_path: Path, document: str, source: str = SOURCE, **kwargs):
    document_path = _write(tmp_path / "paper.md", document)
    source_path = _write(tmp_path / "source.txt", source)
    config = PlagiarismConfig(min_match_words=4, **kwargs)
    return scan_document(document_path, sources=[source_path], config=config)


def test_exact_copy_has_auditable_span_and_coverage(tmp_path: Path) -> None:
    result = _scan(tmp_path, SOURCE)
    match = result.matches[0]
    assert match.match_type == "exact"
    assert match.document_paragraph_end - match.document_paragraph_start == len(match.document_text)
    assert match.source_start >= 0 and match.source_end > match.source_start
    assert result.summary.raw_similarity_percent > 90
    assert result.summary.review_similarity_percent <= result.summary.raw_similarity_percent
    assert match.attribution_status == AttributionStatus.UNATTRIBUTED_EXACT


def test_near_copy_and_unrelated_text_are_distinguished(tmp_path: Path) -> None:
    near = SOURCE.replace("frequently", "often").replace("plausible", "credible")
    result = _scan(tmp_path, near, lexical_threshold=0.35)
    assert any(
        match.match_type in {"near_duplicate", "lexical_overlap"} for match in result.matches
    )
    unrelated = _scan(
        tmp_path,
        "Volcanic minerals cool into crystalline structures beneath the ocean.",
    )
    assert unrelated.summary.raw_similarity_percent == 0


def test_quote_and_citation_states_affect_review_score(tmp_path: Path) -> None:
    quoted = f"“{SOURCE}” (Smith, 2020)."
    result = _scan(tmp_path, quoted)
    assert any(match.quoted and match.citation_present for match in result.matches)
    assert result.summary.review_similarity_percent == 0
    included = _scan(tmp_path, quoted, include_quotes=True)
    assert included.summary.review_similarity_percent > 0


def test_cited_verbatim_is_not_treated_as_safe(tmp_path: Path) -> None:
    result = _scan(tmp_path, f"{SOURCE} (Smith, 2020).", lexical_threshold=0.35)
    statuses = {match.attribution_status for match in result.matches}
    assert AttributionStatus.CITED_BUT_VERBATIM in statuses
    assert result.summary.review_similarity_percent > 0


def test_bibliography_is_visible_but_excluded_by_default(tmp_path: Path) -> None:
    document = f"Original discussion has no copied wording.\n\n# References\n\n{SOURCE}"
    result = _scan(tmp_path, document)
    assert any(match.bibliography_region for match in result.matches)
    assert result.summary.bibliography_similarity_percent > 0
    assert result.summary.review_similarity_percent == 0
    included = _scan(tmp_path, document, include_bibliography=True)
    assert included.summary.review_similarity_percent > 0


def test_duplicate_sources_do_not_double_count_coverage(tmp_path: Path) -> None:
    document = _write(tmp_path / "paper.md", SOURCE)
    one = _write(tmp_path / "one.txt", SOURCE)
    two = _write(tmp_path / "two.txt", SOURCE)
    result = scan_document(
        document,
        sources=[one, two],
        config=PlagiarismConfig(min_match_words=4),
    )
    assert result.summary.raw_similarity_percent <= 100
    assert sum(item.percent for item in result.source_contributions) <= 100
    assert len(result.sources) == 2


def test_generic_phrase_is_suppressed_across_sources(tmp_path: Path) -> None:
    phrase = "Further research is needed."
    document = _write(tmp_path / "paper.md", phrase)
    sources = [_write(tmp_path / f"source-{index}.txt", phrase) for index in range(3)]
    result = scan_document(
        document,
        sources=sources,
        config=PlagiarismConfig(min_match_words=3),
    )
    assert result.matches
    assert all(
        match.attribution_status == AttributionStatus.POSSIBLE_COMMON_PHRASE
        for match in result.matches
    )
    assert result.summary.review_similarity_percent == 0


def test_directory_txt_markdown_and_docx_sources(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write(corpus / "a.txt", SOURCE)
    _write(corpus / "b.md", "Distinct source text with enough words for indexing.")
    docx = Document()
    docx.add_paragraph("A third distinct source passage with enough words for indexing.")
    docx.save(corpus / "c.docx")
    document = _write(tmp_path / "paper.md", SOURCE)
    result = scan_document(
        document,
        corpora=[corpus],
        config=PlagiarismConfig(min_match_words=4),
    )
    assert {source.path and Path(source.path).suffix for source in result.sources} == {
        ".txt",
        ".md",
        ".docx",
    }


def test_offline_scan_makes_zero_network_calls(tmp_path: Path, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    result = _scan(tmp_path, SOURCE, offline=True)
    assert result.matches


def test_semantic_dependency_failure_falls_back(tmp_path: Path, monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        raise ImportError("missing semantic dependency")

    monkeypatch.setattr("citeguard.plagiarism.pipeline._with_embeddings", unavailable)
    result = _scan(tmp_path, SOURCE, semantic=True)
    assert result.matches
    assert any("lexical fallback" in warning for warning in result.warnings)


def test_json_is_deterministic_and_markdown_documents_method(tmp_path: Path) -> None:
    result = _scan(tmp_path, SOURCE)
    first = json.dumps(plagiarism_report(result), sort_keys=True)
    second = json.dumps(plagiarism_report(result), sort_keys=True)
    assert first == second
    markdown = plagiarism_markdown(result)
    assert "unique covered document words" in markdown
    assert "not a Turnitin" in markdown


def test_cli_terminal_json_and_source_url_offline_guard(tmp_path: Path) -> None:
    runner = CliRunner()
    document = _write(tmp_path / "paper.md", SOURCE)
    source = _write(tmp_path / "source.txt", SOURCE)
    terminal = runner.invoke(main, ["plagiarism", str(document), "--source", str(source)])
    assert terminal.exit_code == 0
    assert "Review-relevant similarity" in terminal.output
    json_result = runner.invoke(
        main,
        ["plagiarism", str(document), "--source", str(source), "--format", "json"],
    )
    assert json_result.exit_code == 0
    assert json.loads(json_result.output)["schema_version"] == "1.0"
    guarded = runner.invoke(
        main,
        ["plagiarism", str(document), "--source-url", "https://example.test/source"],
    )
    assert guarded.exit_code == 2
    assert "requires --online" in guarded.output


def test_malformed_empty_and_missing_inputs(tmp_path: Path) -> None:
    empty = _write(tmp_path / "empty.txt", " \n")
    source = _write(tmp_path / "source.txt", SOURCE)
    with pytest.raises(ValueError, match="empty"):
        scan_document(empty, sources=[source])
    runner = CliRunner()
    missing = runner.invoke(
        main,
        ["plagiarism", str(tmp_path / "missing.md"), "--source", str(source)],
    )
    assert missing.exit_code == 2


def test_improve_attribution_accepts_plagiarism_report(tmp_path: Path) -> None:
    runner = CliRunner()
    document = _write(tmp_path / "paper.md", SOURCE)
    source = _write(tmp_path / "source.txt", SOURCE)
    report_path = tmp_path / "review.json"
    scan = runner.invoke(
        main,
        [
            "plagiarism",
            str(document),
            "--source",
            str(source),
            "--format",
            "json",
            "--output",
            str(report_path),
        ],
    )
    assert scan.exit_code == 0
    preview = runner.invoke(
        main,
        ["improve-attribution", str(document), "--from-report", str(report_path), "--dry-run"],
    )
    assert preview.exit_code == 0
    assert "preview_only" in preview.output
    assert "Plagiarism review rescan" in preview.output

    original_bytes = document.read_bytes()
    revised = tmp_path / "paper.revised.md"
    applied = runner.invoke(
        main,
        [
            "improve-attribution",
            str(document),
            "--from-report",
            str(report_path),
            "--apply",
            "--output",
            str(revised),
            "--offline",
            "--format",
            "json",
        ],
    )
    assert applied.exit_code == 0
    payload = json.loads(applied.output)
    assert revised.exists() and revised.read_text(encoding="utf-8") == SOURCE
    assert document.read_bytes() == original_bytes
    assert payload["plagiarism_rescan"]["same_configuration"] is True
    assert payload["plagiarism_rescan"]["manual_reviews_remaining"] >= 1


def test_url_size_content_type_and_no_cache_controls(tmp_path: Path, monkeypatch) -> None:
    class Headers:
        def get_content_type(self):
            return "text/plain"

        def get_content_charset(self):
            return "utf-8"

    class Response:
        headers = Headers()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, maximum):
            return b"source text larger than limit"

    class Opener:
        def open(self, request, timeout):
            return Response()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "citeguard.plagiarism.sources.socket.getaddrinfo",
        lambda *args: [(None, None, None, None, ("93.184.216.34", 443))],
    )
    monkeypatch.setattr(
        "citeguard.plagiarism.sources.urllib.request.build_opener", lambda *a: Opener()
    )
    with pytest.raises(ValueError, match="maximum size"):
        fetch_url_text(
            "https://example.test/source",
            PlagiarismConfig(offline=False, no_cache=True, url_max_bytes=5),
        )
    assert not Path(".local").exists()


def test_private_url_is_rejected_before_fetch(monkeypatch) -> None:
    monkeypatch.setattr(
        "citeguard.plagiarism.sources.socket.getaddrinfo",
        lambda *args: [(None, None, None, None, ("10.0.0.4", 443))],
    )
    with pytest.raises(ValueError, match="Private"):
        fetch_url_text("https://internal.test/source", PlagiarismConfig(offline=False))


def test_local_source_size_guard(tmp_path: Path) -> None:
    document = _write(tmp_path / "paper.md", SOURCE)
    source = _write(tmp_path / "source.txt", SOURCE)
    with pytest.raises(ValueError, match="maximum size"):
        scan_document(
            document,
            sources=[source],
            config=PlagiarismConfig(local_source_max_bytes=4),
        )


def test_no_cache_scan_writes_no_index(tmp_path: Path, monkeypatch) -> None:
    document = _write(tmp_path / "paper.md", SOURCE)
    source = _write(tmp_path / "source.txt", SOURCE)
    monkeypatch.chdir(tmp_path)
    result = scan_document(
        document,
        sources=[source],
        config=PlagiarismConfig(min_match_words=4, no_cache=True),
    )
    assert result.matches
    assert not Path(".local").exists()


def test_corpus_index_command_creates_reusable_artifacts(tmp_path: Path) -> None:
    runner = CliRunner()
    corpus = _write(tmp_path / "source.txt", SOURCE)
    output = tmp_path / "source.ctac"
    result = runner.invoke(main, ["corpus", "index", str(corpus), "--output", str(output)])
    assert result.exit_code == 0
    assert (output / "manifest.json").exists()
    assert (output / "entries.jsonl").exists()
    assert (output / "tfidf_matrix.npz").exists()


def test_academic_discovery_counts_only_returned_abstract_text(tmp_path: Path, monkeypatch) -> None:
    def search(engine, query, max_results):
        assert "Smith" in query
        return [
            SourceCandidate(
                "Reference Verification",
                ["A. Smith"],
                2020,
                None,
                "10.1000/example",
                "https://example.test/work",
                SOURCE,
                "semantic_scholar",
            )
        ]

    monkeypatch.setattr("citeguard.retrieval.RetrievalEngine.search", search)
    document = _write(
        tmp_path / "paper.md",
        f"{SOURCE}\n\n# References\n\nSmith, A. (2020). Reference Verification.",
    )
    result = scan_document(
        document,
        config=PlagiarismConfig(
            min_match_words=4,
            offline=False,
            discover_academic=True,
            no_cache=True,
        ),
    )
    discovered = [source for source in result.sources if source.source_discovered]
    assert discovered and discovered[0].textual_content_compared is True
    assert result.summary.raw_similarity_percent > 0


def test_rewrite_rejects_removed_named_entities() -> None:
    plan = FixPlan("p0s0", FixAction.PARAPHRASE, "test", rewrite_allowed=True)
    candidate = RewriteCandidate(
        "The model was evaluated by another laboratory.", generator="test:model"
    )
    result = validate_candidate(
        "OpenAI Research evaluated GPT models in Ankara University.", candidate, plan
    )
    assert result.accepted is False
    assert result.named_entity_integrity is False
    assert "important named entities were changed or removed" in result.reasons


def test_document_and_source_global_offsets_survive_cached_index(tmp_path: Path) -> None:
    prefix = "An unrelated introductory sentence appears before the copied passage."
    document = _write(tmp_path / "paper.md", f"Original opening paragraph.\n\n{SOURCE}")
    source = _write(tmp_path / "source.txt", f"{prefix}\n\n{SOURCE}")
    config = PlagiarismConfig(min_match_words=4)
    first = scan_document(document, sources=[source], config=config)
    second = scan_document(document, sources=[source], config=config)
    match = next(item for item in second.matches if item.match_type == "exact")
    assert match.document_paragraph == 1
    assert match.document_start > match.document_paragraph_start
    assert match.source_start > 0
    assert plagiarism_report(first)["summary"] == plagiarism_report(second)["summary"]
