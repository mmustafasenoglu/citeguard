"""Deterministic tests for attribution-risk reduction primitives."""

from __future__ import annotations

from citeguard.models import ExistingCitation, Sentence, Verdict
from citeguard.reduction import (
    FixAction,
    PassageRisk,
    ReductionRiskType,
    RewriteCandidate,
    build_fix_plans,
    rank_candidates,
    validate_candidate,
)
from citeguard.reduction.analyzer import analyze_passage_risks
from citeguard.reduction.report import reduction_report
from citeguard.similarity.models import (
    MatchType,
    RiskLevel,
    SimilarityEngineResult,
    SimilarityMatch,
    SimilarityResult,
)


def _similarity_result(
    text: str,
    *,
    exact: float,
    lexical: float,
    citation: bool = False,
    risk: RiskLevel = RiskLevel.HIGH,
) -> SimilarityEngineResult:
    citations = (
        [ExistingCitation("(Smith, 2024)", "Smith", 2024, None, None, 0, 10)]
        if citation
        else []
    )
    sentence = Sentence(text, text.lower(), 0, 0, 0, len(text), citations)
    match = SimilarityMatch(
        source_text=text,
        source_title="Example source",
        source_id="source-1",
        exact_overlap=exact,
        lexical_similarity=lexical,
        combined_score=max(exact, lexical),
        match_type=MatchType.NEAR_DUPLICATE,
    )
    item = SimilarityResult(
        sentence=sentence,
        matches=[match],
        best_match=match,
        attribution_risk=risk,
        attribution_reason="high overlap",
    )
    return SimilarityEngineResult([item], 40.0, 1, 0, 1, 1, len(text), len(text))


def test_analyzer_classifies_uncited_exact_overlap_as_citation_fix() -> None:
    risks = analyze_passage_risks(
        _similarity_result(
            "The method improved accuracy by 12%.",
            exact=0.91,
            lexical=0.88,
        )
    )
    assert risks[0].risk_type == ReductionRiskType.EXACT_COPY
    assert risks[0].recommended_action == FixAction.ADD_CITATION


def test_planner_preserves_citations_and_numbers() -> None:
    risk = analyze_passage_risks(
        _similarity_result(
            "The method improved accuracy by 12%. (Smith, 2024)",
            exact=0.75,
            lexical=0.81,
            citation=True,
        )
    )[0]
    plan = build_fix_plans([risk])[0]
    assert plan.action == FixAction.PARAPHRASE
    assert plan.rewrite_allowed is True
    assert plan.preserve_citations == ("(Smith, 2024)",)
    assert plan.must_preserve_numbers == ("12%", "2024")


def test_validator_rejects_numeric_corruption_and_contradiction() -> None:
    plan = build_fix_plans(
        [
            PassageRisk(
                passage_id="p0s0",
                text="Accuracy was 12%. (Smith, 2024)",
                paragraph_index=0,
                start_offset=0,
                end_offset=31,
                exact_overlap=0.75,
                lexical_similarity=0.81,
                semantic_similarity_raw=None,
                attribution_risk=RiskLevel.HIGH,
                has_citation=True,
                citation_verified=None,
                citation_support=None,
                citation_texts=("(Smith, 2024)",),
                risk_type=ReductionRiskType.TOO_CLOSE_PARAPHRASE,
                recommended_action=FixAction.PARAPHRASE,
                confidence=0.9,
            )
        ]
    )[0]
    candidate = RewriteCandidate("Accuracy was 20%.", "test")
    result = validate_candidate(
        "Accuracy was 12%. (Smith, 2024)",
        candidate,
        plan,
        supported_verdict=Verdict.CONTRADICTED,
    )
    assert result.accepted is False
    assert "numeric values were changed or removed" in result.reasons
    assert "candidate contradicts source support" in result.reasons


def test_ranker_prefers_meaning_and_support_over_lower_overlap() -> None:
    safe = RewriteCandidate(
        "The experiment increased accuracy.",
        "test",
        meaning_score=0.96,
        source_support_score=1.0,
        lexical_overlap=0.35,
        exact_overlap=0.10,
        citations_preserved=True,
        numeric_integrity=True,
        factual_integrity=True,
    )
    drifted = RewriteCandidate(
        "The experiment transformed the field.",
        "test",
        meaning_score=0.70,
        source_support_score=0.2,
        lexical_overlap=0.10,
        exact_overlap=0.01,
        citations_preserved=True,
        numeric_integrity=True,
        factual_integrity=True,
    )
    assert rank_candidates([drifted, safe])[0] is safe


def test_reduction_report_is_json_safe() -> None:
    risks = analyze_passage_risks(_similarity_result("Copied text.", exact=0.9, lexical=0.9))
    report = reduction_report(risks, build_fix_plans(risks))
    assert report["schema_version"] == "1"
    assert report["risks"][0]["risk_type"] == "exact_copy"


def test_improve_attribution_is_preview_only(tmp_path) -> None:
    from click.testing import CliRunner

    from citeguard.cli import main

    paper = tmp_path / "paper.txt"
    corpus = tmp_path / "corpus.txt"
    text = "The method improved accuracy by 12%."
    paper.write_text(text, encoding="utf-8")
    corpus.write_text(text, encoding="utf-8")
    before = paper.read_text(encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "improve-attribution",
            str(paper),
            "--corpus",
            str(corpus),
            "--corpus-license",
            "CC0",
            "--dry-run",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    assert result.output
    assert paper.read_text(encoding="utf-8") == before
