"""Deterministic tests for attribution-risk reduction primitives."""
# ruff: noqa: E501

from __future__ import annotations

import pytest

from citeguard.models import ExistingCitation, Sentence, Verdict
from citeguard.reduction import (
    FixAction,
    MeaningThresholds,
    MeaningVerdict,
    PassageRisk,
    ReductionRiskType,
    RewriteCandidate,
    SequenceSemanticBackend,
    TextReplacement,
    apply_text_replacements,
    build_fix_plans,
    compute_reduction_metrics,
    evaluate_candidate,
    generate_candidates,
    rank_candidates,
    restore_text,
    validate_candidate,
    validate_meaning,
)
from citeguard.reduction.analyzer import analyze_passage_risks
from citeguard.reduction.apply import write_revised_text
from citeguard.reduction.meaning import EntailmentDirection
from citeguard.reduction.models import RewriteRequest
from citeguard.reduction.report import reduction_report
from citeguard.reduction.validator import _strengthens_claim
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
        [ExistingCitation("(Smith, 2024)", "Smith", 2024, None, None, 0, 10)] if citation else []
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


def test_protected_tokens_retain_compound_units() -> None:
    from citeguard.reduction.analyzer import extract_protected_tokens

    assert "3,2 mmol/L" in extract_protected_tokens("Düzey 3,2 mmol/L ölçüldü.")
    assert "10 mg/kg" in extract_protected_tokens("Doz 10 mg/kg olarak uygulandı.")


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
        meaning_verdict=MeaningVerdict.PRESERVED,
        source_overlap_improved=True,
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
        meaning_verdict=MeaningVerdict.PRESERVED,
        source_overlap_improved=True,
    )
    assert rank_candidates([drifted, safe])[0] is safe


def test_candidate_generation_is_bounded_and_does_not_accept_candidates() -> None:
    class Backend:
        def generate(self, request):
            return [RewriteCandidate(f"candidate-{i}", "test") for i in range(5)]

    plan = build_fix_plans(
        analyze_passage_risks(
            _similarity_result(
                "The method improved accuracy by 12%. (Smith, 2024)",
                exact=0.75,
                lexical=0.81,
                citation=True,
            )
        )
    )[0]
    candidates = generate_candidates(
        Backend(),
        RewriteRequest(plan=plan, original_text=plan.reason, candidate_count=2),
    )
    assert len(candidates) == 2
    assert all(candidate.rejection_reasons == [] for candidate in candidates)


def test_evaluator_populates_overlap_measurements() -> None:
    candidate = evaluate_candidate(
        "The method improved accuracy.",
        RewriteCandidate("The method increased accuracy.", "test"),
    )
    assert candidate.lexical_overlap is not None
    assert candidate.exact_overlap is not None
    assert candidate.meaning_score == candidate.semantic_similarity_to_original


def test_validator_requires_exact_citation_text() -> None:
    plan = build_fix_plans(
        analyze_passage_risks(
            _similarity_result(
                "A result was reported. (Smith, 2024)",
                exact=0.75,
                lexical=0.81,
                citation=True,
            )
        )
    )[0]
    candidate = RewriteCandidate("A result was reported. (Jones, 2024)", "test")
    result = validate_candidate(
        "A result was reported. (Smith, 2024)",
        candidate,
        plan,
    )
    assert result.accepted is False
    assert result.citations_preserved is False


@pytest.mark.parametrize(
    ("original", "candidate"),
    [
        ("Maruziyet sonuçla ilişkili bulundu.", "Maruziyet sonuca neden oldu."),
        ("Politika yararlı olabilir.", "Politika kesin olarak yararlıdır."),
        ("Bazı katılımcılar iyileşti.", "Tüm katılımcılar iyileşti."),
        ("Ön bulgular ilişkiye işaret etmektedir.", "İlişki kanıtlanmıştır."),
        (
            "Maruziyet sonuçla ilişkili bulundu.",
            "Maruziyet sonuca neden oldu; ekonomik etkisi kanıtlanmamıştır.",
        ),
        (
            "Politika etkili olabilir.",
            "Politika kesin olarak etkilidir; yan etkileri gösterilmemiştir.",
        ),
        (
            "Bazı öğrenciler iyileşti.",
            "Tüm öğrenciler iyileşti; nedenleri bilinmemektedir.",
        ),
    ],
)
def test_validator_rejects_turkish_claim_strengthening(original: str, candidate: str) -> None:
    risk = PassageRisk(
        passage_id="p0s0",
        text=original,
        paragraph_index=0,
        start_offset=0,
        end_offset=len(original),
        exact_overlap=0.8,
        lexical_similarity=0.8,
        semantic_similarity_raw=None,
        attribution_risk=RiskLevel.HIGH,
        has_citation=False,
        citation_verified=None,
        citation_support=None,
        citation_texts=(),
        risk_type=ReductionRiskType.TOO_CLOSE_PARAPHRASE,
        recommended_action=FixAction.PARAPHRASE,
        confidence=0.9,
    )
    plan = build_fix_plans([risk])[0]
    result = validate_candidate(original, RewriteCandidate(candidate, "test"), plan)
    assert result.factual_integrity is False
    assert "candidate strengthens the claim beyond the original" in result.reasons


def test_meaning_requires_bidirectional_entailment() -> None:
    class Backend:
        def __init__(self, verdicts):
            self.verdicts = iter(verdicts)

        def evaluate(self, _premise, _hypothesis):
            return next(self.verdicts)

    candidate = RewriteCandidate("Treatment reduced mortality by 12%.", "test")
    result = validate_meaning(
        "Treatment reduced mortality by 12%.",
        candidate,
        semantic_backend=SequenceSemanticBackend(),
        entailment_backend=Backend(
            [
                EntailmentDirection(0.95, MeaningVerdict.PRESERVED),
                EntailmentDirection(0.40, MeaningVerdict.INSUFFICIENT),
            ]
        ),
        thresholds=MeaningThresholds(semantic_minimum=0.8, entailment_minimum=0.8),
    )
    assert result.verdict == MeaningVerdict.INSUFFICIENT
    assert result.backward_verdict == MeaningVerdict.INSUFFICIENT
    assert candidate.meaning_verdict == MeaningVerdict.INSUFFICIENT


def test_meaning_rejects_contradiction_in_either_direction() -> None:
    class Backend:
        def evaluate(self, _premise, _hypothesis):
            return EntailmentDirection(0.9, MeaningVerdict.CONTRADICTED)

    candidate = RewriteCandidate("Treatment increased mortality.", "test")
    result = validate_meaning(
        "Treatment reduced mortality.",
        candidate,
        semantic_backend=SequenceSemanticBackend(),
        entailment_backend=Backend(),
    )
    assert result.verdict == MeaningVerdict.CONTRADICTED
    assert candidate.rejection_reasons


def test_integrity_validator_applies_meaning_gate() -> None:
    plan = build_fix_plans(
        analyze_passage_risks(
            _similarity_result(
                "The method improved accuracy. (Smith, 2024)",
                exact=0.75,
                lexical=0.81,
                citation=True,
            )
        )
    )[0]
    candidate = RewriteCandidate("The method improved accuracy. (Smith, 2024)", "test")
    meaning = validate_meaning(
        "The method improved accuracy. (Smith, 2024)",
        candidate,
        semantic_backend=SequenceSemanticBackend(),
        entailment_backend=type(
            "Backend",
            (),
            {
                "evaluate": lambda _self, _premise, _hypothesis: EntailmentDirection(
                    0.0, MeaningVerdict.UNKNOWN
                )
            },
        )(),
    )
    result = validate_candidate(
        "The method improved accuracy. (Smith, 2024)",
        candidate,
        plan,
        meaning_validation=meaning,
    )
    assert result.accepted is False


@pytest.mark.parametrize(
    ("original", "candidate"),
    [
        (
            "Maruziyet ile sonuç arasında ilişki gözlendi.",
            "Maruziyetin sonuca neden olduğu gösterilmemiştir.",
        ),
        ("Politika başarıyı artırabilir.", "Politikanın kesin olduğu gösterilmemiştir."),
        ("Bazı katılımcılar iyileşti.", "Tüm katılımcılarda iyileşme görülmedi."),
        ("Bazı öğrenciler iyileşti.", "Tüm öğrenciler iyileştiği söylenemez."),
    ],
)
def test_turkish_negated_strengthening_is_not_flagged(original: str, candidate: str) -> None:
    assert not _strengthens_claim(original, candidate)


@pytest.mark.parametrize(
    ("original", "candidate"),
    [
        ("Exposure correlated with the outcome.", "Exposure led to the outcome."),
        ("Limited evidence indicates an effect.", "This proves an effect."),
        (
            "X ile Y arasında birlikte değişim gözlendi.",
            "X, Y'ye yol açar.",
        ),
        (
            "İlk bulgular bir etkiye işaret etmektedir.",
            "Etki kesin olarak gösterilmiştir.",
        ),
    ],
)
def test_validator_rejects_general_association_and_evidence_strengthening(
    original: str, candidate: str
) -> None:
    assert _strengthens_claim(original, candidate)


@pytest.mark.parametrize(
    ("original", "candidate"),
    [
        (
            "X ile Y arasında ilişki bulundu.",
            "X'in Y'ye neden olduğu kanıtlanmamıştır.",
        ),
        (
            "X ile Y arasında ilişki bulundu.",
            "X, Y'ye neden olur; mekanizması kanıtlanmamıştır.",
        ),
        (
            "İlk bulgular bir ilişkiye işaret ediyor.",
            "İlişki kesin değildir.",
        ),
        (
            "İlk bulgular bir ilişkiye işaret ediyor.",
            "İlişki kesin olarak etkilidir; nedeni bilinmemektedir.",
        ),
    ],
)
def test_clause_local_negation_neither_creates_nor_hides_strengthening(
    original: str, candidate: str
) -> None:
    expected = "neden olur" in candidate or "kesin olarak etkilidir" in candidate
    assert _strengthens_claim(original, candidate) is expected


def test_offline_heuristic_does_not_claim_entailment() -> None:
    from citeguard.reduction.meaning import HeuristicEntailmentBackend

    result = HeuristicEntailmentBackend().evaluate(
        "The method improved accuracy.",
        "The method increased accuracy.",
    )
    assert result.verdict == MeaningVerdict.UNKNOWN


def test_text_patch_is_exact_and_reversible() -> None:
    original = "First claim. Second claim."
    revised, applied = apply_text_replacements(
        original,
        [TextReplacement("Second claim.", "Rewritten claim.", "p0s1")],
    )
    assert revised == "First claim. Rewritten claim."
    assert restore_text(revised, applied) == original


def test_text_patch_rejects_ambiguous_original() -> None:
    from pytest import raises

    with raises(ValueError, match="ambiguous"):
        apply_text_replacements(
            "Same sentence. Same sentence.",
            [TextReplacement("Same sentence.", "Changed.", "p0s0")],
        )


def test_text_writer_refuses_source_overwrite_and_existing_output(tmp_path) -> None:
    source = tmp_path / "paper.md"
    source.write_text("Original claim.", encoding="utf-8")
    replacement = [TextReplacement("Original claim.", "Revised claim.", "p0s0")]
    from pytest import raises

    with raises(ValueError, match="differ"):
        write_revised_text(source, source, replacement)

    output = tmp_path / "existing.md"
    output.write_text("Do not overwrite.", encoding="utf-8")
    with raises(ValueError, match="already exists"):
        write_revised_text(source, output, replacement)
    assert source.read_text(encoding="utf-8") == "Original claim."
    assert output.read_text(encoding="utf-8") == "Do not overwrite."


def test_restore_text_refuses_duplicate_replacement_at_wrong_recorded_offset() -> None:
    original = "First claim. Second claim."
    revised, applied = apply_text_replacements(
        original,
        [TextReplacement("Second claim.", "Rewritten claim.", "p0s1")],
    )

    # The same replacement text elsewhere must not be chosen by a global find.
    edited = "Rewritten claim. " + revised
    from pytest import raises

    with raises(ValueError, match="refusing an ambiguous restore"):
        restore_text(edited, applied)


def test_reduction_metrics_use_absolute_and_relative_overlap() -> None:
    before = _similarity_result("Before.", exact=0.9, lexical=0.9)
    after = _similarity_result("After.", exact=0.2, lexical=0.2)
    before.overall_similarity_pct = 26.8
    after.overall_similarity_pct = 12.4
    metrics = compute_reduction_metrics(before, after, rewritten_passages=2)
    assert metrics.absolute_reduction_pct == 14.4
    assert round(metrics.relative_reduction_pct or 0, 3) == 53.731
    assert metrics.rewritten_passages == 2


def test_reduction_report_is_json_safe() -> None:
    risks = analyze_passage_risks(_similarity_result("Copied text.", exact=0.9, lexical=0.9))
    report = reduction_report(risks, build_fix_plans(risks))
    assert report["schema_version"] == "2"
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
