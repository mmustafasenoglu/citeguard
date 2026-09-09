"""Safe integration tests for grounded rewrite proposals and reduction gates."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from citeguard.models import Verdict
from citeguard.reduction import (
    FixAction,
    FixPlan,
    MeaningThresholds,
    MeaningVerdict,
    SequenceSemanticBackend,
    evaluate_candidate,
    rank_candidates,
    validate_candidate,
    validate_meaning,
)
from citeguard.reduction.meaning import EntailmentDirection
from citeguard.reduction.models import RewriteRequest as ReductionRewriteRequest
from citeguard.rewrite import (
    EvidenceGroundedReductionBackend,
    RewriteContext,
    RewriteMode,
    RewriteResult,
    RewriteStatus,
)
from citeguard.rewrite.models import RewriteRequest as GroundedRewriteRequest

ORIGINAL = "Treatment reduced mortality by 12% (Smith, 2024)."
CITATION = "(Smith, 2024)"


def _plan(*, rewrite_allowed: bool = True) -> FixPlan:
    return FixPlan(
        passage_id="p0s0",
        action=FixAction.PARAPHRASE,
        reason="too close",
        preserve_citations=(CITATION,),
        must_preserve_numbers=("12%", "2024"),
        rewrite_allowed=rewrite_allowed,
    )


def _grounded_request() -> GroundedRewriteRequest:
    return GroundedRewriteRequest(
        context=RewriteContext(
            original_text=ORIGINAL,
            claim_text="Treatment reduced mortality by 12%.",
            claim_type="quantitative",
            citation_raw=CITATION,
            verification_status="resolved",
            verdict="supported",
            source_title="Trial",
            source_authors=("Smith",),
            source_year=2024,
            evidence_texts=("Mortality was reduced by 12%.",),
        ),
        mode=RewriteMode.CLARIFY,
    )


def _result(status: RewriteStatus, text: str | None = None) -> RewriteResult:
    return RewriteResult(
        original_text=ORIGINAL,
        rewritten_text=text,
        status=status,
        provider="test-provider",
        model="test-model",
    )


def _reduction_request(*, count: int = 1, rewrite_allowed: bool = True):
    return ReductionRewriteRequest(
        original_text=ORIGINAL,
        plan=_plan(rewrite_allowed=rewrite_allowed),
        candidate_count=count,
    )


def test_success_becomes_only_an_unvalidated_candidate() -> None:
    provider = MagicMock()
    provider.rewrite.return_value = _result(
        RewriteStatus.SUCCESS,
        "The treatment lowered mortality by 12% (Smith, 2024).",
    )
    backend = EvidenceGroundedReductionBackend(
        provider, {"p0s0": _grounded_request()}
    )

    candidates = backend.generate(_reduction_request())

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.generator == "test-provider:test-model"
    assert candidate.meaning_verdict is None
    assert candidate.numeric_integrity is None
    assert candidate.factual_integrity is None
    assert rank_candidates(candidates) == []


@pytest.mark.parametrize(
    "status",
    [
        RewriteStatus.PROVIDER_ERROR,
        RewriteStatus.INVALID_RESPONSE,
        RewriteStatus.INSUFFICIENT_EVIDENCE,
        RewriteStatus.UNAVAILABLE,
        RewriteStatus.REFUSED,
    ],
)
def test_provider_non_success_produces_no_candidate(status: RewriteStatus) -> None:
    provider = MagicMock()
    provider.rewrite.return_value = _result(status)
    backend = EvidenceGroundedReductionBackend(
        provider, {"p0s0": _grounded_request()}
    )
    assert backend.generate(_reduction_request()) == []


def test_provider_exception_produces_no_candidate() -> None:
    provider = MagicMock()
    provider.rewrite.side_effect = RuntimeError("transport failure")
    backend = EvidenceGroundedReductionBackend(
        provider, {"p0s0": _grounded_request()}
    )
    assert backend.generate(_reduction_request()) == []


@pytest.mark.parametrize("condition", ["missing", "disabled", "offline"])
def test_fail_closed_conditions_make_zero_provider_calls(condition: str) -> None:
    provider = MagicMock()
    mapping = {} if condition == "missing" else {"p0s0": _grounded_request()}
    backend = EvidenceGroundedReductionBackend(
        provider, mapping, offline=condition == "offline"
    )
    request = _reduction_request(rewrite_allowed=condition != "disabled")
    assert backend.generate(request) == []
    provider.rewrite.assert_not_called()


def test_candidate_count_bounds_provider_calls_and_results() -> None:
    provider = MagicMock()
    provider.rewrite.return_value = _result(RewriteStatus.SUCCESS, ORIGINAL)
    backend = EvidenceGroundedReductionBackend(
        provider, {"p0s0": _grounded_request()}
    )
    assert len(backend.generate(_reduction_request(count=2))) == 2
    assert provider.rewrite.call_count == 2


class _EntailmentBackend:
    def __init__(self, verdict: MeaningVerdict) -> None:
        self.verdict = verdict

    def evaluate(self, _premise: str, _hypothesis: str) -> EntailmentDirection:
        score = 0.95 if self.verdict == MeaningVerdict.PRESERVED else 0.10
        return EntailmentDirection(score, self.verdict)


def _run_safety_path(candidate, *, meaning: MeaningVerdict):
    evaluate_candidate(ORIGINAL, candidate)
    meaning_result = validate_meaning(
        ORIGINAL,
        candidate,
        semantic_backend=SequenceSemanticBackend(),
        entailment_backend=_EntailmentBackend(meaning),
        thresholds=MeaningThresholds(semantic_minimum=0.0, entailment_minimum=0.8),
    )
    validate_candidate(
        ORIGINAL,
        candidate,
        _plan(),
        supported_verdict=Verdict.SUPPORTED,
        meaning_validation=meaning_result,
    )
    return rank_candidates([candidate])


def test_provider_success_cannot_bypass_numeric_or_meaning_gates() -> None:
    provider = MagicMock()
    provider.rewrite.return_value = _result(
        RewriteStatus.SUCCESS, "Treatment reduced mortality (Smith, 2024)."
    )
    backend = EvidenceGroundedReductionBackend(
        provider, {"p0s0": _grounded_request()}
    )
    candidate = backend.generate(_reduction_request())[0]
    assert _run_safety_path(candidate, meaning=MeaningVerdict.CONTRADICTED) == []
    assert candidate.numeric_integrity is False
    assert candidate.meaning_verdict == MeaningVerdict.CONTRADICTED


def test_lower_overlap_unsafe_candidate_cannot_beat_validated_candidate() -> None:
    unsafe = _result(RewriteStatus.SUCCESS, "A caused B (Smith, 2024).")
    safe = _result(
        RewriteStatus.SUCCESS,
        "Treatment reduced mortality by 12% (Smith, 2024).",
    )
    provider = MagicMock()
    provider.rewrite.side_effect = [unsafe, safe]
    backend = EvidenceGroundedReductionBackend(
        provider, {"p0s0": _grounded_request()}
    )
    unsafe_candidate, safe_candidate = backend.generate(_reduction_request(count=2))

    _run_safety_path(unsafe_candidate, meaning=MeaningVerdict.CONTRADICTED)
    _run_safety_path(safe_candidate, meaning=MeaningVerdict.PRESERVED)
    ranked = rank_candidates([unsafe_candidate, safe_candidate])

    assert ranked == [safe_candidate]
    assert unsafe_candidate.lexical_overlap < safe_candidate.lexical_overlap
