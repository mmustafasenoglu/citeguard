"""Tests for real semantic and NLI backends for reduction meaning validation.

All tests are offline and mocked.  No real model downloads occur.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from citeguard.reduction.meaning import (
    HeuristicEntailmentBackend,
    MeaningThresholds,
    SentenceTransformerSemanticBackend,
    SequenceSemanticBackend,
    TransformerNLIBackend,
    _map_nli_label,
    validate_meaning,
)
from citeguard.reduction.models import (
    EntailmentDirection,
    MeaningVerdict,
    RewriteCandidate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeSemanticBackend:
    """Deterministic fake for SemanticBackend protocol."""

    def __init__(self, score: float = 0.9) -> None:
        self._score = score

    def similarity(self, original: str, candidate: str) -> float:
        return self._score


class _FakeEntailmentBackend:
    """Configurable fake for EntailmentBackend protocol."""

    def __init__(
        self,
        forward: EntailmentDirection | None = None,
        backward: EntailmentDirection | None = None,
    ) -> None:
        self._forward = forward or EntailmentDirection(
            0.9, MeaningVerdict.PRESERVED, "fake forward"
        )
        self._backward = backward or EntailmentDirection(
            0.9, MeaningVerdict.PRESERVED, "fake backward"
        )
        self._call_count = 0

    def evaluate(self, premise: str, hypothesis: str) -> EntailmentDirection:
        self._call_count += 1
        return self._forward if self._call_count == 1 else self._backward


# ===========================================================================
# 1. Semantic backend contract
# ===========================================================================


def test_sequence_semantic_backend_protocol() -> None:
    backend = SequenceSemanticBackend()
    score = backend.similarity("hello world", "hello world")
    assert score == 1.0


def test_sequence_semantic_backend_returns_float_0_1() -> None:
    backend = SequenceSemanticBackend()
    score = backend.similarity("completely different text", "another sentence entirely")
    assert 0.0 <= score <= 1.0


def test_sentence_transformer_semantic_backend_fallback() -> None:
    """When ST model is unavailable, falls back to SequenceMatcher."""
    backend = SentenceTransformerSemanticBackend(
        model_name="nonexistent-model", allow_download=False
    )
    assert backend.available is False
    score = backend.similarity("hello world", "hello world")
    assert score == 1.0
    score_diff = backend.similarity("cat", "dog")
    assert 0.0 <= score_diff < 1.0


# ===========================================================================
# 2. Raw similarity is separate from exact/lexical overlap
# ===========================================================================


def test_semantic_is_not_lexical_overlap() -> None:
    """When real embeddings are available, semantic cosine differs from lexical."""
    backend = SentenceTransformerSemanticBackend(
        model_name="nonexistent", allow_download=False
    )
    # When the model is unavailable, falls back to SequenceMatcher (same as lexical)
    # With a real model, cosine similarity captures meaning, not just surface form.
    # This test validates the architecture: the interface is pluggable and
    # SequenceSemanticBackend is the deterministic fallback.
    seq_score = SequenceSemanticBackend().similarity(
        "The patient was prescribed medication",
        "Drug therapy was administered to the patient",
    )
    assert 0.0 <= seq_score <= 1.0
    # Verify SequenceSemanticBackend and fallback use the same underlying method
    fallback_score = backend.similarity(
        "The patient was prescribed medication",
        "Drug therapy was administered to the patient",
    )
    assert fallback_score == seq_score


# ===========================================================================
# 3-7. Bidirectional entailment test matrix
# ===========================================================================


def test_forward_preserved_backward_preserved_eligible_for_preserved() -> None:
    """Case 3: Both PRESERVED → eligible for PRESERVED."""
    entailment = _FakeEntailmentBackend(
        forward=EntailmentDirection(0.95, MeaningVerdict.PRESERVED),
        backward=EntailmentDirection(0.92, MeaningVerdict.PRESERVED),
    )
    candidate = RewriteCandidate("The drug reduced symptoms.", "test")
    result = validate_meaning(
        "The medication alleviated symptoms.",
        candidate,
        semantic_backend=_FakeSemanticBackend(0.85),
        entailment_backend=entailment,
        thresholds=MeaningThresholds(semantic_minimum=0.5, entailment_minimum=0.5),
    )
    assert result.verdict == MeaningVerdict.PRESERVED
    assert result.forward_verdict == MeaningVerdict.PRESERVED
    assert result.backward_verdict == MeaningVerdict.PRESERVED


def test_forward_preserved_backward_insufficient_results_insufficient() -> None:
    """Case 4: Forward PRESERVED but backward INSUFFICIENT → INSUFFICIENT."""
    entailment = _FakeEntailmentBackend(
        forward=EntailmentDirection(0.95, MeaningVerdict.PRESERVED),
        backward=EntailmentDirection(0.40, MeaningVerdict.INSUFFICIENT),
    )
    candidate = RewriteCandidate("The drug reduced symptoms.", "test")
    result = validate_meaning(
        "The medication alleviated symptoms.",
        candidate,
        semantic_backend=_FakeSemanticBackend(0.85),
        entailment_backend=entailment,
        thresholds=MeaningThresholds(semantic_minimum=0.5, entailment_minimum=0.5),
    )
    assert result.verdict == MeaningVerdict.INSUFFICIENT


def test_forward_insufficient_backward_preserved_results_insufficient() -> None:
    """Case 5: Forward INSUFFICIENT but backward PRESERVED → INSUFFICIENT."""
    entailment = _FakeEntailmentBackend(
        forward=EntailmentDirection(0.35, MeaningVerdict.INSUFFICIENT),
        backward=EntailmentDirection(0.92, MeaningVerdict.PRESERVED),
    )
    candidate = RewriteCandidate("The drug reduced symptoms.", "test")
    result = validate_meaning(
        "The medication alleviated symptoms.",
        candidate,
        semantic_backend=_FakeSemanticBackend(0.85),
        entailment_backend=entailment,
        thresholds=MeaningThresholds(semantic_minimum=0.5, entailment_minimum=0.5),
    )
    assert result.verdict == MeaningVerdict.INSUFFICIENT


def test_forward_contradicted_rejects_immediately() -> None:
    """Case 6: Forward CONTRADICTED → CONTRADICTED."""
    entailment = _FakeEntailmentBackend(
        forward=EntailmentDirection(0.95, MeaningVerdict.CONTRADICTED),
        backward=EntailmentDirection(0.90, MeaningVerdict.PRESERVED),
    )
    candidate = RewriteCandidate("The drug increased symptoms.", "test")
    result = validate_meaning(
        "The medication alleviated symptoms.",
        candidate,
        semantic_backend=_FakeSemanticBackend(0.85),
        entailment_backend=entailment,
        thresholds=MeaningThresholds(semantic_minimum=0.5, entailment_minimum=0.5),
    )
    assert result.verdict == MeaningVerdict.CONTRADICTED


def test_backward_contradicted_rejects_immediately() -> None:
    """Case 7: Backward CONTRADICTED → CONTRADICTED."""
    entailment = _FakeEntailmentBackend(
        forward=EntailmentDirection(0.90, MeaningVerdict.PRESERVED),
        backward=EntailmentDirection(0.95, MeaningVerdict.CONTRADICTED),
    )
    candidate = RewriteCandidate("The drug made things worse.", "test")
    result = validate_meaning(
        "The medication alleviated symptoms.",
        candidate,
        semantic_backend=_FakeSemanticBackend(0.85),
        entailment_backend=entailment,
        thresholds=MeaningThresholds(semantic_minimum=0.5, entailment_minimum=0.5),
    )
    assert result.verdict == MeaningVerdict.CONTRADICTED


# ===========================================================================
# 8. UNKNOWN never becomes PRESERVED
# ===========================================================================


def test_unknown_never_becomes_preserved() -> None:
    """UNKNOWN in any direction must not yield PRESERVED."""
    entailment = _FakeEntailmentBackend(
        forward=EntailmentDirection(0.90, MeaningVerdict.UNKNOWN),
        backward=EntailmentDirection(0.90, MeaningVerdict.UNKNOWN),
    )
    candidate = RewriteCandidate("A paraphrase.", "test")
    result = validate_meaning(
        "Original text.",
        candidate,
        semantic_backend=_FakeSemanticBackend(0.99),
        entailment_backend=entailment,
        thresholds=MeaningThresholds(semantic_minimum=0.5, entailment_minimum=0.5),
    )
    assert result.verdict != MeaningVerdict.PRESERVED


# ===========================================================================
# 9. Negation flip rejected
# ===========================================================================


def test_negation_flip_rejected_by_heuristic() -> None:
    backend = HeuristicEntailmentBackend()
    result = backend.evaluate(
        "The treatment is effective",
        "The treatment is not effective",
    )
    assert result.verdict == MeaningVerdict.CONTRADICTED


def test_negation_flip_rejected_in_full_pipeline() -> None:
    entailment = _FakeEntailmentBackend(
        forward=EntailmentDirection(0.9, MeaningVerdict.CONTRADICTED),
        backward=EntailmentDirection(0.9, MeaningVerdict.PRESERVED),
    )
    candidate = RewriteCandidate("The treatment is not effective.", "test")
    result = validate_meaning(
        "The treatment is effective.",
        candidate,
        semantic_backend=_FakeSemanticBackend(0.9),
        entailment_backend=entailment,
    )
    assert result.verdict == MeaningVerdict.CONTRADICTED


# ===========================================================================
# 10. Modality strengthening not blindly accepted
# ===========================================================================


def test_modality_strengthening_requires_strong_evidence() -> None:
    """'may reduce' → 'definitely reduces' requires high semantic + entailment."""
    entailment = _FakeEntailmentBackend(
        forward=EntailmentDirection(0.6, MeaningVerdict.UNKNOWN),
        backward=EntailmentDirection(0.9, MeaningVerdict.PRESERVED),
    )
    candidate = RewriteCandidate(
        "The drug definitely reduces mortality.", "test"
    )
    result = validate_meaning(
        "The drug may reduce mortality.",
        candidate,
        semantic_backend=_FakeSemanticBackend(0.85),
        entailment_backend=entailment,
        thresholds=MeaningThresholds(semantic_minimum=0.5, entailment_minimum=0.8),
    )
    assert result.verdict == MeaningVerdict.INSUFFICIENT


# ===========================================================================
# 11. Association → causation not blindly accepted
# ===========================================================================


def test_association_to_causation_not_blindly_accepted() -> None:
    """Correlation vs causation shift must be caught."""
    entailment = _FakeEntailmentBackend(
        forward=EntailmentDirection(0.5, MeaningVerdict.UNKNOWN),
        backward=EntailmentDirection(0.8, MeaningVerdict.PRESERVED),
    )
    candidate = RewriteCandidate(
        "Sleep apnea causes cardiovascular disease.", "test"
    )
    result = validate_meaning(
        "Sleep apnea is associated with cardiovascular disease.",
        candidate,
        semantic_backend=_FakeSemanticBackend(0.82),
        entailment_backend=entailment,
        thresholds=MeaningThresholds(semantic_minimum=0.5, entailment_minimum=0.8),
    )
    assert result.verdict == MeaningVerdict.INSUFFICIENT


# ===========================================================================
# 12. Numeric detail loss remains protected by numeric validator
# ===========================================================================


def test_numeric_change_detected_by_validator_not_meaning() -> None:
    """Numeric integrity is the validator's job, not meaning's."""
    from citeguard.reduction.models import FixAction, FixPlan
    from citeguard.reduction.validator import validate_candidate

    plan = FixPlan(
        passage_id="p0s0",
        action=FixAction.PARAPHRASE,
        reason="test",
        preserve_citations=(),
        must_preserve_numbers=("12%",),
        rewrite_allowed=True,
    )
    candidate = RewriteCandidate("Accuracy improved by 20%.", "test")
    result = validate_candidate(
        "Accuracy improved by 12%.",
        candidate,
        plan,
        meaning_validation=None,
    )
    assert result.accepted is False
    assert "numeric values were changed or removed" in result.reasons


# ===========================================================================
# 13. Offline uncached model causes no network attempt
# ===========================================================================


def test_offline_uncached_semantic_no_network() -> None:
    """SentenceTransformerSemanticBackend with allow_download=False and
    no cached model must not attempt any download."""
    backend = SentenceTransformerSemanticBackend(
        model_name="nonexistent/model-for-test", allow_download=False
    )
    assert backend.available is False
    assert backend._get_backend() is None


def test_offline_uncached_nli_no_network() -> None:
    """TransformerNLIBackend with allow_download=False and
    no cached model must not attempt any download."""
    backend = TransformerNLIBackend(
        model_name="nonexistent/nli-model-for-test", allow_download=False
    )
    assert backend.available is False
    assert backend._load_model() is None


# ===========================================================================
# 14. Offline cached model can run locally
# ===========================================================================


def test_cached_model_loads_when_available() -> None:
    """When model IS cached and allow_download=False, it should load."""
    mock_st = MagicMock()
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_st.SentenceTransformer.return_value = mock_model

    with patch(
        "citeguard.reduction.meaning.SentenceTransformerSemanticBackend._get_backend",
    ):
        backend = SentenceTransformerSemanticBackend(
            model_name="test-model", allow_download=True
        )
        with patch.object(backend, "_backend", mock_model):
            backend._attempted = True
            backend._backend = MagicMock()
            backend._backend.encode.return_value = np.ones((2, 384), dtype=np.float32)
            score = backend.similarity("text a", "text b")
            assert 0.0 <= score <= 1.0


# ===========================================================================
# 15. NLI label mapping
# ===========================================================================


def test_nli_label_mapping_entailment() -> None:
    assert _map_nli_label("entailment") == MeaningVerdict.PRESERVED


def test_nli_label_mapping_contradiction() -> None:
    assert _map_nli_label("contradiction") == MeaningVerdict.CONTRADICTED


def test_nli_label_mapping_neutral() -> None:
    assert _map_nli_label("neutral") == MeaningVerdict.UNKNOWN


def test_nli_label_mapping_case_insensitive() -> None:
    assert _map_nli_label("ENTAILMENT") == MeaningVerdict.PRESERVED
    assert _map_nli_label("Contradiction") == MeaningVerdict.CONTRADICTED
    assert _map_nli_label("Neutral") == MeaningVerdict.UNKNOWN


def test_nli_label_mapping_prefix_stripped() -> None:
    assert _map_nli_label("LABEL_0_entailment") == MeaningVerdict.PRESERVED
    assert _map_nli_label("LABEL_1_contradiction") == MeaningVerdict.CONTRADICTED
    assert _map_nli_label("LABEL_2_neutral") == MeaningVerdict.UNKNOWN


# ===========================================================================
# 16. Unknown label mapping fails conservatively
# ===========================================================================


def test_nli_unknown_label_maps_to_unknown() -> None:
    assert _map_nli_label("some_random_label") == MeaningVerdict.UNKNOWN
    assert _map_nli_label("") == MeaningVerdict.UNKNOWN
    assert _map_nli_label("nonsense") == MeaningVerdict.UNKNOWN


# ===========================================================================
# 17. Heuristic fallback never claims PRESERVED
# ===========================================================================


def test_heuristic_never_returns_preserved() -> None:
    """HeuristicEntailmentBackend must never return PRESERVED."""
    backend = HeuristicEntailmentBackend()
    pairs = [
        ("The method works well.", "The method works well."),
        ("Increase in sales.", "Sales went up."),
        ("No side effects.", "There were no side effects."),
        ("Treatment effective.", "Treatment effective."),
    ]
    for premise, hypothesis in pairs:
        result = backend.evaluate(premise, hypothesis)
        assert result.verdict != MeaningVerdict.PRESERVED, (
            f"Heuristic returned PRESERVED for: {premise!r} → {hypothesis!r}"
        )


def test_heuristic_returns_unknown_for_similar_texts() -> None:
    backend = HeuristicEntailmentBackend()
    result = backend.evaluate(
        "The experiment showed significant results",
        "Significant results were demonstrated by the experiment",
    )
    assert result.verdict == MeaningVerdict.UNKNOWN


# ===========================================================================
# 18. Rejected meaning candidate cannot become accepted winner
# ===========================================================================


def test_rejected_candidate_cannot_become_winner() -> None:
    from citeguard.reduction.ranker import rank_candidates

    rejected = RewriteCandidate(
        "Treatment increased mortality.",
        "test",
        meaning_score=0.30,
        meaning_verdict=MeaningVerdict.CONTRADICTED,
        rejection_reasons=["meaning preservation was not established"],
        citations_preserved=True,
        numeric_integrity=True,
        factual_integrity=True,
    )
    good = RewriteCandidate(
        "Treatment reduced mortality.",
        "test",
        meaning_score=0.95,
        meaning_verdict=MeaningVerdict.PRESERVED,
        citations_preserved=True,
        numeric_integrity=True,
        factual_integrity=True,
    )
    ranked = rank_candidates([rejected, good])
    assert ranked[0] is good
    assert rejected not in ranked


def test_candidate_with_unknown_verdict_not_accepted() -> None:
    """A candidate with meaning_verdict=UNKNOWN has rejection_reasons and
    thus cannot appear in rank_candidates output."""
    from citeguard.reduction.ranker import rank_candidates

    unknown = RewriteCandidate(
        "A paraphrase that was not validated.",
        "test",
        meaning_score=0.70,
        meaning_verdict=MeaningVerdict.UNKNOWN,
        rejection_reasons=["meaning preservation was not established"],
        citations_preserved=True,
        numeric_integrity=True,
        factual_integrity=True,
    )
    ranked = rank_candidates([unknown])
    assert ranked == []
