"""Tests for real semantic and NLI backends for reduction meaning validation.

All tests are offline and mocked.  No real model downloads occur.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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


def test_unavailable_semantic_backend_raises_explicit_error() -> None:
    """An unavailable real semantic backend must NOT silently return
    SequenceMatcher output as semantic evidence."""
    from citeguard.reduction.meaning import LocalModelUnavailableError

    backend = SentenceTransformerSemanticBackend(
        model_name="nonexistent-model", allow_download=False
    )
    assert backend.available is False
    with pytest.raises(LocalModelUnavailableError):
        backend.similarity("hello world", "hello world")


def test_unavailable_semantic_is_distinguishable_from_sequence_fallback() -> None:
    """REAL_SEMANTIC vs SEQUENCE_FALLBACK vs UNAVAILABLE are distinct."""
    from citeguard.reduction.meaning import LocalModelUnavailableError

    unavailable = SentenceTransformerSemanticBackend(
        model_name="nonexistent-model", allow_download=False
    )
    fallback = SequenceSemanticBackend()
    # Fallback still works as an explicit, separate screening backend.
    assert fallback.similarity("hello world", "hello world") == 1.0
    # Unavailable real backend raises instead of impersonating the fallback.
    with pytest.raises(LocalModelUnavailableError):
        unavailable.similarity("hello world", "hello world")


def test_negative_cosine_remains_negative() -> None:
    """Raw cosine < 0 must NOT be clamped to 0."""
    from citeguard.similarity.embeddings import sentence_transformers as st_mod

    vectors = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32
    )

    class _FakeSTBackend:
        def encode(self, texts: list[str]) -> np.ndarray:
            return vectors

    backend = SentenceTransformerSemanticBackend(
        model_name="test-model", allow_download=True
    )
    with patch.object(
        st_mod, "SentenceTransformerBackend", lambda **kw: _FakeSTBackend()
    ):
        score = backend.similarity("original text", "opposite text")
    assert score == pytest.approx(-1.0)
    assert score < 0


def test_positive_cosine_returned_unchanged() -> None:
    """Raw cosine values pass through without rescaling."""
    from citeguard.similarity.embeddings import sentence_transformers as st_mod

    vectors = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
    )

    class _FakeSTBackend:
        def encode(self, texts: list[str]) -> np.ndarray:
            return vectors

    backend = SentenceTransformerSemanticBackend(
        model_name="test-model", allow_download=True
    )
    with patch.object(
        st_mod, "SentenceTransformerBackend", lambda **kw: _FakeSTBackend()
    ):
        identical = backend.similarity("same text", "same text")
    assert identical == pytest.approx(1.0)

    orthogonal = np.array(
        [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )

    class _FakeSTOrtho:
        def encode(self, texts: list[str]) -> np.ndarray:
            return orthogonal

    backend2 = SentenceTransformerSemanticBackend(
        model_name="test-model", allow_download=True
    )
    with patch.object(
        st_mod, "SentenceTransformerBackend", lambda **kw: _FakeSTOrtho()
    ):
        score = backend2.similarity("text a", "text b")
    assert score == pytest.approx(0.0)


# ===========================================================================
# 2. Raw similarity is separate from exact/lexical overlap
# ===========================================================================


def test_semantic_evidence_never_touches_textual_overlap_metrics() -> None:
    """MeaningValidation carries only semantic/entailment evidence.

    It must never contain exact/lexical overlap fields or
    overall_similarity_pct — semantic evidence is a separate signal.
    """
    candidate = RewriteCandidate("The drug reduced symptoms.", "test")
    result = validate_meaning(
        "The medication alleviated symptoms.",
        candidate,
        semantic_backend=_FakeSemanticBackend(-0.25),
        entailment_backend=_FakeEntailmentBackend(
            forward=EntailmentDirection(0.9, MeaningVerdict.PRESERVED),
            backward=EntailmentDirection(0.9, MeaningVerdict.PRESERVED),
        ),
        thresholds=MeaningThresholds(semantic_minimum=0.5, entailment_minimum=0.5),
    )
    assert result.semantic_similarity_raw == pytest.approx(-0.25)
    assert not hasattr(result, "exact_overlap")
    assert not hasattr(result, "lexical_similarity")
    assert not hasattr(result, "overall_similarity_pct")
    assert result.verdict == MeaningVerdict.INSUFFICIENT


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


def test_offline_enforces_hf_hub_offline_at_loader_boundary() -> None:
    """allow_download=False must set HF_HUB_OFFLINE=1 around backend load."""
    import os

    from citeguard.similarity.embeddings import sentence_transformers as st_mod

    seen: dict[str, object] = {}
    real_getenv = os.environ.get

    class _FakeSTBackend:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)
            seen["HF_HUB_OFFLINE"] = real_getenv("HF_HUB_OFFLINE")

        def encode(self, texts: list[str]) -> np.ndarray:
            return np.ones((len(texts), 8), dtype=np.float32)

    backend = SentenceTransformerSemanticBackend(
        model_name="test-model", allow_download=False
    )
    with (
        patch.object(st_mod, "_is_model_cached", return_value=True),
        patch.object(st_mod, "SentenceTransformerBackend", _FakeSTBackend),
    ):
        assert backend.available is True
    assert seen.get("HF_HUB_OFFLINE") == "1"
    assert real_getenv("HF_HUB_OFFLINE") is None


def test_bare_model_name_resolves_org_prefixed_cache(tmp_path) -> None:
    """_is_model_cached must find org-prefixed cache dirs for bare names."""
    from pathlib import Path

    from citeguard.similarity.embeddings import sentence_transformers as st_mod

    snap = (
        tmp_path
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--sentence-transformers--my-model"
        / "snapshots"
        / "abc123"
    )
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"fake")
    with patch.object(Path, "home", return_value=tmp_path):
        assert st_mod._is_model_cached("my-model") is True
        assert st_mod._is_model_cached("other-org/my-model") is False


def test_nli_offline_passes_local_files_only_to_loaders() -> None:
    """allow_download=False must pass local_files_only=True to both the
    tokenizer and the model loader."""
    import sys

    calls: dict[str, dict[str, object]] = {}

    class _FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: object) -> MagicMock:
            calls["tokenizer"] = {"name": name, **kwargs}
            return MagicMock()

    class _FakeModel:
        config = MagicMock(label2id={"entailment": 0, "neutral": 1})

        @classmethod
        def from_pretrained(cls, name: str, **kwargs: object) -> MagicMock:
            calls["model"] = {"name": name, **kwargs}
            instance = MagicMock()
            instance.config.label2id = {"entailment": 0, "neutral": 1}
            return instance

    fake_transformers = MagicMock()
    fake_transformers.AutoTokenizer = _FakeTokenizer
    fake_transformers.AutoModelForSequenceClassification = _FakeModel

    backend = TransformerNLIBackend(
        model_name="test-nli-model", allow_download=False
    )
    with (
        patch.dict(sys.modules, {"transformers": fake_transformers}),
        patch(
            "citeguard.similarity.embeddings.sentence_transformers._is_model_cached",
            return_value=True,
        ),
    ):
        assert backend.available is True
    assert calls["tokenizer"].get("local_files_only") is True
    assert calls["model"].get("local_files_only") is True


def test_nli_online_does_not_force_local_files_only() -> None:
    """allow_download=True must not inject local_files_only."""
    import sys

    calls: dict[str, dict[str, object]] = {}

    class _FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: object) -> MagicMock:
            calls["tokenizer"] = {"name": name, **kwargs}
            return MagicMock()

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: object) -> MagicMock:
            calls["model"] = {"name": name, **kwargs}
            instance = MagicMock()
            instance.config.label2id = {}
            return instance

    fake_transformers = MagicMock()
    fake_transformers.AutoTokenizer = _FakeTokenizer
    fake_transformers.AutoModelForSequenceClassification = _FakeModel

    backend = TransformerNLIBackend(
        model_name="test-nli-model", allow_download=True
    )
    with patch.dict(sys.modules, {"transformers": fake_transformers}):
        assert backend.available is True
    assert "local_files_only" not in calls["tokenizer"]
    assert "local_files_only" not in calls["model"]


def test_offline_unavailable_semantic_fails_closed_in_pipeline() -> None:
    """Unavailable real semantic model → INSUFFICIENT, never PRESERVED."""
    from citeguard.reduction.meaning import LocalModelUnavailableError

    unavailable = SentenceTransformerSemanticBackend(
        model_name="nonexistent-model", allow_download=False
    )
    with pytest.raises(LocalModelUnavailableError):
        unavailable.similarity("a", "b")

    candidate = RewriteCandidate("The drug reduced symptoms.", "test")
    result = validate_meaning(
        "The medication alleviated symptoms.",
        candidate,
        semantic_backend=unavailable,
        entailment_backend=_FakeEntailmentBackend(
            forward=EntailmentDirection(0.99, MeaningVerdict.PRESERVED),
            backward=EntailmentDirection(0.99, MeaningVerdict.PRESERVED),
        ),
        thresholds=MeaningThresholds(semantic_minimum=0.0, entailment_minimum=0.0),
    )
    assert result.verdict == MeaningVerdict.INSUFFICIENT
    assert result.semantic_similarity_raw is None
    assert "unavailable" in " ".join(result.reasons).lower()


def test_offline_unavailable_nli_yields_unknown_never_preserved() -> None:
    """Unavailable NLI model → UNKNOWN direction; pipeline stays INSUFFICIENT."""
    backend = TransformerNLIBackend(
        model_name="nonexistent/nli-model-for-test", allow_download=False
    )
    direction = backend.evaluate("premise text", "hypothesis text")
    assert direction.verdict == MeaningVerdict.UNKNOWN

    candidate = RewriteCandidate("A paraphrase.", "test")
    result = validate_meaning(
        "Original text.",
        candidate,
        semantic_backend=_FakeSemanticBackend(0.99),
        entailment_backend=backend,
        thresholds=MeaningThresholds(semantic_minimum=0.0, entailment_minimum=0.0),
    )
    assert result.verdict == MeaningVerdict.INSUFFICIENT


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
    assert _map_nli_label("LABEL_0") == MeaningVerdict.UNKNOWN
    assert _map_nli_label("LABEL_1") == MeaningVerdict.UNKNOWN
    assert _map_nli_label("LABEL_2") == MeaningVerdict.UNKNOWN
    assert _map_nli_label("unknown") == MeaningVerdict.UNKNOWN


def test_nli_negated_entailment_alias_maps_to_unknown() -> None:
    """'non-entailment' must NEVER map to PRESERVED."""
    assert _map_nli_label("non-entailment") == MeaningVerdict.UNKNOWN
    assert _map_nli_label("non_entailment") == MeaningVerdict.UNKNOWN
    assert _map_nli_label("NOT ENTAILMENT") == MeaningVerdict.UNKNOWN
    assert _map_nli_label("not-entailed") == MeaningVerdict.UNKNOWN


def test_nli_contradict_forms_map_to_contradicted() -> None:
    assert _map_nli_label("contradictory") == MeaningVerdict.CONTRADICTED
    assert _map_nli_label("CONTRADICTS") == MeaningVerdict.CONTRADICTED


def test_nli_score_is_predicted_class_confidence() -> None:
    """EntailmentDirection.score is the model confidence for the SELECTED
    directional NLI class — not a rewrite-safety probability."""
    import sys

    import torch

    class _FakeTokenizer:
        def __call__(self, *args: object, **kwargs: object) -> dict[str, object]:
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    class _FakeModel:
        config = MagicMock(
            label2id={"contradiction": 0, "neutral": 1, "entailment": 2}
        )

        def eval(self) -> None:
            return None

        def __call__(self, **kwargs: object) -> MagicMock:
            out = MagicMock()
            # Logits strongly favor class 0 (contradiction).
            out.logits = torch.tensor([[5.0, 1.0, 0.5]])
            return out

    fake_transformers = MagicMock()
    fake_transformers.AutoTokenizer = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained.return_value = _FakeTokenizer()
    fake_transformers.AutoModelForSequenceClassification = MagicMock()
    fake_transformers.AutoModelForSequenceClassification.from_pretrained.return_value = (
        _FakeModel()
    )

    backend = TransformerNLIBackend(
        model_name="test-nli-model", allow_download=True
    )
    with patch.dict(sys.modules, {"transformers": fake_transformers}):
        direction = backend.evaluate("premise", "hypothesis")
    assert direction.verdict == MeaningVerdict.CONTRADICTED
    # Score equals the contradiction-class softmax confidence.
    expected = float(torch.softmax(torch.tensor([5.0, 1.0, 0.5]), dim=-1)[0].item())
    assert direction.score == pytest.approx(expected)
    assert direction.score > 0.9


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
