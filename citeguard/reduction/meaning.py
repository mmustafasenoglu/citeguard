"""Semantic and bidirectional meaning-preservation validation.

Provides three SemanticBackend and two EntailmentBackend implementations:

- ``SequenceSemanticBackend`` — deterministic lexical fallback
- ``SentenceTransformerSemanticBackend`` — local embedding cosine similarity
- ``HeuristicEntailmentBackend`` — offline negation screening (NEVER claims PRESERVED)
- ``TransformerNLIBackend`` — local cross-encoder NLI classification
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from ..models import Verdict
from .models import (
    EntailmentBackend,
    EntailmentDirection,
    MeaningValidation,
    MeaningVerdict,
    RewriteCandidate,
    SemanticBackend,
)

log = logging.getLogger(__name__)

_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|neither|nor|failed|lack|absence|"
    r"didn't|doesn't|isn't|wasn't|weren't)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MeaningThresholds:
    """Calibration-ready thresholds; no production default claims calibration."""

    semantic_minimum: float = 0.0
    entailment_minimum: float = 0.0


class SequenceSemanticBackend:
    """Deterministic fallback that exposes lexical sequence similarity only."""

    def similarity(self, original: str, candidate: str) -> float:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, original.casefold(), candidate.casefold()).ratio()


class HeuristicEntailmentBackend:
    """Offline directional screening backend.

    This backend never claims a definitive entailment result.  It detects
    obvious contradiction risk and returns ``UNKNOWN`` otherwise.
    """

    def evaluate(self, premise: str, hypothesis: str) -> EntailmentDirection:
        premise_tokens = set(re.findall(r"[a-z0-9]+", premise.casefold()))
        hypothesis_tokens = set(re.findall(r"[a-z0-9]+", hypothesis.casefold()))
        if not hypothesis_tokens:
            return EntailmentDirection(0.0, MeaningVerdict.UNKNOWN, "Empty hypothesis.")
        overlap = len(premise_tokens & hypothesis_tokens) / len(hypothesis_tokens)
        if bool(_NEGATION_RE.search(premise)) != bool(_NEGATION_RE.search(hypothesis)):
            return EntailmentDirection(
                min(overlap, 0.5),
                MeaningVerdict.CONTRADICTED,
                "Negation polarity differs.",
            )
        return EntailmentDirection(
            overlap,
            MeaningVerdict.UNKNOWN,
            "Offline heuristic is not a definitive entailment model.",
        )


# ---------------------------------------------------------------------------
# Real local semantic backend (sentence-transformer embeddings)
# ---------------------------------------------------------------------------


class SentenceTransformerSemanticBackend:
    """Local sentence-transformer semantic similarity backend.

    Reuses the existing ``SentenceTransformerBackend`` from
    ``citeguard.similarity.embeddings`` for model loading and caching.

    Lazily initialized on first ``similarity()`` call.  No network or
    model loading at import time.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.
        Default: ``paraphrase-multilingual-MiniLM-L12-v2``.
    allow_download:
        When *False*, only use locally cached models.
    """

    _backend: object | None = None
    _attempted: bool = False

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        allow_download: bool = True,
    ) -> None:
        self._model_name = model_name
        self._allow_download = allow_download

    def _get_backend(self):
        """Return the SentenceTransformerBackend singleton, or None."""
        if self._attempted:
            return self._backend
        self._attempted = True
        try:
            from ..similarity.embeddings.sentence_transformers import (
                SentenceTransformerBackend,
                _is_model_cached,
            )

            if not self._allow_download and not _is_model_cached(
                self._model_name
            ):
                log.debug(
                    "Semantic backend: model '%s' not cached, "
                    "allow_download=False.",
                    self._model_name,
                )
                self._backend = None
                return self._backend

            self._backend = SentenceTransformerBackend(
                model_name=self._model_name,
                allow_download=self._allow_download,
            )
        except Exception as exc:
            log.debug("Semantic backend unavailable: %s", exc)
            self._backend = None
        return self._backend

    @property
    def available(self) -> bool:
        """Return True when the embedding backend is loaded and ready."""
        return self._get_backend() is not None

    def similarity(self, original: str, candidate: str) -> float:
        """Compute cosine similarity between original and candidate.

        Returns a score in [0, 1] where 1 means identical semantic meaning.
        Falls back to SequenceSemanticBackend when the embedding model
        is unavailable.
        """
        backend = self._get_backend()
        if backend is None:
            from difflib import SequenceMatcher

            return SequenceMatcher(None, original.casefold(), candidate.casefold()).ratio()

        import numpy as np

        embeddings = backend.encode([original, candidate])
        vec_a = embeddings[0]
        vec_b = embeddings[1]
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        cos = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        return max(0.0, min(1.0, cos))


# ---------------------------------------------------------------------------
# Real local NLI backend (transformer cross-encoder)
# ---------------------------------------------------------------------------

_DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

_LABEL_MAP: dict[str, MeaningVerdict] = {
    "entailment": MeaningVerdict.PRESERVED,
    "contradiction": MeaningVerdict.CONTRADICTED,
    "neutral": MeaningVerdict.UNKNOWN,
}


def _map_nli_label(raw_label: str) -> MeaningVerdict:
    """Map a model's output label to a MeaningVerdict.

    Case-insensitive, prefix-stripped matching.  Unknown labels map to
    UNKNOWN (conservative).
    """
    normalized = raw_label.strip().lower()
    for key, verdict in _LABEL_MAP.items():
        if key in normalized:
            return verdict
    return MeaningVerdict.UNKNOWN


class TransformerNLIBackend:
    """Local transformer-based natural language inference backend.

    Uses a cross-encoder NLI model from HuggingFace transformers.
    Lazily loaded on first ``evaluate()`` call.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for a sequence-classification NLI model.
    allow_download:
        When *False*, only use locally cached models.
    """

    _model = None
    _attempted: bool = False
    _label2id: dict[str, int] | None = None
    _id2label: dict[int, str] | None = None

    def __init__(
        self,
        model_name: str = _DEFAULT_NLI_MODEL,
        allow_download: bool = True,
    ) -> None:
        self._model_name = model_name
        self._allow_download = allow_download

    def _load_model(self):
        """Lazily load the NLI model. Returns None on failure."""
        if self._attempted:
            return self._model
        self._attempted = True
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if not self._allow_download:
                from ..similarity.embeddings.sentence_transformers import (
                    _is_model_cached,
                )

                if not _is_model_cached(self._model_name):
                    log.debug(
                        "NLI backend: model '%s' not cached, "
                        "allow_download=False.",
                        self._model_name,
                    )
                    self._model = None
                    return self._model

            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name
            )
            model.eval()

            config = model.config
            if hasattr(config, "label2id") and config.label2id:
                self._label2id = {
                    k.lower(): v for k, v in config.label2id.items()
                }
                self._id2label = {
                    v: k for k, v in self._label2id.items()
                }
            else:
                self._label2id = None
                self._id2label = None

            self._model = (tokenizer, model)
            log.info("Loaded NLI model '%s'", self._model_name)
        except Exception as exc:
            log.debug("NLI backend unavailable: %s", exc)
            self._model = None
        return self._model

    @property
    def available(self) -> bool:
        """Return True when the NLI model is loaded and ready."""
        return self._load_model() is not None

    def evaluate(self, premise: str, hypothesis: str) -> EntailmentDirection:
        """Evaluate whether premise entails hypothesis.

        Returns an EntailmentDirection with a MeaningVerdict of
        PRESERVED (entailment), CONTRADICTED, or UNKNOWN (neutral/unknown).
        Never returns PRESERVED on failure — UNKNOWN is the safe default.
        """
        loaded = self._load_model()
        if loaded is None:
            return EntailmentDirection(
                0.0,
                MeaningVerdict.UNKNOWN,
                "NLI model not available.",
            )

        import torch

        tokenizer, model = loaded
        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        if self._id2label is not None:
            top_idx = int(torch.argmax(probs).item())
            raw_label = self._id2label.get(top_idx, "unknown")
        else:
            raw_label = "unknown"

        verdict = _map_nli_label(raw_label)
        score = float(probs.max().item())
        reasoning = f"NLI model output: {raw_label} (confidence: {score:.3f})"

        return EntailmentDirection(score, verdict, reasoning)


def _coerce_verdict(value: MeaningVerdict | Verdict) -> MeaningVerdict:
    if isinstance(value, MeaningVerdict):
        return value
    if value == Verdict.CONTRADICTED:
        return MeaningVerdict.CONTRADICTED
    if value in {Verdict.SUPPORTED, Verdict.PARTIALLY_SUPPORTED}:
        return MeaningVerdict.PRESERVED
    return MeaningVerdict.UNKNOWN


def validate_meaning(
    original: str,
    candidate: RewriteCandidate,
    *,
    semantic_backend: SemanticBackend,
    entailment_backend: EntailmentBackend,
    thresholds: MeaningThresholds | None = None,
    score_to_verdict: Callable[[float], MeaningVerdict] | None = None,
) -> MeaningValidation:
    """Validate meaning using semantic similarity and both entailment directions.

    Any contradiction rejects immediately.  Otherwise acceptance requires both
    directions and semantic similarity to clear calibrated thresholds.
    """
    thresholds = thresholds or MeaningThresholds()
    semantic = semantic_backend.similarity(original, candidate.text)
    forward = entailment_backend.evaluate(original, candidate.text)
    backward = entailment_backend.evaluate(candidate.text, original)
    forward_verdict = _coerce_verdict(forward.verdict)
    backward_verdict = _coerce_verdict(backward.verdict)
    reasons: list[str] = []

    if semantic < thresholds.semantic_minimum:
        reasons.append("semantic similarity is below the calibrated minimum")
    if forward_verdict == MeaningVerdict.CONTRADICTED:
        reasons.append("original passage contradicts the candidate")
    if backward_verdict == MeaningVerdict.CONTRADICTED:
        reasons.append("candidate contradicts the original passage")
    for direction, value in (("forward", forward), ("backward", backward)):
        if value.score < thresholds.entailment_minimum:
            reasons.append(f"{direction} entailment is below the calibrated minimum")

    definitive = (
        semantic >= thresholds.semantic_minimum
        and forward_verdict == MeaningVerdict.PRESERVED
        and backward_verdict == MeaningVerdict.PRESERVED
        and forward.score >= thresholds.entailment_minimum
        and backward.score >= thresholds.entailment_minimum
    )
    if MeaningVerdict.CONTRADICTED in {forward_verdict, backward_verdict}:
        verdict = MeaningVerdict.CONTRADICTED
    elif definitive:
        verdict = MeaningVerdict.PRESERVED
    else:
        verdict = MeaningVerdict.INSUFFICIENT

    meaning_score = (semantic + forward.score + backward.score) / 3.0
    candidate.semantic_similarity_to_original = semantic
    candidate.forward_entailment_score = forward.score
    candidate.backward_entailment_score = backward.score
    candidate.meaning_score = meaning_score
    candidate.meaning_verdict = verdict
    if verdict != MeaningVerdict.PRESERVED:
        candidate.rejection_reasons.extend(reasons or ["meaning preservation was not established"])
    return MeaningValidation(
        semantic_similarity_raw=semantic,
        forward_entailment_score=forward.score,
        backward_entailment_score=backward.score,
        forward_verdict=forward_verdict,
        backward_verdict=backward_verdict,
        meaning_preservation_score=meaning_score,
        verdict=verdict,
        reasons=tuple(reasons),
    )
