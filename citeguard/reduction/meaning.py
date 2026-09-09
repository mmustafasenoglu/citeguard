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


class LocalModelUnavailableError(Exception):
    """Raised when a real local model backend cannot serve a request.

    This distinguishes three states explicitly:

    - ``REAL_SEMANTIC`` — a local embedding/NLI model produced evidence
    - ``SEQUENCE_FALLBACK`` — ``SequenceSemanticBackend`` lexical screening
    - ``UNAVAILABLE`` — this error; no fake evidence is substituted
    """

    pass


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

    Returns raw cosine similarity in approximately ``[-1.0, 1.0]``.
    Negative values are meaningful (opposing semantics) and are NOT
    clamped to zero.  Only floating-point overshoot beyond ``[-1, 1]``
    is clipped as numerical safety.

    When the real embedding model is unavailable this backend raises
    ``LocalModelUnavailableError``.  It never silently substitutes
    lexical ``SequenceMatcher`` output as semantic evidence; use the
    explicit ``SequenceSemanticBackend`` for screening instead.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.
        Default: ``paraphrase-multilingual-MiniLM-L12-v2``.
    allow_download:
        When *False*, only use locally cached models.  Offline loading
        is enforced with ``HF_HUB_OFFLINE=1`` at the loader boundary,
        so no network request is attempted.
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
        """Return the SentenceTransformerBackend, or None when unavailable."""
        if self._attempted:
            return self._backend
        self._attempted = True
        try:
            from ..similarity.embeddings.sentence_transformers import (
                SentenceTransformerBackend,
                _is_model_cached,
            )

            if not self._allow_download and not _is_model_cached(self._model_name):
                log.debug(
                    "Semantic backend: model '%s' not cached, allow_download=False.",
                    self._model_name,
                )
                self._backend = None
                return self._backend

            if not self._allow_download:
                # Enforce zero-network loading at the loader boundary.
                # HF_HUB_OFFLINE=1 makes huggingface_hub raise instead of
                # downloading; restored afterwards.
                import os

                previous = os.environ.get("HF_HUB_OFFLINE")
                os.environ["HF_HUB_OFFLINE"] = "1"
                try:
                    self._backend = SentenceTransformerBackend(
                        model_name=self._model_name,
                        allow_download=self._allow_download,
                    )
                finally:
                    if previous is None:
                        del os.environ["HF_HUB_OFFLINE"]
                    else:
                        os.environ["HF_HUB_OFFLINE"] = previous
            else:
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
        """Compute raw cosine similarity between original and candidate.

        Returns raw cosine similarity, approximately in ``[-1.0, 1.0]``.
        This is semantic evidence — not a plagiarism probability, not a
        confidence percentage, and never mixed into textual
        ``overall_similarity_pct``.

        Raises
        ------
        LocalModelUnavailableError
            When the real embedding model cannot be loaded.  No lexical
            fallback is substituted.
        """
        backend = self._get_backend()
        if backend is None:
            raise LocalModelUnavailableError(
                f"Semantic embedding model '{self._model_name}' is unavailable. "
                "Use SequenceSemanticBackend explicitly for lexical screening."
            )

        import numpy as np

        embeddings = backend.encode([original, candidate])
        vec_a = embeddings[0]
        vec_b = embeddings[1]
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        cos = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        # Numerical safety only: clip floating-point overshoot, keep negatives.
        return max(-1.0, min(1.0, cos))


# ---------------------------------------------------------------------------
# Real local NLI backend (transformer cross-encoder)
# ---------------------------------------------------------------------------

_DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# Matches negated entailment aliases such as "non-entailment" or
# "not entailment", which must map to UNKNOWN — never PRESERVED.
_NEGATED_ENTAIL_RE = re.compile(r"\b(?:non|not|no|without)\b.*\bentail")


def _map_nli_label(raw_label: str) -> MeaningVerdict:
    """Map a model's output label to a MeaningVerdict.

    Label order is never assumed; the caller reads
    ``model.config.id2label`` dynamically.  Matching is case-insensitive
    and tolerant of ``LABEL_<id>`` prefixes and ``_``/``-`` separators.

    - entailment → PRESERVED
    - contradiction (or any "contradict*" form) → CONTRADICTED
    - neutral → UNKNOWN
    - negated entailment aliases ("non-entailment", ...) → UNKNOWN
    - anything else (including bare ``LABEL_<id>``) → UNKNOWN (conservative)
    """
    normalized = raw_label.strip().lower().replace("_", " ").replace("-", " ")
    normalized = re.sub(r"^label\s+\d+\s*", "", normalized).strip()
    if not normalized or normalized == "unknown":
        return MeaningVerdict.UNKNOWN
    if "contradict" in normalized:
        return MeaningVerdict.CONTRADICTED
    if "neutral" in normalized:
        return MeaningVerdict.UNKNOWN
    if "entail" in normalized:
        if _NEGATED_ENTAIL_RE.search(normalized):
            return MeaningVerdict.UNKNOWN
        return MeaningVerdict.PRESERVED
    return MeaningVerdict.UNKNOWN


class TransformerNLIBackend:
    """Local transformer-based natural language inference backend.

    Uses a cross-encoder NLI model from HuggingFace transformers.
    Lazily loaded on first ``evaluate()`` call.  No network or model
    loading at import time.  CPU-compatible; inference runs without
    gradients and the loaded model is reused across calls.

    Output labels are read from ``model.config.id2label`` dynamically —
    label order is never assumed.  See :func:`_map_nli_label`.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for a sequence-classification NLI model.
    allow_download:
        When *False*, only use locally cached models.  Offline loading
        is enforced with ``local_files_only=True`` at the actual
        tokenizer and model loader calls, so no network request is
        attempted.
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
                        "NLI backend: model '%s' not cached, allow_download=False.",
                        self._model_name,
                    )
                    self._model = None
                    return self._model

            loader_kwargs: dict[str, object] = {}
            if not self._allow_download:
                # Enforce zero-network loading at the loader boundary.
                loader_kwargs["local_files_only"] = True
            tokenizer = AutoTokenizer.from_pretrained(self._model_name, **loader_kwargs)
            model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name, **loader_kwargs
            )
            model.eval()

            config = model.config
            if hasattr(config, "label2id") and config.label2id:
                self._label2id = {k.lower(): v for k, v in config.label2id.items()}
                self._id2label = {v: k for k, v in self._label2id.items()}
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

        ``score`` is the model confidence (softmax probability) for the
        selected directional NLI class — not a rewrite-safety probability.
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

    ``semantic`` is raw cosine similarity (approximately ``[-1.0, 1.0]``),
    kept separate from textual ``overall_similarity_pct``.

    Any contradiction rejects immediately.  Otherwise acceptance requires
    both directions and semantic similarity to clear calibrated thresholds.
    An unavailable real semantic model fails closed: no fake evidence is
    substituted and the verdict can never be PRESERVED.
    """
    thresholds = thresholds or MeaningThresholds()
    try:
        semantic: float | None = semantic_backend.similarity(original, candidate.text)
    except LocalModelUnavailableError as exc:
        semantic = None
        log.debug("Semantic evidence unavailable: %s", exc)
    forward = entailment_backend.evaluate(original, candidate.text)
    backward = entailment_backend.evaluate(candidate.text, original)
    forward_verdict = _coerce_verdict(forward.verdict)
    backward_verdict = _coerce_verdict(backward.verdict)
    reasons: list[str] = []

    if semantic is None:
        reasons.append("real semantic evidence is unavailable (model not loaded)")
    elif semantic < thresholds.semantic_minimum:
        reasons.append("semantic similarity is below the calibrated minimum")
    if forward_verdict == MeaningVerdict.CONTRADICTED:
        reasons.append("original passage contradicts the candidate")
    if backward_verdict == MeaningVerdict.CONTRADICTED:
        reasons.append("candidate contradicts the original passage")
    for direction, value in (("forward", forward), ("backward", backward)):
        if value.score < thresholds.entailment_minimum:
            reasons.append(f"{direction} entailment is below the calibrated minimum")

    definitive = (
        semantic is not None
        and semantic >= thresholds.semantic_minimum
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

    parts = [s for s in (semantic, forward.score, backward.score) if s is not None]
    meaning_score = sum(parts) / len(parts) if parts else None
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
