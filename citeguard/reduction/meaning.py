"""Semantic and bidirectional meaning-preservation validation."""

from __future__ import annotations

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
