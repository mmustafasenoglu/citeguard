"""Dataset-independent utilities for external passage-alignment benchmarks.

The adapters deliberately do not download or redistribute PAN/Webis data.
They parse locally supplied annotations and score document/source span matches.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class GroundTruthPassage:
    suspicious_document: str
    suspicious_offset: int
    suspicious_length: int
    source_document: str
    source_offset: int
    source_length: int
    plagiarism_type: str | None = None
    obfuscation_type: str | None = None


@dataclass(frozen=True, slots=True)
class PredictedPassage:
    suspicious_document: str
    suspicious_offset: int
    suspicious_length: int
    source_document: str | None
    source_offset: int | None
    source_length: int | None
    score: float


def parse_pan_xml(path: str) -> list[GroundTruthPassage]:
    """Parse PAN-style ``feature`` annotations without assuming attribute order."""
    root = ET.parse(path).getroot()
    suspicious = root.attrib.get("reference", "")
    passages: list[GroundTruthPassage] = []
    for feature in root.iter("feature"):
        attrs = feature.attrib
        if "source_reference" not in attrs or "this_offset" not in attrs:
            continue
        passages.append(
            GroundTruthPassage(
                suspicious_document=suspicious,
                suspicious_offset=int(attrs["this_offset"]),
                suspicious_length=int(attrs["this_length"]),
                source_document=attrs["source_reference"],
                source_offset=int(attrs.get("source_offset", 0)),
                source_length=int(attrs.get("source_length", 0)),
                plagiarism_type=attrs.get("name"),
                obfuscation_type=attrs.get("obfuscation"),
            )
        )
    return passages


def overlap(start_a: int, length_a: int, start_b: int, length_b: int) -> float:
    """Return intersection divided by ground-truth span length."""
    if length_b <= 0:
        return 0.0
    shared = max(0, min(start_a + length_a, start_b + length_b) - max(start_a, start_b))
    return shared / length_b


def score_passages(
    truth: list[GroundTruthPassage],
    predictions: list[PredictedPassage],
    *,
    minimum_overlap: float = 0.5,
) -> dict[str, object]:
    """One-to-one span scoring requiring suspicious and source overlap.

    Predictions cannot inflate recall by matching multiple ground-truth cases.
    """
    used: set[int] = set()
    matches: list[tuple[int, int]] = []
    for prediction_index, prediction in enumerate(predictions):
        best: tuple[float, int] | None = None
        for truth_index, item in enumerate(truth):
            if truth_index in used or prediction.suspicious_document != item.suspicious_document:
                continue
            if prediction.source_document != item.source_document:
                continue
            suspicious = overlap(
                prediction.suspicious_offset, prediction.suspicious_length,
                item.suspicious_offset, item.suspicious_length,
            )
            source = overlap(
                prediction.source_offset or 0, prediction.source_length or 0,
                item.source_offset, item.source_length,
            )
            quality = min(suspicious, source)
            if quality >= minimum_overlap and (best is None or quality > best[0]):
                best = (quality, truth_index)
        if best is not None:
            used.add(best[1])
            matches.append((prediction_index, best[1]))
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": precision, "recall": recall, "f1": f1, "matches": matches}


def document_split(
    documents: list[str], *, seed: int = 20260909, holdout_fraction: float = 0.2
) -> dict[str, list[str]]:
    """Deterministic document-level split; paired passages never drive the split."""
    ordered = sorted(set(documents))
    random.Random(seed).shuffle(ordered)
    boundary = round(len(ordered) * (1 - holdout_fraction))
    return {"development": sorted(ordered[:boundary]), "holdout": sorted(ordered[boundary:])}


def bootstrap_document_ci(values: dict[str, dict[str, object]], *, seed: int = 20260909,
                          samples: int = 1000) -> dict[str, tuple[float, float]]:
    """Deterministic document bootstrap CI for already-scored document metrics."""
    if not values:
        return {name: (0.0, 0.0) for name in ("precision", "recall", "f1")}
    rng = random.Random(seed)
    rows = list(values.values())
    output: dict[str, list[float]] = {name: [] for name in ("precision", "recall", "f1")}
    for _ in range(samples):
        chosen = [rng.choice(rows) for _ in rows]
        for name in output:
            output[name].append(sum(float(row[name]) for row in chosen) / len(chosen))
    return {name: (series[int(samples * .025)], series[int(samples * .975)])
            for name, series in ((key, sorted(value)) for key, value in output.items())}


def serialise_passages(items: list[GroundTruthPassage]) -> list[dict[str, object]]:
    return [asdict(item) for item in items]
