#!/usr/bin/env python3
"""Run the deterministic bilingual plagiarism-review engineering benchmark."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from citeguard.plagiarism.models import PlagiarismConfig
from citeguard.plagiarism.pipeline import scan_document


def _family(match_type: str) -> str:
    if match_type == "exact":
        return "exact"
    if match_type in {"near_duplicate", "lexical_overlap"}:
        return "lexical"
    if match_type == "semantic_overlap":
        return "semantic"
    return "none"


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def run(dataset: Path) -> dict[str, object]:
    cases = json.loads(dataset.read_text(encoding="utf-8"))["cases"]
    families = {"exact": [0, 0, 0], "lexical": [0, 0, 0]}
    attribution = [0, 0, 0]
    localized = 0
    rows = []
    with tempfile.TemporaryDirectory(prefix="citeguard-plagiarism-benchmark-") as directory:
        root = Path(directory)
        for index, case in enumerate(cases):
            document = root / f"document-{index}.txt"
            source = root / f"source-{index}.txt"
            document.write_text(case["document"], encoding="utf-8")
            source.write_text(case["source"], encoding="utf-8")
            result = scan_document(
                document,
                sources=[source],
                config=PlagiarismConfig(
                    min_match_words=4,
                    lexical_threshold=0.35,
                    no_cache=True,
                ),
            )
            match = result.matches[0] if result.matches else None
            predicted = _family(match.match_type) if match else "none"
            expected = case["expected_family"]
            for family, counts in families.items():
                expected_positive = expected == family
                predicted_positive = predicted == family
                if expected_positive and predicted_positive:
                    counts[0] += 1
                elif predicted_positive:
                    counts[1] += 1
                elif expected_positive:
                    counts[2] += 1
            expected_attr = case["expected_attribution"]
            predicted_attr = match.attribution_status.value if match else "none"
            if predicted_attr == expected_attr:
                attribution[0] += 1
            else:
                attribution[1] += 1
                attribution[2] += 1
            if match and match.document_text and match.document_text in case["document"]:
                localized += 1
            rows.append(
                {
                    "id": case["id"],
                    "expected_family": expected,
                    "predicted_family": predicted,
                    "expected_attribution": expected_attr,
                    "predicted_attribution": predicted_attr,
                }
            )
    positive = sum(case["expected_family"] != "none" for case in cases)
    return {
        "schema_version": "1",
        "dataset": str(dataset),
        "cases": len(cases),
        "exact": _prf(*families["exact"]),
        "lexical": _prf(*families["lexical"]),
        "semantic_retrieval": {
            "status": "not_run",
            "reason": "optional model evaluated by the existing multilingual retrieval benchmark",
        },
        "attribution": _prf(*attribution),
        "passage_localization_recall": round(localized / positive, 4) if positive else 1.0,
        "rows": rows,
        "limitations": (
            "Synthetic engineering evidence; not Turnitin equivalence or real-world accuracy."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("benchmarks/plagiarism/engineering.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run(args.dataset)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
