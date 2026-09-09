#!/usr/bin/env python3
"""Benchmark real sentence embeddings on Turkish STS and retrieval data.

This script is intentionally outside normal CI. Model downloads occur only when
the operator executes it explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


def read_sts(paths: list[Path]) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                first = row.get("sentence1_tr") or row.get("sentence1")
                second = row.get("sentence2_tr") or row.get("sentence2")
                if first and second and row.get("score"):
                    rows.append((first, second, float(row["score"])))
    return rows


def evaluate(model: object, rows: list[tuple[str, str, float]]) -> dict[str, float | int]:
    started = time.perf_counter()
    left = model.encode([row[0] for row in rows], normalize_embeddings=True)
    right = model.encode([row[1] for row in rows], normalize_embeddings=True)
    scores = np.sum(np.asarray(left) * np.asarray(right), axis=1)
    gold = [row[2] for row in rows]
    return {
        "cases": len(rows),
        "pearson": float(pearsonr(gold, scores).statistic),
        "spearman": float(spearmanr(gold, scores).statistic),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--development", nargs="+", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--human", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    development = read_sts(args.development)
    results: list[dict[str, object]] = []
    loaded: dict[str, object] = {}
    for model_id in args.models:
        model = SentenceTransformer(model_id)
        loaded[model_id] = model
        results.append(
            {
                "model": model_id,
                "dimension": model.get_sentence_embedding_dimension(),
                "development": evaluate(model, development),
            }
        )
    selected = max(results, key=lambda item: item["development"]["spearman"])
    selected_id = str(selected["model"])
    selected["official_test"] = evaluate(loaded[selected_id], read_sts([args.test]))
    selected["human_sanity"] = evaluate(loaded[selected_id], read_sts(args.human))
    report = {
        "kind": "external_sts_semantic_capability_not_plagiarism_accuracy",
        "selection_data": "train+development only",
        "holdout_touched_before_selection": False,
        "selected_model": selected_id,
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
