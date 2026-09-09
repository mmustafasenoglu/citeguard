#!/usr/bin/env python3
"""Measure semantic candidate retrieval without altering textual overlap scores."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    cases = json.loads(args.fixture.read_text(encoding="utf-8"))
    source_by_group: dict[str, str] = {}
    queries: list[dict[str, object]] = []
    for case in cases:
        source_by_group.setdefault(str(case["group"]), str(case["source"]))
        if case["category"] == "close_paraphrase":
            queries.append(case)
    groups = sorted(source_by_group)
    model = SentenceTransformer(args.model)
    started = time.perf_counter()
    source_vectors = model.encode(
        [source_by_group[group] for group in groups],
        normalize_embeddings=True,
    )
    query_vectors = model.encode(
        [str(case["candidate"]) for case in queries],
        normalize_embeddings=True,
    )
    similarities = np.asarray(query_vectors) @ np.asarray(source_vectors).T
    ranks: list[int] = []
    for case, row in zip(queries, similarities, strict=True):
        order = np.argsort(-row)
        target = groups.index(str(case["group"]))
        ranks.append(int(np.where(order == target)[0][0]) + 1)
    report = {
        "kind": "semantic_candidate_retrieval_not_textual_overlap",
        "model": args.model,
        "cases": len(ranks),
        "recall_at_1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "recall_at_3": sum(rank <= 3 for rank in ranks) / len(ranks),
        "recall_at_5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "mrr": sum(1 / rank for rank in ranks) / len(ranks),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "textual_overlap_modified": False,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
