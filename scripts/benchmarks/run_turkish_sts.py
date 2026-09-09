#!/usr/bin/env python3
"""Evaluate CiteGuard's textual score on Turkish STS files.

This intentionally reports correlation, not plagiarism classification.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

from citeguard.similarity.lexical import char_ngram_jaccard


def read_rows(path: Path) -> list[tuple[str, str, float]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            first = row.get("sentence1_tr") or row.get("sentence1")
            second = row.get("sentence2_tr") or row.get("sentence2")
            if first and second and row.get("score"):
                rows.append((first, second, float(row["score"])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    rows = [row for path in args.files for row in read_rows(path)]
    gold = [row[2] for row in rows]
    textual = [char_ngram_jaccard(row[0], row[1]) for row in rows]
    report = {
        "benchmark": args.name,
        "kind": "external_sts_not_plagiarism",
        "cases": len(rows),
        "model": "citeguard_char_3gram_textual_baseline",
        "pearson": float(pearsonr(gold, textual).statistic),
        "spearman": float(spearmanr(gold, textual).statistic),
        "limitation": "textual baseline only; no local semantic model was cached",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
