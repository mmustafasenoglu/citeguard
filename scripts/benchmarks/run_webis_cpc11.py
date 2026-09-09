#!/usr/bin/env python3
"""Validate a locally supplied Webis-CPC-11 directory without redistribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeguard.similarity.lexical import char_ngram_jaccard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.30)
    args = parser.parse_args()
    if not args.dataset_dir.exists():
        raise SystemExit("Webis-CPC-11 directory does not exist.")
    root = args.dataset_dir / "Webis-CPC-11"
    if not root.exists():
        root = args.dataset_dir
    cases = []
    for metadata in sorted(root.glob("*-metadata.txt")):
        identifier = metadata.name.removesuffix("-metadata.txt")
        original = root / f"{identifier}-original.txt"
        paraphrase = root / f"{identifier}-paraphrase.txt"
        if not original.exists() or not paraphrase.exists():
            continue
        label = next(
            (line.partition(":")[2].strip().lower() == "yes"
             for line in metadata.read_text(encoding="utf-8").splitlines()
             if line.startswith("Paraphrase:")),
            None,
        )
        if label is not None:
            cases.append((label, char_ngram_jaccard(
                original.read_text(encoding="utf-8"), paraphrase.read_text(encoding="utf-8")
            )))
    if not cases:
        raise SystemExit("No complete Webis-CPC-11 triplets with labels found.")
    tp = sum(label and score >= args.threshold for label, score in cases)
    fp = sum(not label and score >= args.threshold for label, score in cases)
    fn = sum(label and score < args.threshold for label, score in cases)
    tn = sum(not label and score < args.threshold for label, score in cases)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(json.dumps({
        "benchmark": "Webis-CPC-11", "kind": "paraphrase_not_plagiarism",
        "cases": len(cases), "accepted_paraphrases": tp + fn,
        "rejected_non_paraphrases": tn + fp, "threshold": args.threshold,
        "precision": precision, "recall": recall, "f1": f1,
        "paraphrase_recall": recall, "non_paraphrase_rejection": tn / (tn + fp),
        "roc_auc": None, "pr_auc": None,
    }, indent=2))


if __name__ == "__main__":
    main()
