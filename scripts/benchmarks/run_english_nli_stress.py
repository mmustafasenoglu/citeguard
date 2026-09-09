#!/usr/bin/env python3
"""Score a deterministic English NLI stress suite with a real NLI model."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from sklearn.metrics import classification_report


def build_cases() -> list[dict[str, str]]:
    patterns = {
        "safe_paraphrase": (
            "Some students improved.",
            "A subset of students improved.",
            "entailment",
        ),
        "neutral": ("Students joined the study.", "Students passed the exam.", "neutral"),
        "contradiction": (
            "The result was significant.",
            "The result was not significant.",
            "contradiction",
        ),
        "negation": ("An effect was observed.", "No effect was observed.", "contradiction"),
        "association_to_causation": ("X was associated with Y.", "X caused Y.", "neutral"),
        "possibility_to_certainty": (
            "The policy may improve outcomes.",
            "The policy definitely improves outcomes.",
            "neutral",
        ),
        "scope_strengthening": ("Some patients recovered.", "All patients recovered.", "neutral"),
        "evidence_strengthening": (
            "Preliminary evidence suggests an effect.",
            "The effect has been conclusively demonstrated.",
            "neutral",
        ),
        "numeric_change": (
            "Mortality declined by 12 percent.",
            "Mortality declined by 15 percent.",
            "contradiction",
        ),
        "unit_change": ("The dose was 10 mg.", "The dose was 10 g.", "contradiction"),
    }
    return [
        {
            "category": category,
            "premise": f"Study {index + 1}: {premise}",
            "hypothesis": f"Study {index + 1}: {hypothesis}",
            "label": label,
        }
        for category, (premise, hypothesis, label) in patterns.items()
        for index in range(20)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = build_cases()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()
    labels = {int(key): value.lower() for key, value in model.config.id2label.items()}
    predictions: list[str] = []
    started = time.perf_counter()
    for offset in range(0, len(rows), 32):
        batch = rows[offset : offset + 32]
        encoded = tokenizer(
            [row["premise"] for row in batch],
            [row["hypothesis"] for row in batch],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            predictions.extend(
                labels[int(index)] for index in model(**encoded).logits.argmax(dim=-1)
            )
    gold = [row["label"] for row in rows]
    report = classification_report(
        gold,
        predictions,
        labels=["entailment", "neutral", "contradiction"],
        output_dict=True,
        zero_division=0,
    )
    total, correct = Counter(), Counter()
    for row, predicted in zip(rows, predictions, strict=True):
        total[row["category"]] += 1
        correct[row["category"]] += int(row["label"] == predicted)
    result = {
        "model": args.model,
        "cases": len(rows),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "contradiction_recall": report["contradiction"]["recall"],
        "category_accuracy": {key: correct[key] / value for key, value in total.items()},
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
