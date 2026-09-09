#!/usr/bin/env python3
"""Evaluate CiteGuard's actual transformer NLI model on Turkish data."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

CATEGORIES = (
    "entailment",
    "neutral",
    "contradiction",
    "negation",
    "association_to_causation",
    "possibility_to_certainty",
    "scope_strengthening",
    "evidence_strengthening",
    "numeric_change",
    "unit_change",
)


def stress_cases() -> list[dict[str, str]]:
    patterns = {
        "entailment": (
            "Bazı öğrenciler iyileşti.",
            "Öğrencilerin bir bölümü iyileşti.",
            "entailment",
        ),
        "neutral": ("Öğrenciler çalışmaya katıldı.", "Öğrenciler sınavı geçti.", "neutral"),
        "contradiction": ("Sonuç anlamlıydı.", "Sonuç anlamlı değildi.", "contradiction"),
        "negation": ("Etki gözlendi.", "Etki gözlenmedi.", "contradiction"),
        "association_to_causation": (
            "Maruziyet sonuçla ilişkiliydi.",
            "Maruziyet sonuca neden oldu.",
            "neutral",
        ),
        "possibility_to_certainty": (
            "Politika başarıyı artırabilir.",
            "Politika başarıyı kesin artırır.",
            "neutral",
        ),
        "scope_strengthening": ("Bazı hastalar iyileşti.", "Tüm hastalar iyileşti.", "neutral"),
        "evidence_strengthening": (
            "Ön bulgular etkiye işaret ediyor.",
            "Etki kesin olarak kanıtlanmıştır.",
            "neutral",
        ),
        "numeric_change": (
            "Ölüm oranı yüzde 12 azaldı.",
            "Ölüm oranı yüzde 15 azaldı.",
            "contradiction",
        ),
        "unit_change": ("Doz 10 mg/kg uygulandı.", "Doz 10 g/kg uygulandı.", "contradiction"),
    }
    return [
        {
            "id": f"stress-{category}-{index:02d}",
            "category": category,
            "premise": f"Çalışma {index + 1}: {patterns[category][0]}",
            "hypothesis": f"Çalışma {index + 1}: {patterns[category][1]}",
            "label": patterns[category][2],
        }
        for category in CATEGORIES
        for index in range(20)
    ]


def predict(model_id: str, rows: list[dict[str, str]]) -> tuple[list[str], float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    id2label = {int(key): value.lower() for key, value in model.config.id2label.items()}
    predictions: list[str] = []
    started = time.perf_counter()
    for offset in range(0, len(rows), 32):
        batch = rows[offset : offset + 32]
        encoded = tokenizer(
            [row["premise"] for row in batch],
            [row["hypothesis"] for row in batch],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**encoded).logits
        predictions.extend(id2label[int(index)] for index in np.argmax(logits.numpy(), axis=1))
    return predictions, time.perf_counter() - started


def score(rows: list[dict[str, str]], predictions: list[str]) -> dict[str, object]:
    labels = ["entailment", "neutral", "contradiction"]
    gold = [row["label"] for row in rows]
    report = classification_report(
        gold,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "cases": len(rows),
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {label: report[label] for label in labels},
    }


def read_snli_split(archive: Path, split: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.endswith(f"_{split}.jsonl"))
        rows = []
        for line in bundle.read(member).decode("utf-8").splitlines():
            item = json.loads(line)
            if item.get("gold_label") in {"entailment", "neutral", "contradiction"}:
                rows.append(
                    {
                        "premise": item["sentence1"],
                        "hypothesis": item["sentence2"],
                        "label": item["gold_label"],
                    }
                )
        return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--snli-archive", type=Path)
    parser.add_argument("--snli-split", choices=("dev", "test"), default="test")
    parser.add_argument("--dataset-name", default="NLI-TR SNLI-TR")
    parser.add_argument("--stress-output", type=Path, required=True)
    parser.add_argument("--nli-output", type=Path, required=True)
    args = parser.parse_args()

    stress = stress_cases()
    predictions, runtime = predict(args.model, stress)
    stress_report = score(stress, predictions)
    category_correct: Counter[str] = Counter()
    category_total: Counter[str] = Counter()
    for row, predicted in zip(stress, predictions, strict=True):
        category_total[row["category"]] += 1
        category_correct[row["category"]] += int(predicted == row["label"])
    stress_report.update(
        {
            "model": args.model,
            "runtime_seconds": round(runtime, 3),
            "category_accuracy": {
                key: category_correct[key] / value for key, value in category_total.items()
            },
        }
    )
    args.stress_output.write_text(
        json.dumps(stress_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.snli_archive and args.snli_archive.exists():
        rows = read_snli_split(args.snli_archive, args.snli_split)
        predictions, runtime = predict(args.model, rows)
        nli_report = score(rows, predictions)
        nli_report.update(
            {
                "status": "AVAILABLE",
                "dataset": f"{args.dataset_name} official {args.snli_split}",
                "model": args.model,
                "archive_sha256": hashlib.sha256(args.snli_archive.read_bytes()).hexdigest(),
                "runtime_seconds": round(runtime, 3),
            }
        )
    else:
        nli_report = {"status": "UNAVAILABLE", "model": args.model}
    args.nli_output.write_text(
        json.dumps(nli_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"stress": stress_report, "nli_tr": nli_report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
