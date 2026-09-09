#!/usr/bin/env python3
"""Evaluate bidirectional CiteGuard meaning semantics with real local models."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from citeguard.reduction.models import MeaningVerdict


def build_cases(language: str) -> list[dict[str, str]]:
    if language == "en":
        patterns = {
            "safe_paraphrase": (
                "Some patients improved after treatment.",
                "A subset of patients improved following treatment.",
                "safe",
            ),
            "citation_preserving_safe_rewrite": (
                "The intervention reduced mortality. (Smith, 2024)",
                "Mortality declined after the intervention. (Smith, 2024)",
                "safe",
            ),
            "neutral_addition": (
                "The study measured blood pressure.",
                "The study measured blood pressure and improved survival.",
                "unsafe",
            ),
            "contradiction": (
                "The result was statistically significant.",
                "The result was not statistically significant.",
                "contradiction",
            ),
            "negation": ("An effect was observed.", "No effect was observed.", "contradiction"),
            "causal_strengthening": (
                "Exposure was associated with the outcome.",
                "Exposure caused the outcome.",
                "unsafe",
            ),
            "modality_strengthening": (
                "The policy may improve outcomes.",
                "The policy definitely improves outcomes.",
                "unsafe",
            ),
            "scope_strengthening": ("Some students improved.", "All students improved.", "unsafe"),
            "evidence_strengthening": (
                "Preliminary evidence suggests an effect.",
                "The effect has been conclusively demonstrated.",
                "unsafe",
            ),
            "numeric_change": (
                "Mortality declined by 12 percent.",
                "Mortality declined by 15 percent.",
                "contradiction",
            ),
            "unit_change": ("The dose was 10 mg.", "The dose was 10 g.", "contradiction"),
        }
    else:
        patterns = {
            "safe_paraphrase": (
                "Bazı hastalar tedavi sonrasında iyileşti.",
                "Hastaların bir bölümü tedaviyi takiben iyileşme gösterdi.",
                "safe",
            ),
            "citation_preserving_safe_rewrite": (
                "Müdahale ölüm oranını azalttı. (Yılmaz, 2024)",
                "Müdahale sonrasında ölüm oranı düştü. (Yılmaz, 2024)",
                "safe",
            ),
            "neutral_addition": (
                "Çalışma kan basıncını ölçtü.",
                "Çalışma kan basıncını ölçtü ve yaşam süresini artırdı.",
                "unsafe",
            ),
            "contradiction": (
                "Sonuç istatistiksel olarak anlamlıydı.",
                "Sonuç istatistiksel olarak anlamlı değildi.",
                "contradiction",
            ),
            "negation": ("Bir etki gözlendi.", "Hiçbir etki gözlenmedi.", "contradiction"),
            "causal_strengthening": (
                "Maruziyet sonuçla ilişkiliydi.",
                "Maruziyet sonuca neden oldu.",
                "unsafe",
            ),
            "modality_strengthening": (
                "Politika sonuçları iyileştirebilir.",
                "Politika sonuçları kesin olarak iyileştirir.",
                "unsafe",
            ),
            "scope_strengthening": (
                "Bazı öğrenciler iyileşti.",
                "Tüm öğrenciler iyileşti.",
                "unsafe",
            ),
            "evidence_strengthening": (
                "Ön bulgular bir etkiye işaret ediyor.",
                "Etki kesin olarak kanıtlanmıştır.",
                "unsafe",
            ),
            "numeric_change": (
                "Ölüm oranı yüzde 12 azaldı.",
                "Ölüm oranı yüzde 15 azaldı.",
                "contradiction",
            ),
            "unit_change": ("Doz 10 mg idi.", "Doz 10 g idi.", "contradiction"),
        }
    return [
        {
            "id": f"{language}-{category}-{index:02d}",
            "category": category,
            "original": f"Kohort {index + 1}: {original}"
            if language == "tr"
            else f"Cohort {index + 1}: {original}",
            "candidate": f"Kohort {index + 1}: {candidate}"
            if language == "tr"
            else f"Cohort {index + 1}: {candidate}",
            "expected": expected,
        }
        for category, (original, candidate, expected) in patterns.items()
        for index in range(30)
    ]


def classify(model_id: str, rows: list[dict[str, str]]) -> list[MeaningVerdict]:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    embedding = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    originals = [row["original"] for row in rows]
    candidates = [row["candidate"] for row in rows]
    vectors_a = embedding.encode(originals, normalize_embeddings=True)
    vectors_b = embedding.encode(candidates, normalize_embeddings=True)
    semantic = np.sum(np.asarray(vectors_a) * np.asarray(vectors_b), axis=1)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    labels = {int(key): value.lower() for key, value in model.config.id2label.items()}

    def directions(left: list[str], right: list[str]) -> list[MeaningVerdict]:
        output: list[MeaningVerdict] = []
        for start in range(0, len(rows), 32):
            encoded = tokenizer(
                left[start : start + 32],
                right[start : start + 32],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                indices = model(**encoded).logits.argmax(dim=-1)
            for index in indices:
                label = labels[int(index)]
                output.append(
                    MeaningVerdict.CONTRADICTED
                    if "contradict" in label
                    else MeaningVerdict.PRESERVED
                    if "entail" in label
                    else MeaningVerdict.UNKNOWN
                )
        return output

    forward, backward = directions(originals, candidates), directions(candidates, originals)
    return [
        MeaningVerdict.CONTRADICTED
        if MeaningVerdict.CONTRADICTED in {left, right}
        else MeaningVerdict.PRESERVED
        if score >= 0.0 and left == right == MeaningVerdict.PRESERVED
        else MeaningVerdict.UNKNOWN
        for score, left, right in zip(semantic, forward, backward, strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("en", "tr"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_cases(args.language)
    verdicts = classify(args.model, rows)
    total, safe_total, safe_preserved = Counter(), 0, 0
    unsafe_total = unsafe_preserved = contradiction_total = contradiction_hits = 0
    for row, verdict in zip(rows, verdicts, strict=True):
        total[row["category"]] += 1
        if row["expected"] == "safe":
            safe_total += 1
            safe_preserved += int(verdict == MeaningVerdict.PRESERVED)
        else:
            unsafe_total += 1
            unsafe_preserved += int(verdict == MeaningVerdict.PRESERVED)
        if row["expected"] == "contradiction":
            contradiction_total += 1
            contradiction_hits += int(verdict == MeaningVerdict.CONTRADICTED)
    result = {
        "language": args.language,
        "model": args.model,
        "cases": len(rows),
        "safe_preserved_recall": safe_preserved / safe_total,
        "unsafe_preserved_false_positive_rate": unsafe_preserved / unsafe_total,
        "contradiction_recall": contradiction_hits / contradiction_total,
        "verdict_counts": dict(Counter(verdict.value for verdict in verdicts)),
        "category_cases": dict(total),
        "implementation": "batched evaluation of CiteGuard's current bidirectional "
        "verdict contract",
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
