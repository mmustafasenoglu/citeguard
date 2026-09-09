#!/usr/bin/env python3
"""Generate and score leakage-resistant Turkish detection benchmark v2."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from citeguard.similarity.fingerprint import exact_overlap, generate_shingles, winnow
from citeguard.similarity.lexical import char_ngram_jaccard, normalize_turkish
from citeguard.similarity.models import Fingerprint

ROOT = Path(__file__).resolve().parents[2]
OLD_FIXTURE = ROOT / "benchmarks/turkish/detection_engineering.json"
FIXTURE = ROOT / "benchmarks/turkish/detection_v2.json"
RESULT = ROOT / "benchmarks/results/turkish/engineering_detection_v2.json"
SEED = 20260909

CATEGORIES = (
    "exact_copy",
    "near_copy",
    "word_order",
    "morphology",
    "punctuation",
    "close_paraphrase",
    "legitimate_semantic_paraphrase",
    "common_academic_phrase",
    "technical_terminology",
    "unrelated_negative",
    "citation_present",
    "quotation_present",
    "semantic_related_not_copy",
    "short_generic_negative",
)
POSITIVE = {
    "exact_copy",
    "near_copy",
    "word_order",
    "morphology",
    "punctuation",
    "close_paraphrase",
    "citation_present",
    "quotation_present",
}

CONTEXTS = (
    (
        "Araştırma Ankara'daki yetişkinlerle yürütüldü.",
        "İnceleme başkentte yaşayan yetişkinleri kapsadı.",
    ),
    (
        "Veriler kırsal kliniklerden toplandı.",
        "Ölçümler kent dışındaki sağlık merkezlerinden alındı.",
    ),
    ("İzlem süresi altı aydı.", "Katılımcılar yarım yıl boyunca takip edildi."),
    (
        "Analiz genç yetişkinlerle sınırlandırıldı.",
        "Değerlendirmeye yalnızca genç erişkinler alındı.",
    ),
)


def _base_rows() -> list[dict[str, str]]:
    old = json.loads(OLD_FIXTURE.read_text(encoding="utf-8"))
    grouped: dict[str, dict[str, str]] = {}
    for case in old:
        source = str(case["source"])
        item = grouped.setdefault(
            source,
            {"source": source, "domain": str(case["domain"])},
        )
        if case["category"] == "legitimate_paraphrase":
            item["paraphrase"] = str(case["candidate"])
        if case["category"] == "unrelated":
            item["unrelated"] = str(case["candidate"])
    return [row for row in grouped.values() if "paraphrase" in row and "unrelated" in row]


def build_cases() -> list[dict[str, object]]:
    bases = _base_rows()
    cases: list[dict[str, object]] = []
    for group_index in range(60):
        base = bases[group_index % len(bases)]
        source_context, paraphrase_context = CONTEXTS[group_index // len(bases)]
        source = f"{base['source']} {source_context}"
        paraphrase = f"{base['paraphrase']} {paraphrase_context}"
        words = source.rstrip(".").split()
        for category in CATEGORIES:
            if category == "exact_copy":
                candidate = source
            elif category == "near_copy":
                candidate = " ".join(words[:-1]) + "."
            elif category == "word_order":
                candidate = " ".join(words[:3] + words[5:] + words[3:5]) + "."
            elif category == "morphology":
                candidate = source.replace("çalışmada", "çalışma kapsamında").replace(
                    "bulundu", "bulunmuştur"
                )
            elif category == "punctuation":
                candidate = source.upper().replace(":", " —").replace(".", "!")
            elif category == "close_paraphrase":
                candidate = paraphrase
            elif category == "legitimate_semantic_paraphrase":
                candidate = paraphrase + " Bu ifade bağımsız biçimde özetlenmiştir."
            elif category == "common_academic_phrase":
                candidate = "Elde edilen sonuçlar istatistiksel olarak değerlendirilmiştir."
            elif category == "technical_terminology":
                candidate = "Güven aralığı, örneklem ve regresyon katsayısı raporlandı."
            elif category == "unrelated_negative":
                candidate = str(base["unrelated"])
            elif category == "citation_present":
                candidate = f"{source} (Yılmaz, 2024)"
            elif category == "quotation_present":
                candidate = f'"{source}" (Yılmaz, 2024)'
            elif category == "semantic_related_not_copy":
                candidate = paraphrase + " Nedensel bir sonuç çıkarılmamıştır."
            else:
                candidate = "Sonuçlar ayrıca değerlendirilmiştir."
            cases.append(
                {
                    "id": f"tr-v2-{group_index:03d}-{category}",
                    "group": f"source-{group_index:03d}",
                    "domain": base["domain"],
                    "category": category,
                    "source": source,
                    "candidate": candidate,
                    "label": "TEXTUAL_OVERLAP_POSITIVE" if category in POSITIVE else "NEGATIVE",
                }
            )
    return cases


def group_split(cases: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    groups = sorted({str(case["group"]) for case in cases})
    random.Random(SEED).shuffle(groups)
    selected = {
        "development": set(groups[:36]),
        "validation": set(groups[36:48]),
        "holdout": set(groups[48:]),
    }
    return {
        name: [case for case in cases if case["group"] in group_ids]
        for name, group_ids in selected.items()
    }


def _score(source: str, candidate: str) -> float:
    def fingerprint(text: str) -> Fingerprint:
        return Fingerprint(points=winnow(generate_shingles(normalize_turkish(text))))

    return max(
        exact_overlap(fingerprint(source), fingerprint(candidate)),
        char_ngram_jaccard(source, candidate),
    )


def metrics(cases: list[dict[str, object]], threshold: float) -> dict[str, object]:
    tp = fp = fn = tn = 0
    hits: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for case in cases:
        expected = case["label"] == "TEXTUAL_OVERLAP_POSITIVE"
        predicted = _score(str(case["source"]), str(case["candidate"])) >= threshold
        tp += int(expected and predicted)
        fp += int(not expected and predicted)
        fn += int(expected and not predicted)
        tn += int(not expected and not predicted)
        category = str(case["category"])
        totals[category] += 1
        hits[category] += int(expected == predicted)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "cases": len(cases),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "per_category_recall_or_accuracy": {
            category: hits[category] / count for category, count in totals.items()
        },
    }


def main() -> None:
    cases = build_cases()
    splits = group_split(cases)
    choices = [
        (threshold, metrics(splits["development"], threshold))
        for threshold in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
    ]
    eligible = [item for item in choices if item[1]["precision"] >= 0.95]
    threshold, development = max(
        eligible,
        key=lambda item: (item[1]["f1"], item[1]["precision"]),
    )
    validation = metrics(splits["validation"], threshold)
    holdout = metrics(splits["holdout"], threshold)
    report = {
        "kind": "turkish_engineering_not_real_world_plagiarism_accuracy",
        "version": 2,
        "seed": SEED,
        "group_aware": True,
        "holdout_touched_before_selection": False,
        "selected_threshold": threshold,
        "development": development,
        "validation": validation,
        "locked_holdout": holdout,
    }
    FIXTURE.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
