#!/usr/bin/env python3
"""Generate and score the deterministic Turkish rewrite-safety suite."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from citeguard.models import Verdict
from citeguard.reduction.models import (
    FixAction,
    FixPlan,
    MeaningValidation,
    MeaningVerdict,
    RewriteCandidate,
)
from citeguard.reduction.ranker import rank_candidates
from citeguard.reduction.source_evaluator import evaluate_source_overlap
from citeguard.reduction.validator import validate_candidate

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "benchmarks/turkish/rewrite_safety.json"
RESULT = ROOT / "benchmarks/results/turkish/rewrite_safety.json"

BASES = [
    ("Müdahale ölüm oranını %12 azalttı", "ölüm oranında %12 düşüş müdahale sonrasında gözlendi"),
    (
        "Model doğruluğu 6,4 puan artırabilir",
        "modelin doğrulukta 6,4 puan artış sağlaması mümkündür",
    ),
    ("Maruziyet sonuçla ilişkili bulundu", "sonuç ile maruziyet arasında ilişki gözlendi"),
    ("Bazı öğrenciler programdan yarar gördü", "program bazı öğrenciler için yararlı oldu"),
    ("Ön bulgular başarı artışına işaret etmektedir", "başarı artışı ilk bulgularda görülmektedir"),
    ("Tedavi 10 mg/kg dozunda uygulanabilir", "10 mg/kg tedavi dozunun uygulanması mümkündür"),
    ("Örneklem n=1.248 katılımcıdan oluştu", "çalışmaya n=1.248 katılımcı dahil edildi"),
    ("Tahmin için %95 güven aralığı raporlandı", "%95 güven aralığı tahminle birlikte sunuldu"),
    ("Sürüm 1.2 gecikmeyi azaltabilir", "gecikme sürüm 1.2 ile azalabilir"),
    ("Politika geliri yüzde 12 artırabilir", "gelirde yüzde 12 artış politika ile mümkün olabilir"),
    (
        "Düzenli egzersiz uyku ile bağlantılı bulundu",
        "uyku ile düzenli egzersiz arasında bağlantı görüldü",
    ),
    ("Belirli katılımcılar olumlu yanıt verdi", "olumlu yanıt belirli katılımcılarda gözlendi"),
    (
        "Sınırlı kanıt kirliliğin azalabileceğini göstermektedir",
        "kirlilikte azalma olasılığını sınırlı kanıt desteklemektedir",
    ),
    ("Ölçüm 3,2 mmol/L olarak kaydedildi", "kaydedilen ölçüm değeri 3,2 mmol/L idi"),
    ("Çalışma 2024 yılında tamamlandı", "araştırmanın tamamlandığı yıl 2024'tür"),
]
CATEGORIES = (
    "safe",
    "citation_removal",
    "citation_mutation",
    "number_mutation",
    "unit_mutation",
    "unsupported_claim",
    "contradiction",
    "causal_strengthening",
    "modality_strengthening",
    "scope_strengthening",
    "evidence_strengthening",
    "quotation_needed",
    "missing_citation",
    "source_overlap_not_improved",
)


def build_cases() -> list[dict[str, object]]:
    cases = []
    for index, (claim, safe) in enumerate(BASES):
        original = f"{claim} (Yılmaz, 2024)."
        for category in CATEGORIES:
            candidate = f"{safe} (Yılmaz, 2024)."
            if category == "citation_removal":
                candidate = safe + "."
            elif category == "citation_mutation":
                candidate = f"{safe} (Demir, 2025)."
            elif category == "number_mutation":
                candidate = (
                    candidate.replace("12", "15").replace("6,4", "6,5").replace("1.248", "1.200")
                )
                if candidate == f"{safe} (Yılmaz, 2024).":
                    candidate = candidate.replace("2024", "2025")
            elif category == "unit_mutation":
                candidate = candidate.replace("mg/kg", "g/kg").replace("mmol/L", "mol/L")
                if candidate == f"{safe} (Yılmaz, 2024).":
                    candidate = candidate.replace(" (Yılmaz, 2024).", " 10 g (Yılmaz, 2024).")
            elif category == "unsupported_claim":
                candidate += " Ayrıca maliyetler azaldı."
            elif category == "contradiction":
                candidate = "Bu sonuç gözlenmedi (Yılmaz, 2024)."
            elif category == "causal_strengthening":
                candidate = original.replace("ilişkili bulundu", "neden oldu").replace(
                    "bağlantılı bulundu", "sebep oldu"
                )
            elif category == "modality_strengthening":
                candidate = (
                    original.replace("artırabilir", "kesin olarak artırır")
                    .replace("uygulanabilir", "mutlaka uygulanır")
                    .replace("azaltabilir", "kesin olarak azaltır")
                )
            elif category == "scope_strengthening":
                candidate = original.replace("Bazı ", "Tüm ").replace("Belirli ", "Bütün ")
            elif category == "evidence_strengthening":
                candidate = original.replace("Ön bulgular", "Kesin kanıt").replace(
                    "Sınırlı kanıt", "Kesin kanıt"
                )
            elif category == "source_overlap_not_improved":
                candidate = original
            cases.append(
                {
                    "id": f"safety-{index:02d}-{category}",
                    "category": category,
                    "source": claim + ".",
                    "original": original,
                    "candidate": candidate,
                    "expected_accepted": category == "safe",
                }
            )
    return cases


def main() -> None:
    cases = build_cases()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    safe_tp = unsafe_fp = safe_fn = unsafe_tn = 0
    unexpected_acceptances: list[str] = []
    safe_rejections: list[str] = []
    for case in cases:
        category = str(case["category"])
        candidate = RewriteCandidate(str(case["candidate"]), "engineering")
        plan = FixPlan(
            passage_id=str(case["id"]),
            action=FixAction.PARAPHRASE,
            reason="engineering",
            preserve_citations=("(Yılmaz, 2024)",),
            rewrite_allowed=category not in {"quotation_needed", "missing_citation"},
        )
        evaluate_source_overlap(str(case["original"]), candidate, str(case["source"]))
        verdict = (
            MeaningVerdict.CONTRADICTED if category == "contradiction" else MeaningVerdict.PRESERVED
        )
        meaning = MeaningValidation(0.95, 0.95, 0.95, verdict, verdict, 0.95, verdict)
        unsupported = ["maliyetler azaldı"] if category == "unsupported_claim" else []
        if not plan.rewrite_allowed:
            candidate.rejection_reasons.append("planner disallows rewrite")
        validate_candidate(
            str(case["original"]),
            candidate,
            plan,
            supported_verdict=Verdict.SUPPORTED,
            unsupported_claims=unsupported,
            meaning_validation=meaning,
        )
        candidate.meaning_verdict = verdict
        candidate.meaning_score = 0.95
        candidate.source_support_score = 1.0
        accepted = bool(rank_candidates([candidate]))
        expected = bool(case["expected_accepted"])
        counts[category] += 1
        correct[category] += accepted == expected
        safe_tp += accepted and expected
        unsafe_fp += accepted and not expected
        safe_fn += not accepted and expected
        unsafe_tn += not accepted and not expected
        if accepted and not expected:
            unexpected_acceptances.append(str(case["id"]))
        if not accepted and expected:
            safe_rejections.append(str(case["id"]))
    precision = safe_tp / (safe_tp + unsafe_fp)
    recall = safe_tp / (safe_tp + safe_fn)
    report = {
        "kind": "turkish_engineering_not_external_accuracy",
        "cases": len(cases),
        "safe_acceptance_precision": precision,
        "safe_acceptance_recall": recall,
        "safe_acceptance_f1": 2 * precision * recall / (precision + recall),
        "unsafe_rejection_rate": unsafe_tn / (unsafe_tn + unsafe_fp),
        "per_category_rejection_or_acceptance_accuracy": {
            k: correct[k] / v for k, v in counts.items()
        },
        "unexpected_acceptances": unexpected_acceptances,
        "safe_rejections": safe_rejections,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
