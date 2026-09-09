#!/usr/bin/env python3
# ruff: noqa: E501
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
RESULT_V2 = ROOT / "benchmarks/results/turkish/rewrite_safety_v2.json"
GATE_RESULT = ROOT / "benchmarks/results/turkish/rewrite_safety_gate_coverage.json"

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
    ("Beslenme skoru 8 puan artabilir", "beslenme skorunda 8 puan artış mümkün olabilir"),
    ("Uyku ile stres arasında ilişki bulundu", "stres ve uyku arasında bağlantı gözlendi"),
    ("Bazı hastalar tedaviye yanıt verdi", "belirli hastalarda tedavi yanıtı gözlendi"),
    ("Ön sonuçlar güvenlik iyileşmesine işaret ediyor", "ilk veriler güvenlikte iyileşme olasılığını gösteriyor"),
    ("Ölçüm 0,4 µg/mL olarak kaydedildi", "kaydedilen konsantrasyon 0,4 µg/mL idi"),
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


def validate_case(case: dict[str, object]) -> list[str]:
    """Return category-purity violations for one benchmark case.

    This validator is deliberately small and lexical: it validates fixture intent,
    while production meaning/NLI gates remain responsible for semantic decisions.
    """
    category = str(case["category"])
    original = str(case["original"])
    candidate = str(case["candidate"])
    errors: list[str] = []
    if category == "safe" and original.strip() == candidate.strip():
        errors.append("safe case is a no-op")
    if category in {"causal_strengthening", "modality_strengthening", "scope_strengthening", "evidence_strengthening", "contradiction"} and original.strip() == candidate.strip():
        errors.append(f"{category} case is a no-op")
    if category == "citation_removal" and "(Yılmaz, 2024)" not in original:
        errors.append("required citation absent from original")
    if category == "citation_removal" and "(Yılmaz, 2024)" in candidate:
        errors.append("citation was not removed")
    if category == "citation_mutation" and "(Demir, 2025)" not in candidate:
        errors.append("citation was not mutated")
    if category == "number_mutation" and not any(token in original for token in ("12", "6,4", "1.248", "2024")):
        errors.append("no protected number in original")
    if category == "unit_mutation" and not any(token in original for token in ("mg/kg", "mmol/L", "°C", "%")):
        errors.append("no protected unit/value in original")
    if category == "unsupported_claim" and "maliyetler azaldı" not in candidate:
        errors.append("unsupported proposition absent")
    if category == "contradiction" and not any(token in candidate.casefold() for token in ("gözlenmedi", "azalmadı", "değildi")):
        errors.append("contradiction marker absent")
    transitions = {
        "causal_strengthening": (("ilişkili", "bağlantılı"), ("neden oldu", "sebep oldu")),
        "modality_strengthening": (("abilir", "olabilir", "mümkündür"), ("kesin", "mutlaka")),
        "scope_strengthening": (("bazı", "belirli"), ("tüm", "bütün")),
        "evidence_strengthening": (("ön bulgular", "sınırlı kanıt"), ("kesin kanıt", "kanıtlanmıştır")),
    }
    if category in transitions:
        weak, strong = transitions[category]
        if not any(x in original.casefold() for x in weak):
            errors.append("weak source marker absent")
        if not any(x in candidate.casefold() for x in strong):
            errors.append("strengthening marker absent")
    if category == "quotation_needed" and "\"" not in original + candidate and "alıntı" not in original.casefold() + candidate.casefold():
        errors.append("quotation signal absent")
    if category == "missing_citation" and "(Yılmaz, 2024)" not in original:
        errors.append("citation absent from missing-citation source")
    if category == "source_overlap_not_improved" and original.strip() != candidate.strip():
        errors.append("candidate unexpectedly improves overlap")
    return errors


def group_aware_split(cases: list[dict[str, object]], seed: int = 20260909) -> dict[str, list[dict[str, object]]]:
    """Deterministically split by base source group, preventing leakage."""
    import random

    groups: dict[str, list[dict[str, object]]] = {}
    for case in cases:
        groups.setdefault(str(case.get("group", case.get("source", ""))), []).append(case)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    cuts = (round(n * 0.6), round(n * 0.8))
    return {"development": [c for k in keys[:cuts[0]] for c in groups[k]], "validation": [c for k in keys[cuts[0]:cuts[1]] for c in groups[k]], "holdout": [c for k in keys[cuts[1]:] for c in groups[k]]}


def build_cases() -> list[dict[str, object]]:
    cases = []
    for index, (claim, safe) in enumerate(BASES):
        original = f"{claim} (Yılmaz, 2024)."
        for category in CATEGORIES:
            category_claim, category_safe = claim, safe
            if category == "causal_strengthening" and not any(x in claim.casefold() for x in ("ilişkili", "bağlantılı")):
                category_claim, category_safe = "Maruziyet sonuçla ilişkili bulundu", "Maruziyet sonuca neden oldu"
            elif category == "modality_strengthening" and not any(x in claim.casefold() for x in ("artırabilir", "azaltabilir", "uygulanabilir", "olabilir")):
                category_claim, category_safe = "Politika başarıyı artırabilir", "Politika başarıyı kesin olarak artırır"
            elif category == "scope_strengthening" and not any(x in claim.casefold() for x in ("bazı", "belirli")):
                category_claim, category_safe = "Bazı katılımcılar iyileşme gösterdi", "Tüm katılımcılar iyileşme gösterdi"
            elif category == "evidence_strengthening" and not any(x in claim.casefold() for x in ("ön bulgular", "sınırlı kanıt")):
                category_claim, category_safe = "Ön bulgular başarı artışına işaret etmektedir", "Kesin kanıt başarı artışını kanıtlamaktadır"
            elif category == "unit_mutation" and not any(x in claim for x in ("mg/kg", "mmol/L", "°C", "%")):
                category_claim, category_safe = "Tedavi 10 mg/kg dozunda uygulandı", "Tedavi 10 g/kg dozunda uygulandı"
            original = f"{category_claim} (Yılmaz, 2024)."
            candidate = f"{category_safe} (Yılmaz, 2024)."
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
            elif category == "quotation_needed":
                original = f'"{category_claim}" (Yılmaz, 2024).'
                candidate = f'"{category_safe}" (Yılmaz, 2024).'
            elif category == "source_overlap_not_improved":
                candidate = original
            cases.append(
                {
                    "id": f"safety-{index:02d}-{category}",
                    "category": category,
                    "source": claim + ".",
                    "group": f"{index:02d}",
                    "original": original,
                    "candidate": candidate,
                    "expected_accepted": category == "safe",
                }
            )
    return cases


def main() -> None:
    cases = build_cases()
    invalid = {str(case["id"]): validate_case(case) for case in cases}
    invalid = {key: value for key, value in invalid.items() if value}
    if invalid:
        raise ValueError(f"invalid benchmark cases: {invalid}")
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    safe_tp = unsafe_fp = safe_fn = unsafe_tn = 0
    unexpected_acceptances: list[str] = []
    safe_rejections: list[str] = []
    gate_counts: Counter[str] = Counter()
    category_stats: dict[str, dict[str, int]] = {category: {"cases": 0, "accepted": 0, "rejected": 0, "correct": 0} for category in CATEGORIES}
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
        category_stats[category]["cases"] += 1
        category_stats[category]["accepted"] += int(accepted)
        category_stats[category]["rejected"] += int(not accepted)
        category_stats[category]["correct"] += int(accepted == expected)
        for reason in candidate.rejection_reasons:
            text = reason.casefold()
            if "citation" in text:
                gate_counts["citation_gate"] += 1
            elif "numeric" in text:
                gate_counts["numeric_gate"] += 1
            elif "strengthens" in text or "contradict" in text:
                gate_counts["claim_strengthening_gate"] += 1
            elif "unsupported" in text:
                gate_counts["unsupported_claim_gate"] += 1
            elif "meaning" in text:
                gate_counts["meaning_gate"] += 1
            elif "planner" in text:
                gate_counts["planner_gate"] += 1
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
    v2 = dict(report)
    v2["version"] = 2
    v2["per_category"] = category_stats
    v2["split"] = {name: len(items) for name, items in group_aware_split(cases).items()}
    RESULT_V2.write_text(json.dumps(v2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    GATE_RESULT.write_text(json.dumps({"gate_counts": dict(gate_counts), "masked_success_warnings": []}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
