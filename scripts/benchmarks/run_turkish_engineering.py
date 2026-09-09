#!/usr/bin/env python3
"""Generate and evaluate deterministic Turkish engineering benchmarks."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from citeguard.similarity.fingerprint import exact_overlap, generate_shingles, winnow
from citeguard.similarity.lexical import char_ngram_jaccard, normalize_turkish
from citeguard.similarity.models import Fingerprint

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "benchmarks/turkish/detection_engineering.json"
RESULTS = ROOT / "benchmarks/results/turkish"
SEED = 20260909

SEEDS = [
    (
        "medicine",
        "Müdahale randomize çalışmada ölüm oranını %12 azalttı.",
        "Tedavi sonrasında ölüm oranında yüzde on iki düşüş gözlendi.",
        "Hastane yeni bir görüntüleme cihazı satın aldı.",
    ),
    (
        "medicine",
        "Düzenli egzersiz uyku kalitesiyle ilişkili bulundu.",
        "Sık fiziksel aktivite yapanlar daha iyi uyku bildirdi.",
        "Aşının saklama sıcaklığı yeniden düzenlendi.",
    ),
    (
        "engineering",
        "Kompozit malzeme kiriş dayanımını önemli ölçüde artırdı.",
        "Yeni kompozit kullanıldığında kirişler daha yüksek yük taşıdı.",
        "Rüzgâr tüneli laboratuvarın kuzeyindedir.",
    ),
    (
        "engineering",
        "Sensör 10 mg/kg düzeyindeki değişimi güvenilir biçimde ölçtü.",
        "Algılayıcı kilogram başına 10 mg değişime duyarlıydı.",
        "Köprü projesinin bütçesi onaylandı.",
    ),
    (
        "computer_science",
        "Model değerlendirme kümesinde doğruluğu 6,4 puan artırdı.",
        "Sistem test verilerinde 6,4 puanlık doğruluk kazancı sağladı.",
        "Sunucu bakım penceresi gece başladı.",
    ),
    (
        "computer_science",
        "Sürüm 1.2 sorgu gecikmesini azaltabilir.",
        "Yazılımın 1.2 sürümü daha kısa sorgu süreleri sağlayabilir.",
        "Veri merkezi farklı bir şehirde kuruludur.",
    ),
    (
        "economics",
        "Faiz oranı yatırımlarla negatif ilişkili bulundu.",
        "Faiz yükseldiğinde yatırımların azalma eğiliminde olduğu görüldü.",
        "İhracat raporu cuma günü yayımlandı.",
    ),
    (
        "economics",
        "Politika küçük işletmelerin gelirini artırabilir.",
        "Uygulamanın küçük firmaların kazancına katkı sunması mümkündür.",
        "Merkez bankası binası restore edildi.",
    ),
    (
        "education",
        "Geri bildirim öğrencilerin başarılarıyla bağlantılı bulundu.",
        "Düzenli dönüt alan öğrenciler daha yüksek başarı eğilimi gösterdi.",
        "Okul kütüphanesi yazın kapalıdır.",
    ),
    (
        "education",
        "Ön bulgular devam oranının yükselebileceğine işaret etmektedir.",
        "İlk sonuçlar öğrencilerin devamının artabileceğini düşündürmektedir.",
        "Ders programı yönetmelikte listelenmiştir.",
    ),
    (
        "psychology",
        "Kaygı düzeyi uyku süresiyle korelasyon gösterdi.",
        "Daha kaygılı katılımcılar daha kısa uyuma eğilimindeydi.",
        "Anket odasının duvarları maviye boyandı.",
    ),
    (
        "psychology",
        "Bazı katılımcılar müdahale sonrasında iyileşme bildirdi.",
        "Katılımcıların bir bölümü uygulamadan sonra iyileştiğini belirtti.",
        "Araştırmacılar toplantıyı erteledi.",
    ),
    (
        "environment",
        "Yağış miktarı toprak nemiyle ilişkili bulundu.",
        "Daha çok yağmur alan bölgelerde zemin nemi daha yüksekti.",
        "Milli parkın giriş kapısı yenilendi.",
    ),
    (
        "environment",
        "Sınırlı kanıt hava kirliliğinin azalabileceğini göstermektedir.",
        "Mevcut bulgular kirlilikte olası bir düşüşe işaret ediyor.",
        "Gözlem istasyonu deniz seviyesindedir.",
    ),
    (
        "statistics",
        "Tahminin %95 güven aralığı 3,2 ile 4,8 arasındadır.",
        "Kestirim için yüzde 95 güven sınırları 3,2 ve 4,8'dir.",
        "Tablonun başlığı ikinci sayfadadır.",
    ),
    (
        "statistics",
        "Örneklem n=1.248 katılımcıdan oluşmaktadır.",
        "Araştırma örnekleminde toplam 1.248 kişi yer aldı.",
        "Analiz yazılımı açık kaynaklıdır.",
    ),
    (
        "social_science",
        "Sosyal destek yaşam doyumuyla bağlantılı bulundu.",
        "Daha fazla destek alanların yaşam doyumu daha yüksekti.",
        "Görüşmeler belediye binasında yapıldı.",
    ),
    (
        "social_science",
        "Belirli katılımcılar programa olumlu yanıt verdi.",
        "Program bazı katılımcılarda olumlu sonuç verdi.",
        "Proje ekibi üç kentte toplandı.",
    ),
]


def build_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    categories = (
        "exact_copy",
        "near_copy",
        "punctuation_case",
        "word_reordering",
        "morphological_variation",
        "light_lexical_substitution",
        "close_paraphrase",
        "citation_present",
        "missing_citation",
        "quotation_needed",
        "legitimate_paraphrase",
        "semantic_related_not_copy",
        "unrelated",
        "common_academic_phrase",
        "technical_term_overlap",
        "short_generic_phrase",
        "near_copy",
        "legitimate_paraphrase",
    )
    for seed_index, (domain, source, paraphrase, unrelated) in enumerate(SEEDS):
        words = source.rstrip(".").split()
        for variant, category in enumerate(categories):
            positive = variant < 10 or variant == 16
            if category == "exact_copy":
                candidate = source
            elif category == "near_copy":
                candidate = " ".join(words[:-1]) + "."
            elif category == "punctuation_case":
                candidate = source.upper().replace(".", "!")
            elif category == "word_reordering":
                candidate = " ".join(words[2:] + words[:2]) + "."
            elif category == "morphological_variation":
                candidate = source.replace("çalışmada", "çalışmanın kapsamında").replace(
                    "katılımcı", "katılımcılar"
                )
            elif category == "light_lexical_substitution":
                candidate = source.replace("önemli ölçüde", "belirgin biçimde").replace(
                    "bulundu", "gözlendi"
                )
            elif category == "close_paraphrase":
                candidate = paraphrase + " " + " ".join(words[:4])
            elif category == "citation_present":
                candidate = source + " (Yılmaz, 2024)"
            elif category == "missing_citation":
                candidate = source
            elif category == "quotation_needed":
                candidate = f'"{source}" (Yılmaz, 2024)'
            elif category == "legitimate_paraphrase":
                candidate = paraphrase
            elif category == "semantic_related_not_copy":
                candidate = paraphrase.split(".")[0] + "; ilişki nedensellik anlamına gelmez."
            elif category == "unrelated":
                candidate = unrelated
            elif category == "common_academic_phrase":
                candidate = "Elde edilen bulgular istatistiksel olarak anlamlıdır."
            elif category == "technical_term_overlap":
                candidate = "Çalışmada güven aralığı ve örneklem büyüklüğü raporlandı."
            else:
                candidate = "Sonuçlar değerlendirildiğinde."
            cases.append(
                {
                    "id": f"tr-{seed_index:02d}-{variant:02d}",
                    "domain": domain,
                    "category": category,
                    "source": source,
                    "candidate": candidate,
                    "label": "TEXTUAL_OVERLAP_POSITIVE" if positive else "NEGATIVE",
                }
            )
    return cases


def _score(source: str, candidate: str, n: int = 3) -> float:
    def fp(text: str) -> Fingerprint:
        return Fingerprint(points=winnow(generate_shingles(normalize_turkish(text))))

    return max(exact_overlap(fp(source), fp(candidate)), char_ngram_jaccard(source, candidate, n=n))


def metrics(cases: list[dict[str, object]], threshold: float, n: int = 3) -> dict[str, object]:
    tp = fp = fn = tn = 0
    category_hits: Counter[str] = Counter()
    category_total: Counter[str] = Counter()
    for case in cases:
        expected = case["label"] == "TEXTUAL_OVERLAP_POSITIVE"
        predicted = _score(str(case["source"]), str(case["candidate"]), n) >= threshold
        tp += expected and predicted
        fp += not expected and predicted
        fn += expected and not predicted
        tn += not expected and not predicted
        category_total[str(case["category"])] += 1
        category_hits[str(case["category"])] += expected == predicted
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "cases": len(cases),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "per_category_accuracy": {
            key: category_hits[key] / value for key, value in category_total.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    cases = build_cases()
    if args.write_fixture:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shuffled = cases.copy()
    random.Random(SEED).shuffle(shuffled)
    split = round(len(shuffled) * 0.8)
    development, holdout = shuffled[:split], shuffled[split:]
    baseline = metrics(development, 0.70)
    choices = [
        (metrics(development, threshold), threshold)
        for threshold in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
    ]
    eligible = [item for item in choices if item[0]["precision"] >= 0.90]
    selected_metrics, threshold = max(
        eligible or choices, key=lambda item: (item[0]["f1"], item[0]["precision"])
    )
    result = {
        "kind": "turkish_engineering_not_external_accuracy",
        "seed": SEED,
        "development": baseline,
        "selected_threshold": threshold,
        "selected_development": selected_metrics,
        "holdout": metrics(holdout, threshold),
        "char_ngram": {str(n): metrics(development, threshold, n) for n in (3, 4, 5)},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "engineering_detection_final.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
