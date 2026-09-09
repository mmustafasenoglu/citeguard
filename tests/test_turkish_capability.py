import json
import unicodedata
from pathlib import Path

from citeguard.similarity.lexical import normalize_turkish


def test_turkish_normalization_unifies_unicode_and_punctuation_variants() -> None:
    nfd = unicodedata.normalize("NFD", "İlişkili ölçüm")
    assert normalize_turkish(nfd) == normalize_turkish("İLİŞKİLİ ÖLÇÜM")
    assert normalize_turkish("Türkiye’nin sonucu") == normalize_turkish("Türkiye'nin sonucu")
    assert normalize_turkish("önemli–ölçüm") == normalize_turkish("önemli-ölçüm")


def test_turkish_engineering_suites_meet_minimum_sizes() -> None:
    root = Path(__file__).parents[1] / "benchmarks/turkish"
    detection = json.loads((root / "detection_engineering.json").read_text(encoding="utf-8"))
    safety = json.loads((root / "rewrite_safety.json").read_text(encoding="utf-8"))
    assert len(detection) >= 300
    assert len(safety) >= 150
    assert len({case["domain"] for case in detection}) >= 8
    assert len({case["category"] for case in safety}) >= 12
