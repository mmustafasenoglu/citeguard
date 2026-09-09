"""Turkish capability regression tests."""
# ruff: noqa: E501
import importlib.util
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
    assert len(safety) >= 250
    assert len({case["domain"] for case in detection}) >= 8
    assert len({case["category"] for case in safety}) >= 12


def test_rewrite_safety_cases_are_category_pure() -> None:
    spec = importlib.util.spec_from_file_location("rewrite_safety", Path(__file__).parents[1] / "scripts/benchmarks/run_turkish_rewrite_safety.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    root = Path(__file__).parents[1] / "benchmarks/turkish/rewrite_safety.json"
    cases = json.loads(root.read_text(encoding="utf-8"))
    assert all(not module.validate_case(case) for case in cases)


def test_group_aware_split_has_no_source_leakage() -> None:
    spec = importlib.util.spec_from_file_location("rewrite_safety", Path(__file__).parents[1] / "scripts/benchmarks/run_turkish_rewrite_safety.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    root = Path(__file__).parents[1] / "benchmarks/turkish/rewrite_safety.json"
    cases = json.loads(root.read_text(encoding="utf-8"))
    splits = module.group_aware_split(cases)
    groups = [set(case["group"] for case in values) for values in splits.values()]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
