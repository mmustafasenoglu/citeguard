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
    spec = importlib.util.spec_from_file_location(
        "rewrite_safety",
        Path(__file__).parents[1] / "scripts/benchmarks/run_turkish_rewrite_safety.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    root = Path(__file__).parents[1] / "benchmarks/turkish/rewrite_safety.json"
    cases = json.loads(root.read_text(encoding="utf-8"))
    assert all(not module.validate_case(case) for case in cases)


def test_group_aware_split_has_no_source_leakage() -> None:
    spec = importlib.util.spec_from_file_location(
        "rewrite_safety",
        Path(__file__).parents[1] / "scripts/benchmarks/run_turkish_rewrite_safety.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    root = Path(__file__).parents[1] / "benchmarks/turkish/rewrite_safety.json"
    cases = json.loads(root.read_text(encoding="utf-8"))
    splits = module.group_aware_split(cases)
    groups = [set(case["group"] for case in values) for values in splits.values()]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_detection_v2_is_large_and_group_isolated() -> None:
    root = Path(__file__).parents[1]
    fixture = json.loads(
        (root / "benchmarks/turkish/detection_v2.json").read_text(encoding="utf-8")
    )
    result = json.loads(
        (root / "benchmarks/results/turkish/engineering_detection_v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(fixture) >= 800
    assert len({case["group"] for case in fixture}) == 60
    assert result["group_aware"] is True
    assert result["holdout_touched_before_selection"] is False



def test_turkish_author_normalization_and_vd_matching() -> None:
    from citeguard.bibliography import _normalize_author_key, citation_matches_entry
    from citeguard.models import BibliographyEntry, ExistingCitation

    assert _normalize_author_key("Şenoğlu, M.") == "senoglu"
    assert _normalize_author_key("Senoglu, M.") == "senoglu"
    assert _normalize_author_key("Şenoğlu vd.") == "senoglu"
    assert _normalize_author_key("Demirkol, D.") == "demirkol"
    assert _normalize_author_key("Ağaoğlu, K.") == "agaoglu"
    assert _normalize_author_key("Işık, A.") == "isik"

    cit = ExistingCitation(
        raw_text="(Şenoğlu vd., 2026)",
        authors="Şenoğlu vd.",
        year=2026,
        doi=None,
        numbered_ref=None,
        paragraph_index=1,
        char_offset=10,
    )
    entry = BibliographyEntry(
        raw_text="• Şenoğlu, M. (2026). Test Paper.",
        authors="• Şenoğlu, M",
        year=2026,
        title="Test Paper",
        doi=None,
        numbered_ref=None,
    )
    assert citation_matches_entry(cit, entry) is True


def test_audit_preserves_cited_claims_with_paragraph_offset() -> None:
    from citeguard.audit import AuditOptions, _extract_claims_product
    from citeguard.models import ExistingCitation, ParsedDocument

    paragraphs = [
        "First paragraph with background info.",
        "This sentence cites previous research (Şenoğlu vd., 2026).",
    ]
    citations = [
        ExistingCitation(
            raw_text="(Şenoğlu vd., 2026)",
            authors="Şenoğlu vd.",
            year=2026,
            doi=None,
            numbered_ref=None,
            paragraph_index=1,
            char_offset=38,
        )
    ]
    parsed = ParsedDocument(
        path="test.md",
        paragraphs=paragraphs,
        citations=citations,
        bibliography_entries=[],
        bibliography_start_index=None,
    )
    claims, _ = _extract_claims_product(parsed, AuditOptions(offline=True))
    cited = [c for c in claims if c.has_existing_citation]
    assert len(cited) == 1
    assert cited[0].paragraph_index == 1
    assert "(Şenoğlu vd., 2026)" in [c.raw_text for c in cited[0].linked_citations]


def test_turkish_claim_signals_and_search_query() -> None:
    from citeguard.claims import (
        _CAUSAL_SIGNALS,
        _COMPARATIVE_SIGNALS,
        _STATISTIC_SIGNALS,
        _build_search_query,
        _is_non_claim_paragraph,
    )

    # 1. Statistic signals
    assert _STATISTIC_SIGNALS.search("EFFIS verileri %20 oranında yanık alanları temsil eder.")
    assert _STATISTIC_SIGNALS.search("Saha yaklaşık 30 hektar alandan oluşmaktadır.")
    assert _STATISTIC_SIGNALS.search("Model yüzde 95 doğruluk oranına ulaştı.")

    # 2. Causal signals
    assert _CAUSAL_SIGNALS.search("Yangınlar önemli ekolojik kayıplara yol açmaktadır.")
    assert _CAUSAL_SIGNALS.search("İklim değişikliği kuraklığa neden olur.")

    # 3. Comparative signals
    assert _COMPARATIVE_SIGNALS.search("Attention U-Net klasik modele kıyasla daha yüksek IoU verdi.")

    # 4. Form instruction filtering
    inst = "Araştırmada yer alacak başlıca faaliyetler “Çalışma Takvimi” doldurularak sunulur."
    assert _is_non_claim_paragraph(inst) is True

    # 5. Search query unicode & Turkish preservation
    s = "Orman yangınları, özellikle Akdeniz iklim kuşağındaki orman ekosistemlerinde önemli ekolojik ve ekonomik kayıplara yol açmaktadır."
    q = _build_search_query(s)
    assert "yangınları" in q
    assert "özellikle" in q
    assert "ve" not in q.split()
