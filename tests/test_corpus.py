"""Tests for citeguard.corpus package."""

from __future__ import annotations

import pytest

from citeguard.corpus.deduplicate import deduplicate_entries, find_duplicates
from citeguard.corpus.ingest import _detect_language, ingest_directory, ingest_file
from citeguard.corpus.licenses import (
    LicenseType,
    classify_license,
    is_open_license,
    is_restricted_license,
)
from citeguard.corpus.models import (
    CorpusDocument,
    CorpusEntry,
    CorpusLanguage,
    CorpusMetadata,
)
from citeguard.corpus.normalize import normalize_corpus_text, strip_section_headers
from citeguard.corpus.synthetic import (
    ENGLISH_ACADEMIC_SENTENCES,
    TURKISH_ACADEMIC_SENTENCES,
    make_test_document,
    make_test_entry,
)

# ===================================================================
# Models
# ===================================================================


class TestCorpusModels:
    """Basic smoke tests for corpus data models."""

    def test_corpus_metadata_defaults(self):
        meta = CorpusMetadata(title="Test")
        assert meta.title == "Test"
        assert meta.authors == []
        assert meta.year is None
        assert meta.language == CorpusLanguage.TURKISH

    def test_corpus_entry_fields(self):
        entry = CorpusEntry(
            text="Hello world",
            normalized_text="hello world",
            doc_id="test-doc",
            entry_index=0,
        )
        assert entry.text == "Hello world"
        assert entry.doc_id == "test-doc"
        assert entry.fingerprint_hash is None

    def test_corpus_document_empty_entries(self):
        doc = CorpusDocument(
            doc_id="empty",
            metadata=CorpusMetadata(title="Empty"),
        )
        assert doc.entries == []
        assert doc.raw_text == ""

    def test_corpus_language_enum(self):
        assert CorpusLanguage.TURKISH.value == "tr"
        assert CorpusLanguage.ENGLISH.value == "en"


# ===================================================================
# Licenses
# ===================================================================


class TestLicenses:
    """License classification and validation tests."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("CC-BY", LicenseType.CC_BY),
            ("cc-by", LicenseType.CC_BY),
            ("CC BY", LicenseType.CC_BY),
            ("cc-by-4.0", LicenseType.CC_BY),
            ("CC-BY-SA", LicenseType.CC_BY_SA),
            ("cc-by-sa-4.0", LicenseType.CC_BY_SA),
            ("CC-BY-NC", LicenseType.CC_BY_NC),
            ("CC-BY-NC-SA", LicenseType.CC_BY_NC_SA),
            ("cc-by-nc-sa-4.0", LicenseType.CC_BY_NC_SA),
            ("CC-BY-ND", LicenseType.CC_BY_ND),
            ("CC-BY-NC-ND", LicenseType.CC_BY_NC_ND),
            ("CC0", LicenseType.CC0),
            ("cc0-1.0", LicenseType.CC0),
            ("public domain", LicenseType.PUBLIC_DOMAIN),
            ("PD", LicenseType.PUBLIC_DOMAIN),
            ("open access", LicenseType.OPEN_ACCESS),
            ("OA", LicenseType.OPEN_ACCESS),
            ("proprietary", LicenseType.PROPRIETARY),
            ("all rights reserved", LicenseType.PROPRIETARY),
        ],
    )
    def test_classify_known_licenses(self, raw, expected):
        assert classify_license(raw) == expected

    def test_classify_unknown(self):
        assert classify_license("some random string") == LicenseType.UNKNOWN

    def test_classify_none(self):
        assert classify_license(None) == LicenseType.UNKNOWN

    def test_classify_empty(self):
        assert classify_license("") == LicenseType.UNKNOWN

    def test_classify_with_whitespace(self):
        assert classify_license("  CC-BY  ") == LicenseType.CC_BY

    def test_classify_with_underscores(self):
        assert classify_license("cc_by") == LicenseType.CC_BY

    @pytest.mark.parametrize(
        "lt, expected",
        [
            (LicenseType.CC_BY, True),
            (LicenseType.CC_BY_SA, True),
            (LicenseType.CC_BY_NC, True),
            (LicenseType.CC0, True),
            (LicenseType.PUBLIC_DOMAIN, True),
            (LicenseType.OPEN_ACCESS, False),
            (LicenseType.CC_BY_ND, False),
            (LicenseType.CC_BY_NC_ND, False),
            (LicenseType.PROPRIETARY, False),
            (LicenseType.UNKNOWN, False),
        ],
    )
    def test_is_open_license(self, lt, expected):
        assert is_open_license(lt) is expected

    @pytest.mark.parametrize(
        "lt, expected",
        [
            (LicenseType.CC_BY_ND, True),
            (LicenseType.CC_BY_NC_ND, True),
            (LicenseType.PROPRIETARY, True),
            (LicenseType.UNKNOWN, True),
            (LicenseType.CC_BY, False),
            (LicenseType.CC0, False),
        ],
    )
    def test_is_restricted_license(self, lt, expected):
        assert is_restricted_license(lt) is expected


# ===================================================================
# Normalization
# ===================================================================


class TestCorpusNormalization:
    """Corpus text normalization tests."""

    def test_empty_string(self):
        assert normalize_corpus_text("") == ""

    def test_whitespace_collapse(self):
        assert normalize_corpus_text("hello   world") == "hello world"

    def test_bracketed_ref_removal(self):
        text = "Machine learning [1] is powerful [2,3]."
        result = normalize_corpus_text(text)
        assert "[1]" not in result
        assert "[2,3]" not in result

    def test_range_ref_removal(self):
        text = "Results shown in [1-5] confirm this."
        result = normalize_corpus_text(text)
        assert "[1-5]" not in result

    def test_turkish_lowercasing(self):
        result = normalize_corpus_text("İSTANBUL'da yapılan çalışma")
        assert "istanbul" in result

    def test_preserves_content(self):
        text = "Bu yöntem etkili sonuçlar vermiştir."
        result = normalize_corpus_text(text)
        assert "yöntem" in result
        assert "etkili" in result

    def test_strip_section_headers_numbered(self):
        text = "1. Giriş\nBu çalışma önemlidir.\n2.1 Yöntem\nDeneysel tasarım."
        result = strip_section_headers(text)
        assert "Giriş" not in result
        assert "çalışma" in result

    def test_strip_section_headers_no_header(self):
        text = "Basit bir paragraf."
        assert strip_section_headers(text) == text


# ===================================================================
# Deduplication
# ===================================================================


class TestDeduplication:
    """Corpus entry deduplication tests."""

    def _make_entry(self, text, doc_id="doc1", idx=0):
        return CorpusEntry(
            text=text,
            normalized_text=text.lower(),
            doc_id=doc_id,
            entry_index=idx,
        )

    def test_no_duplicates(self):
        entries = [
            self._make_entry("First sentence"),
            self._make_entry("Second sentence"),
        ]
        result = deduplicate_entries(entries)
        assert len(result) == 2

    def test_exact_duplicate_removed(self):
        entries = [
            self._make_entry("Same text", doc_id="doc1"),
            self._make_entry("Same text", doc_id="doc2"),
        ]
        result = deduplicate_entries(entries, cross_document=True)
        assert len(result) == 1

    def test_within_document_dedup(self):
        entries = [
            self._make_entry("Same text", doc_id="doc1", idx=0),
            self._make_entry("Same text", doc_id="doc1", idx=1),
            self._make_entry("Same text", doc_id="doc2", idx=0),
        ]
        result = deduplicate_entries(entries, cross_document=False)
        # doc1 keeps 1, doc2 keeps 1
        assert len(result) == 2

    def test_fingerprint_hash_set(self):
        entries = [self._make_entry("Test text")]
        result = deduplicate_entries(entries)
        assert result[0].fingerprint_hash is not None

    def test_find_duplicates_returns_groups(self):
        entries = [
            self._make_entry("Duplicate text", doc_id="doc1"),
            self._make_entry("Duplicate text", doc_id="doc2"),
            self._make_entry("Unique text", doc_id="doc3"),
        ]
        groups = find_duplicates(entries)
        assert len(groups) == 1
        group = list(groups.values())[0]
        assert len(group) == 2

    def test_find_duplicates_no_dupes(self):
        entries = [
            self._make_entry("Text A"),
            self._make_entry("Text B"),
        ]
        groups = find_duplicates(entries)
        assert len(groups) == 0

    def test_empty_input(self):
        assert deduplicate_entries([]) == []
        assert find_duplicates([]) == {}


# ===================================================================
# Ingest
# ===================================================================


class TestIngest:
    """File ingest tests using temporary fixtures."""

    def test_ingest_txt_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text(
            "Bu çalışmada önemli sonuçlar elde edilmiştir. "
            "Deneysel veriler hipotezi desteklemektedir.",
            encoding="utf-8",
        )
        doc = ingest_file(f, license_str="CC0")
        assert doc.doc_id == "test"
        assert len(doc.entries) > 0
        assert all(e.normalized_text for e in doc.entries)

    def test_ingest_md_file(self, tmp_path):
        f = tmp_path / "paper.md"
        f.write_text(
            "# Araştırma Makalesi\n\n"
            "Bu çalışma akademik yazımı incelemektedir. "
            "Sonuçlar istatistiksel olarak anlamlıdır.\n",
            encoding="utf-8",
        )
        doc = ingest_file(f, license_str="CC0")
        assert doc.doc_id == "paper"
        assert len(doc.entries) >= 1

    def test_ingest_custom_doc_id(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text(
            "Bu bir test cümlesidir ve yeterince uzundur.",
            encoding="utf-8",
        )
        doc = ingest_file(f, doc_id="custom-id", license_str="CC0")
        assert doc.doc_id == "custom-id"

    def test_ingest_unsupported_extension(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            ingest_file(f)

    def test_ingest_missing_file(self):
        with pytest.raises(FileNotFoundError):
            ingest_file("/nonexistent/path.txt")

    def test_ingest_with_license(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text(
            "Araştırma sonuçları önemli bulgular ortaya koymuştur.",
            encoding="utf-8",
        )
        doc = ingest_file(f, license_str="CC-BY")
        assert doc.metadata.license == "CC-BY"
        assert doc.metadata.similarity_index_allowed is True

    def test_ingest_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text(
            "İlk dosyadaki uzun bir akademik cümle burada yer almaktadır.",
            encoding="utf-8",
        )
        (tmp_path / "b.md").write_text(
            "İkinci dosyadaki uzun bir akademik cümle burada yer almaktadır.",
            encoding="utf-8",
        )
        (tmp_path / "c.pdf").write_text("ignored")
        docs = ingest_directory(tmp_path, license_str="CC0")
        assert len(docs) == 2

    def test_ingest_directory_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.txt").write_text(
            "Üst dizindeki uzun bir akademik cümle yer almaktadır.",
            encoding="utf-8",
        )
        (sub / "nested.txt").write_text(
            "Alt dizindeki uzun bir akademik cümle yer almaktadır.",
            encoding="utf-8",
        )
        docs = ingest_directory(tmp_path, recursive=True, license_str="CC0")
        assert len(docs) == 2

    def test_ingest_directory_not_a_dir(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            ingest_directory(f)

    def test_detect_language_turkish(self):
        text = "Çalışma sonuçları öğrencilerin başarı düzeyini göstermektedir."
        assert _detect_language(text) == CorpusLanguage.TURKISH

    def test_detect_language_english(self):
        text = "The study results demonstrate significant improvements."
        assert _detect_language(text) == CorpusLanguage.ENGLISH

    def test_short_entries_filtered(self, tmp_path):
        """Entries shorter than 10 chars after normalization are skipped."""
        f = tmp_path / "short.txt"
        f.write_text("Hi. OK.", encoding="utf-8")
        doc = ingest_file(f)
        assert len(doc.entries) == 0


# ===================================================================
# Synthetic
# ===================================================================


class TestSynthetic:
    """Synthetic test fixture generation tests."""

    def test_make_test_entry(self):
        entry = make_test_entry("Bu bir test cümlesidir.")
        assert entry.doc_id == "synthetic-test"
        assert entry.normalized_text
        assert entry.entry_index == 0

    def test_make_test_document(self):
        sentences = [
            "İlk cümle burada yer almaktadır.",
            "İkinci cümle burada yer almaktadır.",
        ]
        doc = make_test_document(sentences, doc_id="test-1")
        assert doc.doc_id == "test-1"
        assert len(doc.entries) == 2
        assert doc.metadata.license == "CC0"

    def test_turkish_fixtures_available(self):
        assert len(TURKISH_ACADEMIC_SENTENCES) >= 5

    def test_english_fixtures_available(self):
        assert len(ENGLISH_ACADEMIC_SENTENCES) >= 5

    def test_make_test_document_offsets_increasing(self):
        doc = make_test_document(TURKISH_ACADEMIC_SENTENCES[:3])
        offsets = [(e.char_offset, e.char_end) for e in doc.entries]
        for i in range(1, len(offsets)):
            assert offsets[i][0] >= offsets[i - 1][1]

    def test_make_test_document_language(self):
        doc = make_test_document(
            ["Test sentence."],
            language=CorpusLanguage.ENGLISH,
        )
        assert doc.metadata.language == CorpusLanguage.ENGLISH
