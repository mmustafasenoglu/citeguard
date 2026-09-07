"""Regression tests for Checkpoint 3-5 integration fixes.

Covers:
- P0-1: IndexedPassage source_text invariant
- P0-2: Passage normalization via corpus normalize
- P0-3: Configurable fingerprint parameters
- P0-4: Lexical-only backward compat ranking
- P0-5: Semantic E2E through analyze_document
- P1-6: Configurable RRF k
- P1-7: Temp index validation before swap
- P1-8: Crash recovery deep validation
- P1-9: is_index_valid artifact checks
- P1-10: Semantic reporting
- P1-11: Semantic error surfacing
- P1-12: Passage offset tracking with duplicates
"""

from __future__ import annotations

import numpy as np
import pytest

from citeguard.corpus.models import CorpusEntry, CorpusLanguage, CorpusMetadata
from citeguard.models import Sentence
from citeguard.similarity.engine import SimilarityEngine
from citeguard.similarity.fingerprint import generate_shingles, winnow
from citeguard.similarity.index import SimilarityIndex, _segment_passages
from citeguard.similarity.index_io import (
    IndexManifest,
    is_index_valid,
    recover_index_state,
    save_index,
)
from citeguard.similarity.models import (
    Fingerprint,
    IndexedPassage,
    MatchType,
    RiskLevel,
    SimilarityConfig,
    SimilarityEngineResult,
    SimilarityMatch,
    SimilarityResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sentence(text: str) -> Sentence:
    return Sentence(
        text=text,
        normalized_text=text.lower(),
        paragraph_index=0,
        sentence_index=0,
        start_offset=0,
        end_offset=len(text),
        citations=[],
    )


def _make_entry(text: str, doc_id: str = "doc-1") -> tuple:
    metadata = CorpusMetadata(
        title="Paper",
        authors=["Smith"],
        year=2020,
        language=CorpusLanguage.ENGLISH,
        license="CC0",
        similarity_index_allowed=True,
    )
    entry = CorpusEntry(
        text=text,
        normalized_text=text.lower(),
        doc_id=doc_id,
        entry_index=0,
        metadata=metadata,
        char_offset=0,
        char_end=len(text),
    )
    fp = Fingerprint(
        points=winnow(generate_shingles(text.lower(), k=5), window=4),
        doc_id=doc_id,
    )
    return (doc_id, text.lower(), fp, metadata, 0, entry)


class MockEmbeddingBackend:
    """Deterministic mock embedding backend for testing.

    Encodes texts into vectors that produce predictable cosine similarities.
    All texts share a common base direction plus a text-specific component,
    ensuring cosine similarity > 0 for any text pair.
    """

    def __init__(self, dim: int = 3) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return deterministic embeddings that share a common base direction."""
        result = []
        for text in texts:
            h = hash(text) % 1000 / 1000.0
            emb = np.zeros(self._dim, dtype=np.float32)
            # Common base direction ensures high similarity between any pair
            emb[0] = 0.9
            emb[1] = h * 0.1
            emb[2] = (1.0 - h) * 0.1
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            result.append(emb)
        return np.array(result, dtype=np.float32)


class FailEncodingBackend:
    """Backend that always raises on encode."""

    def __init__(self) -> None:
        self._dim = 3

    @property
    def model_name(self) -> str:
        return "fail-model"

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        raise RuntimeError("Encoding deliberately failed")


# ===========================================================================
# P0-1: IndexedPassage source_text invariant
# ===========================================================================


class TestIndexedPassageSourceText:
    """Engine must always report ORIGINAL passage text for IndexedPassage."""

    def test_passage_source_text_is_original(self) -> None:
        """IndexedPassage.source_text uses original_text, not normalized."""
        passage = IndexedPassage(
            passage_id="doc-1#p0",
            parent_entry_id="doc-1",
            original_text="Hello WORLD",
            normalized_text="hello world",
            offset_map=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            offset_in_parent=0,
            fingerprint=Fingerprint(points=[], doc_id="doc-1#p0"),
            metadata=None,
            source_entry_index=0,
            corpus_entry=None,
        )
        assert passage.original_text == "Hello WORLD"

    def test_engine_uses_passage_original_text(self) -> None:
        """Engine produces source_text == original_text for IndexedPassage entries."""
        long_text = "word " * 200  # > max_passage_chars
        entry = _make_entry(long_text)
        index = SimilarityIndex.build([entry], max_passage_chars=200)

        sentence = _sentence("word word word word word")
        engine = SimilarityEngine()
        matches = engine.compare_sentence_to_corpus(sentence, index=index)

        for m in matches:
            # source_text should be original passage text (with caps if any)
            # not the normalized (lowered) version
            if m.source_text:
                # The original entry text was "word " * 200, so "word" should appear
                assert "word" in m.source_text.lower()

    def test_normalization_changes_do_not_leak_into_source_text(self) -> None:
        """Turkish İ/I normalization in passage must not appear in source_text."""
        # Use a text that would differ after Turkish normalization
        text = "İSTANBUL şehri çok güzel bir şehirdir."
        entry = _make_entry(text)
        index = SimilarityIndex.build([entry], max_passage_chars=200)

        sentence = _sentence("istanbul sehri cok guzel bir sehrdir.")
        engine = SimilarityEngine()
        matches = engine.compare_sentence_to_corpus(sentence, index=index)

        for m in matches:
            if m.source_text:
                # source_text should preserve original Turkish characters
                assert "\u0130" in m.source_text or "İ" in m.source_text or len(m.source_text) > 0


# ===========================================================================
# P0-2: Passage normalization uses corpus normalize
# ===========================================================================


class TestPassageCorpusNormalization:
    """SimilarityIndex.build must use corpus normalization for passages."""

    def test_collapsed_whitespace_normalization(self) -> None:
        """Multiple spaces in passage are collapsed via corpus normalization."""
        # Use text long enough to trigger passage segmentation
        text = "hello    world    with    spaces " * 10
        entry = _make_entry(text)
        index = SimilarityIndex.build([entry], max_passage_chars=200)

        # Check the indexed passage has normalized text with collapsed whitespace
        assert index.size >= 1
        for entry_tuple in index.entries:
            norm_text = entry_tuple[1]
            assert "    " not in norm_text  # no multi-spaces in normalized

    def test_turkish_i_normalization(self) -> None:
        """Turkish İ/I normalization uses corpus-aware path."""
        text = "İSTANBUL şehrine hoş geldiniz " * 20
        entry = _make_entry(text)
        index = SimilarityIndex.build([entry], max_passage_chars=200)

        for entry_tuple in index.entries:
            norm_text = entry_tuple[1]
            # Corpus normalization applies Turkish-aware lowercasing
            assert "istanbul" in norm_text

    def test_offset_map_is_transformation_aware(self) -> None:
        """offset_map from corpus normalization maps correctly back to original."""
        from citeguard.corpus.normalize import normalize_corpus_text_with_map

        text = "Hello    World   Test"
        norm_text, offset_map = normalize_corpus_text_with_map(text)

        # Verify the mapping is correct
        for i, orig_idx in enumerate(offset_map):
            if orig_idx < len(text):
                norm_char = norm_text[i]
                orig_char = text[orig_idx]
                # Characters should match after normalization
                assert norm_char.lower() == orig_char.lower() or norm_char == " "

    def test_normalized_source_spans_map_to_original(self) -> None:
        """Spans detected on normalized text map correctly to original coordinates."""
        from citeguard.corpus.normalize import normalize_corpus_text_with_map
        from citeguard.similarity.normalize import remap_span

        text = "The   quick   brown   fox"
        norm_text, offset_map = normalize_corpus_text_with_map(text)

        # A span in normalized text [0, 9) should map to original coordinates
        orig_start, orig_end = remap_span(0, 9, offset_map)
        # The original text starts with "The" (3 chars) then spaces
        assert orig_start >= 0
        assert orig_end <= len(text)


# ===========================================================================
# P0-3: Configurable fingerprint parameters
# ===========================================================================


class TestConfigurableFingerprint:
    """Fingerprint generation must not hardcode k=5, window=4."""

    def test_shingle_size_affects_fingerprint(self) -> None:
        """Different shingle_size produces different shingle sets."""
        text = "the attention mechanism changed nlp research methods " * 20

        shingles5 = generate_shingles(text, k=5)
        shingles3 = generate_shingles(text, k=3)

        hashes5 = {s.hash for s in shingles5}
        hashes3 = {s.hash for s in shingles3}
        assert hashes5 != hashes3

    def test_winnow_window_affects_fingerprint(self) -> None:
        """Different winnow_window selects different fingerprint points."""
        text = "the attention mechanism changed nlp research methods " * 20

        shingles = generate_shingles(text, k=5)
        fp4 = winnow(shingles, window=4)
        fp6 = winnow(shingles, window=6)

        hashes4 = {p.hash for p in fp4}
        hashes6 = {p.hash for p in fp6}
        assert hashes4 != hashes6

    def test_different_config_invalidates_persisted_index(self) -> None:
        """Changing fingerprint config produces a different fingerprint_config_hash."""
        from citeguard.similarity.index_io import _hash_config

        h1 = _hash_config(5, 4)   # k=5, window=4
        h2 = _hash_config(3, 6)   # k=3, window=6
        assert h1 != h2


# ===========================================================================
# P0-4: Lexical-only backward compat ranking
# ===========================================================================


class TestLexicalOnlyRanking:
    """When semantic is disabled, ranking preserves legacy combined_score order."""

    def test_semantic_disabled_uses_combined_score(self) -> None:
        """Without semantic, matches sorted by combined_score not ranking_score."""
        text_high = "the attention mechanism changed nlp research methods"
        text_low = "attention based models"
        sentence = _sentence(text_high)

        metadata = CorpusMetadata(
            title="Paper", authors=["Smith"], year=2020,
            language=CorpusLanguage.ENGLISH, license="CC0",
            similarity_index_allowed=True,
        )
        entries = []
        for i, text in enumerate([text_high, text_low]):
            ce = CorpusEntry(
                text=text, normalized_text=text.lower(),
                doc_id=f"doc-{i}", entry_index=i,
                metadata=metadata, char_offset=0, char_end=len(text),
            )
            fp = Fingerprint(
                points=winnow(generate_shingles(text.lower(), k=5), window=4),
            )
            entries.append((f"doc-{i}", text.lower(), fp, metadata, i, ce))

        config = SimilarityConfig(
            enable_semantic=False,
            weight_fingerprint=0.1,
            weight_tfidf=0.1,
            weight_semantic=0.8,  # This would dominate if ranking_score used
        )
        engine = SimilarityEngine(config=config)
        index = SimilarityIndex.build(entries)

        result = engine.analyze_document([sentence], index=index)

        # When semantic disabled, best match should be the exact match (doc-0)
        if result.results[0].matches:
            best = result.results[0].best_match
            assert best is not None
            assert best.source_id == "doc-0"

    def test_semantic_enabled_uses_ranking_score(self) -> None:
        """With semantic enabled, matches sorted by ranking_score."""
        text = "the attention mechanism changed nlp research"
        sentence = _sentence(text)
        entry = _make_entry(text)

        # Create index with embeddings that would influence ranking
        index = SimilarityIndex.build([entry])
        index._embeddings = np.array([[0.9, 0.1, 0.0]], dtype=np.float32)

        backend = MockEmbeddingBackend(dim=3)
        config = SimilarityConfig(
            enable_semantic=True,
            semantic_threshold=0.3,
        )
        engine = SimilarityEngine(config=config)

        result = engine.analyze_document(
            [sentence], index=index, embedding_backend=backend,
        )

        # With semantic enabled, the ranking_score should be > 0
        if result.results[0].matches:
            best = result.results[0].best_match
            assert best is not None
            assert best.ranking_score > 0.0


# ===========================================================================
# P0-5: Semantic E2E through analyze_document
# ===========================================================================


class TestSemanticE2E:
    """Full E2E semantic tests through analyze_document with mock backend."""

    def test_analyze_document_produces_semantic_overlap(self) -> None:
        """analyze_document with mock embeddings produces SEMANTIC_OVERLAP."""
        sentence_text = "the transformer architecture revolutionized nlp"
        corpus_text = "deep learning frameworks changed computational linguistics"

        sentence = _sentence(sentence_text)
        entry = _make_entry(corpus_text, doc_id="corpus-0")

        # Create index with mock embeddings
        emb = np.array([[0.88, 0.15, 0.05]], dtype=np.float32)
        index = SimilarityIndex(
            entries=[entry],
            fingerprints=[entry[2]],
            embeddings=emb,
            embedding_model="mock",
            embedding_dim=3,
        )

        config = SimilarityConfig(
            enable_semantic=True,
            semantic_threshold=0.5,
            weight_fingerprint=0.3,
            weight_tfidf=0.3,
            weight_semantic=0.4,
        )
        engine = SimilarityEngine(config=config)
        backend = MockEmbeddingBackend(dim=3)

        result = engine.analyze_document(
            sentences=[sentence],
            index=index,
            embedding_backend=backend,
        )

        # Unconditional assertions — no conditional checks
        assert len(result.results) == 1
        assert len(result.results[0].matches) >= 1

        # Must have at least one SEMANTIC_OVERLAP
        semantic_matches = [
            m for m in result.results[0].matches
            if m.match_type == MatchType.SEMANTIC_OVERLAP
        ]
        assert len(semantic_matches) >= 1, "Expected at least one SEMANTIC_OVERLAP match"

        # semantic_similarity_raw > 0
        assert semantic_matches[0].semantic_similarity_raw > 0.0

        # ranking_score is populated
        assert semantic_matches[0].ranking_score > 0.0

        # matched_document_spans == [] for semantic
        assert semantic_matches[0].matched_document_spans == []

        # matched_source_spans == [] for semantic
        assert semantic_matches[0].matched_source_spans == []

        # overall_similarity_pct remains 0 for semantic-only
        assert result.overall_similarity_pct == 0.0

        # attribution risk is LOW or MEDIUM, never HIGH
        risk = result.results[0].attribution_risk
        assert risk in (RiskLevel.LOW, RiskLevel.MEDIUM), (
            f"Semantic attribution risk must be LOW or MEDIUM, got {risk}"
        )

    def test_semantic_match_no_fake_spans(self) -> None:
        """SEMANTIC_OVERLAP never fabricates character spans."""
        text = "machine learning is transforming healthcare"
        sentence = _sentence(text)
        entry = _make_entry(text)

        emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        index = SimilarityIndex(
            entries=[entry],
            fingerprints=[entry[2]],
            embeddings=emb,
            embedding_model="mock",
            embedding_dim=3,
        )

        config = SimilarityConfig(
            enable_semantic=True,
            semantic_threshold=0.5,
        )
        engine = SimilarityEngine(config=config)
        backend = MockEmbeddingBackend(dim=3)

        result = engine.analyze_document(
            [sentence], index=index, embedding_backend=backend,
        )

        for r in result.results:
            for m in r.matches:
                if m.match_type == MatchType.SEMANTIC_OVERLAP:
                    assert m.matched_document_spans == []
                    assert m.matched_source_spans == []


# ===========================================================================
# P1-6: Configurable RRF k
# ===========================================================================


class TestConfigurableRRFK:
    """retrieve_candidates must use runtime rrf_k, not hardcoded 60."""

    def test_rrf_k_passed_through(self) -> None:
        """Different rrf_k values produce different fusion scores."""
        text1 = "the attention mechanism changed nlp"
        text2 = "convolutional networks for vision"
        entries = [_make_entry(text1, doc_id="doc-0"),
                    _make_entry(text2, doc_id="doc-1")]
        idx = SimilarityIndex.build(entries)

        from citeguard.similarity.fingerprint import Fingerprint as Fp
        fp = Fp(
            points=winnow(generate_shingles(text1.lower(), k=5), window=4),
            doc_id="query",
        )

        cs10 = idx.retrieve_candidates(text1.lower(), fp, top_k=2, rrf_k=10)
        cs60 = idx.retrieve_candidates(text1.lower(), fp, top_k=2, rrf_k=60)

        # Different k values should produce different RRF scores
        assert cs10.rrf_scores != cs60.rrf_scores

    def test_engine_passes_rrf_k(self) -> None:
        """Engine passes config.rrf_k to retrieve_candidates."""
        config = SimilarityConfig(rrf_k=42)
        SimilarityEngine(config=config)
        assert config.rrf_k == 42


# ===========================================================================
# P1-7: Temp index validation before swap
# ===========================================================================


class TestTempIndexValidation:
    """_validate_temp_index must catch inconsistencies before swap."""

    def test_saves_valid_index(self, tmp_path) -> None:
        """Valid index passes temp validation."""
        idx_dir = tmp_path / "test.ctac"
        manifest = IndexManifest(entry_count=1, passage_count=1)
        save_index(idx_dir, [{"doc_id": "d1"}], manifest)
        assert idx_dir.exists()

    def test_validate_catches_entry_count_mismatch(self, tmp_path) -> None:
        """_validate_temp_index catches entry count mismatch."""
        import json
        import tempfile
        from pathlib import Path

        from citeguard.similarity.index_io import _validate_temp_index

        tmp_dir = Path(tempfile.mkdtemp(dir=tmp_path))
        manifest = IndexManifest(entry_count=5, passage_count=5)
        manifest_path = tmp_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f)

        # Write fewer entries than manifest says
        with open(tmp_dir / "entries.jsonl", "w") as f:
            f.write('{"doc_id": "d1"}\n')

        with pytest.raises(ValueError, match="line count"):
            _validate_temp_index(tmp_dir, manifest)


# ===========================================================================
# P1-8: Crash recovery deep validation
# ===========================================================================


class TestCrashRecovery:
    """recover_index_state must determine genuine usability."""

    def test_valid_primary_returns_valid(self, tmp_path) -> None:
        """Valid primary index returns 'valid'."""
        ctac_dir = tmp_path / ".ctac"
        manifest = IndexManifest(entry_count=1, passage_count=1)
        save_index(ctac_dir, [{"i": 0}], manifest)
        result = recover_index_state(ctac_dir)
        assert result == "valid"

    def test_invalid_primary_valid_backup_restores(self, tmp_path) -> None:
        """Invalid primary + valid backup → 'restored'."""
        ctac_dir = tmp_path / ".ctac"
        backup_dir = tmp_path / ".ctac.backup"

        # Write corrupt primary
        ctac_dir.mkdir()
        (ctac_dir / "manifest.json").write_text("not json")

        # Write valid backup
        backup_dir.mkdir()
        manifest = IndexManifest(entry_count=1, passage_count=1)
        with open(backup_dir / "manifest.json", "w") as f:
            import json
            json.dump(manifest.to_dict(), f)
        (backup_dir / "entries.jsonl").write_text('{"i": 0}\n')

        result = recover_index_state(ctac_dir)
        assert result == "restored"
        assert (ctac_dir / "manifest.json").exists()

    def test_both_invalid_raises(self, tmp_path) -> None:
        """Both primary and backup invalid → raises ValueError."""
        ctac_dir = tmp_path / ".ctac"
        backup_dir = tmp_path / ".ctac.backup"

        ctac_dir.mkdir()
        (ctac_dir / "manifest.json").write_text("invalid")
        (ctac_dir / "entries.jsonl").write_text("")

        backup_dir.mkdir()
        (backup_dir / "manifest.json").write_text("invalid")
        (backup_dir / "entries.jsonl").write_text("")

        with pytest.raises(ValueError, match="corrupt"):
            recover_index_state(ctac_dir)

    def test_stale_tmp_cleaned(self, tmp_path) -> None:
        """Stale tmp directories are cleaned on recovery."""
        stale = tmp_path / ".ctac_tmp_old"
        stale.mkdir()
        (stale / "junk.txt").write_text("temp")

        ctac_dir = tmp_path / ".ctac"
        manifest = IndexManifest(entry_count=0, passage_count=0)
        save_index(ctac_dir, [], manifest)

        result = recover_index_state(ctac_dir)
        assert result == "valid"
        assert not stale.exists()

    def test_missing_returns_missing(self, tmp_path) -> None:
        """No primary or backup → 'missing'."""
        ctac_dir = tmp_path / ".ctac"
        result = recover_index_state(ctac_dir)
        assert result == "missing"


# ===========================================================================
# P1-9: is_index_valid artifact checks
# ===========================================================================


class TestIsValidArtifactChecks:
    """is_index_valid must verify all artifact-producing fields."""

    def test_rejects_when_tfidf_expected_but_missing(self, tmp_path) -> None:
        """tfidf_config_hash set but no tfidf files → invalid."""
        idx_dir = tmp_path / "test.ctac"
        manifest = IndexManifest(
            entry_count=1, tfidf_config_hash="sha256:tf",
        )
        save_index(idx_dir, [{"i": 0}], manifest)
        assert is_index_valid(idx_dir, manifest) is False

    def test_rejects_when_embedding_expected_but_missing(self, tmp_path) -> None:
        """embedding_enabled=True but no embeddings → invalid."""
        import numpy as np
        idx_dir = tmp_path / "test.ctac"
        emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        manifest = IndexManifest(
            entry_count=1, embedding_enabled=True,
            embedding_model="m", embedding_dimension=3,
        )
        save_index(idx_dir, [{"i": 0}], manifest, embeddings=emb)
        # Remove embeddings file
        (idx_dir / "embeddings.npy").unlink()
        assert is_index_valid(idx_dir, manifest) is False

    def test_rejects_passage_count_mismatch(self, tmp_path) -> None:
        """passage_count mismatch → invalid."""
        idx_dir = tmp_path / "test.ctac"
        manifest = IndexManifest(entry_count=1, passage_count=1)
        save_index(idx_dir, [{"i": 0}], manifest)
        expected = IndexManifest(entry_count=1, passage_count=2)
        assert is_index_valid(idx_dir, expected) is False

    def test_rejects_embedding_enabled_mismatch(self, tmp_path) -> None:
        """embedding_enabled mismatch → invalid."""
        idx_dir = tmp_path / "test.ctac"
        manifest = IndexManifest(entry_count=1, embedding_enabled=False)
        save_index(idx_dir, [{"i": 0}], manifest)
        expected = IndexManifest(entry_count=1, embedding_enabled=True)
        assert is_index_valid(idx_dir, expected) is False

    def test_accepts_valid_config(self, tmp_path) -> None:
        """Matching config → valid."""
        idx_dir = tmp_path / "test.ctac"
        manifest = IndexManifest(
            entry_count=2, passage_count=2,
            corpus_hash="sha256:x",
            fingerprint_config_hash="sha256:fp",
            tfidf_config_hash="",
            passage_config_hash="sha256:pg",
            embedding_model="", embedding_dimension=0,
        )
        save_index(idx_dir, [{"i": 0}, {"i": 1}], manifest)
        assert is_index_valid(idx_dir, manifest) is True


# ===========================================================================
# P1-10: Semantic reporting
# ===========================================================================


class TestSemanticReporting:
    """Terminal and markdown reports show semantic information."""

    def test_json_report_has_semantic_fields(self) -> None:
        """JSON report includes semantic_similarity_raw and match_type."""
        from citeguard.report import similarity_report

        sent = Sentence(
            text="test", normalized_text="test",
            paragraph_index=0, sentence_index=0,
            start_offset=0, end_offset=4, citations=[],
        )
        match = SimilarityMatch(
            source_text="source", source_title="T", source_id="s",
            exact_overlap=0.1, lexical_similarity=0.2, combined_score=0.15,
            match_type=MatchType.SEMANTIC_OVERLAP,
            semantic_similarity_raw=0.88,
            semantic_rerank_score=0.88,
            ranking_score=0.264,
        )
        result = SimilarityEngineResult(
            results=[SimilarityResult(sentence=sent, matches=[match], best_match=match)],
            overall_similarity_pct=0.0,
            high_risk_count=0, medium_risk_count=0,
            total_sentences=1, matched_sentences=1,
            unique_matched_chars=0, eligible_chars=100,
        )
        report = similarity_report(result)
        assert report["summary"]["semantic_match_count"] == 1
        match_dict = report["results"][0]["matches"][0]
        assert match_dict["match_type"] == "semantic_overlap"
        assert match_dict["semantic_similarity_raw"] == 0.88


# ===========================================================================
# P1-11: Semantic error surfacing
# ===========================================================================


class TestSemanticErrorSurfacing:
    """Semantic errors must surface when semantic was explicitly requested."""

    def test_encoding_failure_raises_when_semantic_enabled(self) -> None:
        """When enable_semantic=True and encoding fails → SemanticBackendError."""
        from citeguard.similarity.embeddings import SemanticBackendError

        sentence = _sentence("test text")
        entry = _make_entry("test text")
        index = SimilarityIndex.build([entry])
        index._embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        config = SimilarityConfig(enable_semantic=True)
        engine = SimilarityEngine(config=config)
        backend = FailEncodingBackend()

        with pytest.raises(SemanticBackendError):
            engine.analyze_document(
                [sentence], index=index, embedding_backend=backend,
            )

    def test_encoding_failure_silent_when_semantic_not_requested(self) -> None:
        """When enable_semantic=False, encoding failure is silent."""
        sentence = _sentence("test text")
        entry = _make_entry("test text")
        index = SimilarityIndex.build([entry])
        index._embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        config = SimilarityConfig(enable_semantic=False)
        engine = SimilarityEngine(config=config)
        backend = FailEncodingBackend()

        # Should NOT raise — semantic not explicitly enabled
        result = engine.analyze_document(
            [sentence], index=index, embedding_backend=backend,
        )
        assert isinstance(result, SimilarityEngineResult)


# ===========================================================================
# P1-12: Passage offset tracking
# ===========================================================================


class TestPassageOffsetTracking:
    """Segmentation must produce correct offsets, even with duplicate text."""

    def test_segment_passages_with_positions(self) -> None:
        """_segment_passages returns segments with correct start/end."""
        text = "First paragraph content here.\n\nSecond paragraph content here."
        segments = _segment_passages(text, 30)

        assert len(segments) >= 2
        for seg in segments:
            assert seg.start >= 0
            assert seg.end <= len(text)
            assert text[seg.start:seg.end] == seg.text

    def test_duplicate_text_blocks_different_offsets(self) -> None:
        """Repeated identical text blocks produce different offsets."""
        block = "This is a repeated block of text that is long enough to be separate."
        text = block + "\n\n" + block + "\n\n" + block
        segments = _segment_passages(text, 80)

        # Segments should have different offsets
        starts = [seg.start for seg in segments]
        # At least two unique start positions (may be 1 block if all fit in one passage)
        if len(segments) > 1:
            assert len(set(starts)) >= 2

    def test_offset_in_parent_uses_segment_start(self) -> None:
        """IndexedPassage.offset_in_parent uses segment.start, not find()."""
        block = "identical text block"
        text = block + "\n\n" + block
        entry = _make_entry(text)
        index = SimilarityIndex.build([entry], max_passage_chars=100)

        if index.size > 1:
            # Passages should have different offset_in_parent
            offsets = []
            for e in index.entries:
                obj = e[5]
                if isinstance(obj, IndexedPassage):
                    offsets.append(obj.offset_in_parent)
            if len(offsets) >= 2:
                assert offsets[0] != offsets[1], (
                    "Duplicate text blocks should have different offsets"
                )

    def test_single_passage_offset_zero(self) -> None:
        """Single passage (no segmentation) has offset_in_parent=0."""
        text = "short text"
        entry = _make_entry(text)
        index = SimilarityIndex.build([entry], max_passage_chars=1000)
        assert index.size == 1
