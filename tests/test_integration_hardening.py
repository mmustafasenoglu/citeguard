"""Phase 4 E2E integration tests for hardening Checkpoints 3-5."""

from __future__ import annotations

import numpy as np

from citeguard.corpus.models import CorpusEntry, CorpusLanguage, CorpusMetadata
from citeguard.models import Sentence
from citeguard.report import similarity_report
from citeguard.similarity.engine import SimilarityEngine
from citeguard.similarity.fingerprint import generate_shingles, winnow
from citeguard.similarity.index import SimilarityIndex
from citeguard.similarity.index_io import (
    IndexManifest,
    compute_corpus_hash,
    is_index_valid,
    load_index,
    load_tfidf_matrix,
    recover_index_state,
    save_index,
)
from citeguard.similarity.models import (
    CandidateSet,
    Fingerprint,
    IndexedPassage,
    MatchType,
    SimilarityConfig,
    SimilarityEngineResult,
    SimilarityMatch,
    SimilarityResult,
)


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


# ---------------------------------------------------------------------------
# E2E: Semantic-only match through engine
# ---------------------------------------------------------------------------


def test_e2e_semantic_only_match_end_to_end() -> None:
    """Engine produces SEMANTIC_OVERLAP via mock embeddings."""
    sentence_text = "the transformer architecture revolutionized nlp"
    corpus_text = "deep learning frameworks changed computational linguistics"

    sentence = _sentence(sentence_text)
    entry = _make_entry(corpus_text, doc_id="corpus-0")

    corpus_emb = np.array([[0.88, 0.15, 0.05]], dtype=np.float32)
    query_emb = np.array([0.9, 0.1, 0.0], dtype=np.float32)

    index = SimilarityIndex(
        entries=[entry],
        fingerprints=[entry[2]],
        embeddings=corpus_emb,
        embedding_model="test",
        embedding_dim=3,
    )

    sem_hits = index.retrieve_semantic(query_emb, top_k=5)
    semantic_scores = dict(sem_hits)

    config = SimilarityConfig(semantic_threshold=0.5, enable_semantic=True)
    engine = SimilarityEngine(config=config)
    matches = engine.compare_sentence_to_corpus(
        sentence, index=index, semantic_scores=semantic_scores,
    )

    semantic_matches = [
        m for m in matches if m.match_type == MatchType.SEMANTIC_OVERLAP
    ]
    assert len(semantic_matches) >= 1

    m = semantic_matches[0]
    assert m.semantic_similarity_raw > 0.0
    assert m.ranking_score > 0.0
    assert m.matched_document_spans == []
    assert m.matched_source_spans == []


def test_e2e_semantic_excluded_from_textual_pct() -> None:
    """SEMANTIC_OVERLAP matches have no spans -> overall similarity = 0."""
    sentence_text = "the transformer architecture revolutionized nlp"
    corpus_text = "deep learning frameworks changed computational linguistics"

    sentence = _sentence(sentence_text)
    entry = _make_entry(corpus_text, doc_id="corpus-0")

    corpus_emb = np.array([[0.88, 0.15, 0.05]], dtype=np.float32)

    index = SimilarityIndex(
        entries=[entry],
        fingerprints=[entry[2]],
        embeddings=corpus_emb,
        embedding_model="test",
        embedding_dim=3,
    )

    config = SimilarityConfig(semantic_threshold=0.5, enable_semantic=True)
    engine = SimilarityEngine(config=config)

    result = engine.analyze_document(
        sentences=[sentence],
        index=index,
    )

    all_semantic = all(
        m.match_type == MatchType.SEMANTIC_OVERLAP
        for r in result.results
        for m in r.matches
    )
    if all_semantic and result.results[0].matches:
        assert result.overall_similarity_pct == 0.0


def test_e2e_semantic_attribution_risk_never_high() -> None:
    """Semantic-only match -> risk is LOW or MEDIUM, never HIGH."""
    sentence_text = "the transformer architecture revolutionized nlp"
    corpus_text = "deep learning frameworks changed computational linguistics"

    sentence = _sentence(sentence_text)
    entry = _make_entry(corpus_text, doc_id="corpus-0")

    corpus_emb = np.array([[0.88, 0.15, 0.05]], dtype=np.float32)
    query_emb = np.array([0.9, 0.1, 0.0], dtype=np.float32)

    index = SimilarityIndex(
        entries=[entry],
        fingerprints=[entry[2]],
        embeddings=corpus_emb,
        embedding_model="test",
        embedding_dim=3,
    )

    sem_hits = index.retrieve_semantic(query_emb, top_k=5)
    semantic_scores = dict(sem_hits)

    config = SimilarityConfig(semantic_threshold=0.5, enable_semantic=True)
    engine = SimilarityEngine(config=config)
    matches = engine.compare_sentence_to_corpus(
        sentence, index=index, semantic_scores=semantic_scores,
    )

    for m in matches:
        if m.match_type == MatchType.SEMANTIC_OVERLAP:
            risk, _reason = engine.compute_attribution_risk(m, [], [])
            assert risk.value in ("low", "medium"), (
                f"Semantic match risk should be LOW or MEDIUM, got {risk.value}"
            )


# ---------------------------------------------------------------------------
# Persisted index roundtrip
# ---------------------------------------------------------------------------


def test_persisted_index_roundtrip(tmp_path) -> None:
    """save -> load -> retrieve works for TF-IDF + embeddings."""
    from pathlib import Path

    text1 = "the attention mechanism changed nlp"
    text2 = "convolutional networks for vision"
    entries = [_make_entry(text1, doc_id="doc-0"),
               _make_entry(text2, doc_id="doc-1")]
    index = SimilarityIndex.build(entries)

    index._embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32,
    )

    ctac_dir = Path(str(tmp_path)) / ".ctac"

    entry_dicts = []
    for e in index.entries:
        doc_id, norm_text, fp, meta, entry_idx, corpus_obj = e
        entry_dicts.append({
            "passage_id": doc_id,
            "doc_id": doc_id,
            "normalized_text": norm_text,
            "original_text": (
                corpus_obj.text
                if hasattr(corpus_obj, "text")
                else norm_text
            ),
            "offset_map": (
                corpus_obj.offset_map
                if hasattr(corpus_obj, "offset_map")
                else []
            ),
            "fingerprint": {
                "doc_id": fp.doc_id,
                "points": [
                    {"hash": p.hash, "position": p.position,
                     "start": p.start, "end": p.end}
                    for p in fp.points
                ],
            },
            "metadata": {
                "title": meta.title if meta else "",
                "authors": meta.authors if meta else [],
                "year": meta.year if meta else None,
                "language": meta.language if meta else "",
                "license": meta.license if meta else "",
            },
            "source_entry_index": entry_idx,
        })

    manifest = IndexManifest(
        corpus_hash=compute_corpus_hash("test"),
        entry_count=len(entry_dicts),
        passage_count=len(entry_dicts),
        embedding_model="test-model",
        embedding_dimension=3,
        embedding_enabled=True,
        embedding_normalized=True,
    )

    save_index(
        ctac_dir, entry_dicts, manifest,
        embeddings=index._embeddings,
        tfidf_vectorizer=index.tfidf_vectorizer,
        tfidf_matrix=index.tfidf_matrix,
    )

    assert (ctac_dir / "manifest.json").exists()
    assert (ctac_dir / "tfidf_matrix.npz").exists()

    loaded_manifest, loaded_entries, loaded_emb, loaded_vec = load_index(ctac_dir)
    assert loaded_manifest.entry_count == 2
    assert loaded_emb is not None
    assert loaded_vec is not None

    loaded_matrix = load_tfidf_matrix(ctac_dir)
    assert loaded_matrix is not None

    assert is_index_valid(ctac_dir, manifest)


def test_corrupt_index_recovery(tmp_path) -> None:
    """Corrupt manifest -> recover returns 'restored' if backup exists."""
    from pathlib import Path

    ctac_dir = Path(str(tmp_path)) / ".ctac"
    backup_dir = Path(str(tmp_path)) / ".ctac.backup"
    ctac_dir.mkdir()

    (ctac_dir / "manifest.json").write_text("invalid json")

    backup_dir.mkdir()
    (backup_dir / "manifest.json").write_text(
        '{"schema_version":"1","entry_count":0,"passage_count":0}'
    )
    (backup_dir / "entries.jsonl").write_text("")

    result = recover_index_state(ctac_dir)
    assert result == "restored"


def test_stale_tmp_cleanup(tmp_path) -> None:
    """Stale .ctac_tmp_* directories are cleaned up on recovery."""
    from pathlib import Path

    stale = Path(str(tmp_path)) / ".ctac_tmp_old"
    stale.mkdir()
    (stale / "junk.txt").write_text("temp")

    ctac_dir = Path(str(tmp_path)) / ".ctac"
    ctac_dir.mkdir()
    (ctac_dir / "manifest.json").write_text(
        '{"schema_version":"1","entry_count":0,"passage_count":0}'
    )
    (ctac_dir / "entries.jsonl").write_text("")

    result = recover_index_state(ctac_dir)
    assert result == "valid"
    assert not stale.exists()


# ---------------------------------------------------------------------------
# Offline model check: zero network
# ---------------------------------------------------------------------------


def test_offline_check_zero_network() -> None:
    """_is_model_cached should not call SentenceTransformer."""
    from unittest.mock import patch

    from citeguard.similarity.embeddings.sentence_transformers import (
        _is_model_cached,
    )

    with patch(
        "citeguard.similarity.embeddings.sentence_transformers"
        ".SentenceTransformer",
        create=True,
    ) as mock_st:
        _is_model_cached("some-model-name")
        mock_st.assert_not_called()


# ---------------------------------------------------------------------------
# Report includes semantic fields
# ---------------------------------------------------------------------------


def test_report_includes_semantic_fields() -> None:
    """similarity_report() surfaces semantic fields."""
    sent = Sentence(
        text="test sentence",
        normalized_text="test sentence",
        paragraph_index=0,
        sentence_index=0,
        start_offset=0,
        end_offset=12,
        citations=[],
    )

    match = SimilarityMatch(
        source_text="test source",
        source_title="Source",
        source_id="src-0",
        exact_overlap=0.1,
        lexical_similarity=0.2,
        combined_score=0.15,
        match_type=MatchType.SEMANTIC_OVERLAP,
        semantic_similarity_raw=0.88,
        semantic_rerank_score=0.88,
        ranking_score=0.264,
    )

    result = SimilarityEngineResult(
        results=[SimilarityResult(
            sentence=sent,
            matches=[match],
            best_match=match,
        )],
        overall_similarity_pct=0.0,
        high_risk_count=0,
        medium_risk_count=0,
        total_sentences=1,
        matched_sentences=1,
        unique_matched_chars=0,
        eligible_chars=100,
    )

    report = similarity_report(result)
    assert "semantic_match_count" in report["summary"]
    assert report["summary"]["semantic_match_count"] == 1

    match_dict = report["results"][0]["matches"][0]
    assert match_dict["semantic_similarity_raw"] == 0.88
    assert match_dict["ranking_score"] == 0.264
    assert match_dict["match_type"] == "semantic_overlap"


# ---------------------------------------------------------------------------
# IndexedPassage integration
# ---------------------------------------------------------------------------


def test_indexed_passage_in_build() -> None:
    """SimilarityIndex.build with passage segmentation produces passages."""
    long_text = "word " * 200
    entry = _make_entry(long_text, doc_id="doc-long")

    idx = SimilarityIndex.build([entry], max_passage_chars=200)
    assert idx.size > 1
    for e in idx.entries:
        obj = e[5]
        assert hasattr(obj, "text") or hasattr(obj, "original_text")


def test_indexed_passage_source_text() -> None:
    """IndexedPassage carries original_text for source span remapping."""
    passage = IndexedPassage(
        passage_id="doc-1#p0",
        parent_entry_id="doc-1",
        original_text="Hello World",
        normalized_text="hello world",
        offset_map=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        offset_in_parent=0,
        fingerprint=Fingerprint(points=[], doc_id="doc-1#p0"),
        metadata=None,
        source_entry_index=0,
        corpus_entry=None,
    )
    assert passage.original_text == "Hello World"
    assert passage.offset_map[0] == 0


# ---------------------------------------------------------------------------
# Dimension mismatch raises error
# ---------------------------------------------------------------------------


def test_retrieve_semantic_dimension_mismatch_raises() -> None:
    """retrieve_semantic raises ValueError on dimension mismatch."""
    entries = [_make_entry("test text")]
    idx = SimilarityIndex.build(entries)
    idx._embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    query = np.array([1.0, 0.0], dtype=np.float32)
    try:
        idx.retrieve_semantic(query, top_k=5)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "dimension" in str(e).lower()


# ---------------------------------------------------------------------------
# ranking_score in engine matches
# ---------------------------------------------------------------------------


def test_engine_populates_ranking_score() -> None:
    """Engine matches have ranking_score set (not default 0.0)."""
    text = "the attention mechanism changed nlp research"
    sentence = _sentence(text)
    entry = _make_entry(text, doc_id="doc-0")

    engine = SimilarityEngine()
    matches = engine.compare_sentence_to_corpus(
        sentence, corpus_entries=[entry],
    )
    assert len(matches) >= 1
    assert matches[0].ranking_score > 0.0


# ---------------------------------------------------------------------------
# CandidateSet structure
# ---------------------------------------------------------------------------


def test_candidate_set_frozen_dataclass() -> None:
    """CandidateSet is immutable and has correct fields."""
    cs = CandidateSet(
        indices=[0, 1, 2],
        semantic_scores={0: 0.9},
        rrf_scores={0: 0.03, 1: 0.02, 2: 0.01},
    )
    assert cs.indices == [0, 1, 2]
    assert cs.semantic_scores[0] == 0.9
    try:
        cs.indices = [3]
        raise AssertionError("Should be frozen")
    except AttributeError:
        pass
