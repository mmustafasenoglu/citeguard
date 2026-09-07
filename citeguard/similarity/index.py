"""Corpus index for similarity retrieval.

Holds corpus entries, fingerprints, TF-IDF state, and optional
semantic embeddings in a single object that the engine can consume.

``SimilarityIndex.build()`` supports optional passage segmentation:
when ``max_passage_chars`` is provided, each corpus entry is split
into passages, normalized with offset mapping, and fingerprinted
individually.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from citeguard.similarity.fingerprint import exact_overlap
from citeguard.similarity.lexical import (
    build_tfidf_index,
    compute_cosine_similarity,
)
from citeguard.similarity.models import (
    CandidateSet,
    CorpusEntryTuple,
    Fingerprint,
    IndexedPassage,
)

logger = logging.getLogger(__name__)


def _segment_passages(
    text: str,
    max_chars: int,
) -> list[str]:
    """Split *text* into passages of at most *max_chars* characters.

    Splits on paragraph boundaries first, then sentence boundaries,
    then hard-breaks at *max_chars* as a last resort.
    """
    if len(text) <= max_chars:
        return [text]

    passages: list[str] = []
    # Split on double newline (paragraphs)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                passages.append(current)
            if len(para) > max_chars:
                # Hard split at max_chars
                for i in range(0, len(para), max_chars):
                    passages.append(para[i:i + max_chars])
                current = ""
            else:
                current = para
    if current:
        passages.append(current)
    return passages if passages else [text]


class SimilarityIndex:
    """Pre-built corpus index for similarity retrieval.

    Created via ``SimilarityIndex.build(corpus_entries)``.
    """

    __slots__ = (
        "_entries",
        "_fingerprints",
        "_tfidf_vectorizer",
        "_tfidf_matrix",
        "_embeddings",
        "_embedding_model",
        "_embedding_dim",
    )

    def __init__(
        self,
        entries: list[CorpusEntryTuple] | None = None,
        fingerprints: list[Fingerprint] | None = None,
        tfidf_vectorizer: object | None = None,
        tfidf_matrix: object | None = None,
        embeddings: object | None = None,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        self._entries: list[CorpusEntryTuple] = entries or []
        self._fingerprints: list[Fingerprint] = fingerprints or []
        self._tfidf_vectorizer = tfidf_vectorizer
        self._tfidf_matrix = tfidf_matrix
        self._embeddings = embeddings
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim

    # -- Read-only properties ------------------------------------------------

    @property
    def entries(self) -> list[CorpusEntryTuple]:
        """Corpus entry tuples held by this index."""
        return self._entries

    @property
    def fingerprints(self) -> list[Fingerprint]:
        """Fingerprint for each corpus entry (parallel to *entries*)."""
        return self._fingerprints

    @property
    def tfidf_vectorizer(self) -> object | None:
        """Fitted TfidfVectorizer, or *None* when the index is empty."""
        return self._tfidf_vectorizer

    @property
    def tfidf_matrix(self) -> object | None:
        """Sparse TF-IDF matrix, or *None* when the index is empty."""
        return self._tfidf_matrix

    @property
    def embeddings(self) -> object | None:
        """Dense vector embeddings matrix, or *None*."""
        return self._embeddings

    @property
    def embedding_model(self) -> str | None:
        """Model name used to compute *embeddings*."""
        return self._embedding_model

    @property
    def embedding_dim(self) -> int | None:
        """Embedding dimensionality."""
        return self._embedding_dim

    @property
    def size(self) -> int:
        """Number of corpus entries in the index."""
        return len(self._entries)

    @property
    def has_embeddings(self) -> bool:
        """True when embeddings are loaded and non-empty."""
        return (
            self._embeddings is not None
            and hasattr(self._embeddings, "shape")
            and self._embeddings.shape[0] > 0
        )

    # -- Construction --------------------------------------------------------

    @classmethod
    def build(
        cls,
        corpus_entries: list[CorpusEntryTuple],
        max_passage_chars: int = 0,
    ) -> SimilarityIndex:
        """Build an index from pre-built corpus entry tuples.

        Parameters
        ----------
        corpus_entries:
            List of (doc_id, text, Fingerprint, metadata, entry_index, obj).
        max_passage_chars:
            When > 0, segment each entry into passages of at most this
            many characters before fingerprinting.  Each passage
            normalizes independently and carries its own offset_map.
        """
        if not corpus_entries:
            return cls()

        from citeguard.corpus.models import CorpusMetadata
        from citeguard.similarity.fingerprint import generate_shingles, winnow
        from citeguard.similarity.normalize import normalize_with_map

        entries_to_index: list[CorpusEntryTuple] = []

        for (
            doc_id, _norm_text, _fp, metadata, entry_idx, corpus_entry_obj
        ) in corpus_entries:
            # Get original text from the corpus entry object
            original_text = ""
            if corpus_entry_obj is not None:
                if hasattr(corpus_entry_obj, "text"):
                    original_text = corpus_entry_obj.text
                elif isinstance(corpus_entry_obj, str):
                    original_text = corpus_entry_obj

            if not original_text:
                # Fallback: use normalized text directly
                original_text = _norm_text

            if max_passage_chars > 0 and len(original_text) > max_passage_chars:
                # Passage segmentation
                passages = _segment_passages(original_text, max_passage_chars)
                for p_idx, passage_text in enumerate(passages):
                    norm_text, offset_map = normalize_with_map(passage_text)
                    fp = Fingerprint(
                        points=winnow(
                            generate_shingles(norm_text, k=5), window=4
                        ),
                        doc_id=f"{doc_id}#p{p_idx}",
                    )
                    # Build passage-level metadata (shallow copy)
                    passage_meta = metadata
                    if isinstance(metadata, CorpusMetadata):
                        passage_meta = metadata

                    entries_to_index.append((
                        f"{doc_id}#p{p_idx}",
                        norm_text,
                        fp,
                        passage_meta,
                        entry_idx,
                        IndexedPassage(
                            passage_id=f"{doc_id}#p{p_idx}",
                            parent_entry_id=doc_id,
                            original_text=passage_text,
                            normalized_text=norm_text,
                            offset_map=offset_map,
                            offset_in_parent=_find_offset(
                                original_text, passage_text
                            ),
                            fingerprint=fp,
                            metadata=passage_meta,
                            source_entry_index=entry_idx,
                            corpus_entry=corpus_entry_obj,
                        ),
                    ))
            else:
                # Whole entry (no segmentation)
                entries_to_index.append((
                    doc_id, _norm_text, _fp, metadata, entry_idx,
                    corpus_entry_obj,
                ))

        # Build fingerprints list
        fingerprints = [entry[2] for entry in entries_to_index]

        # Build TF-IDF
        corpus_texts = [entry[1] for entry in entries_to_index]
        vectorizer, matrix = build_tfidf_index(corpus_texts)

        return cls(
            entries=entries_to_index,
            fingerprints=fingerprints,
            tfidf_vectorizer=vectorizer,
            tfidf_matrix=matrix,
        )

    # -- Retrieval -----------------------------------------------------------

    def retrieve_tfidf(
        self, query_text: str, top_k: int = 50,
    ) -> list[tuple[int, float]]:
        """Return ``(entry_index, score)`` sorted descending by TF-IDF cosine."""
        if self._tfidf_vectorizer is None or self._tfidf_matrix is None:
            return []
        hits = compute_cosine_similarity(
            query_text, self._tfidf_vectorizer, self._tfidf_matrix,
        )
        return hits[:top_k]

    def retrieve_fingerprint(
        self, query_fp: Fingerprint, top_k: int = 50,
    ) -> list[tuple[int, float]]:
        """Return ``(entry_index, overlap_score)`` sorted descending by fingerprint overlap."""
        scores: list[tuple[int, float]] = []
        for i, fp in enumerate(self._fingerprints):
            eo = exact_overlap(query_fp, fp)
            if eo > 0:
                scores.append((i, eo))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def retrieve_semantic(
        self, query_embedding: Any, top_k: int = 50,
    ) -> list[tuple[int, float]]:
        """Return ``(entry_index, raw_cosine)`` sorted descending.

        L2-normalizes the query, then computes dot product (cosine)
        against stored embeddings.  Raises ``ValueError`` on dimension
        mismatch instead of silently returning empty.
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).ravel()
        if query.shape[0] != self._embeddings.shape[1]:
            raise ValueError(
                f"Query dimension {query.shape[0]} != "
                f"embedding dimension {self._embeddings.shape[1]}"
            )

        # L2-normalize query to guarantee cosine = dot product
        norm = np.linalg.norm(query)
        if norm < 1e-8:
            raise ValueError("Query embedding has near-zero norm")
        query = query / norm

        # Corpus embeddings are L2-normalized at build time (by backend)
        scores = self._embeddings @ query
        indexed = list(enumerate(scores.tolist()))
        indexed.sort(key=lambda x: (-x[1], x[0]))
        return indexed[:top_k]

    def retrieve_candidates(
        self,
        query_text: str,
        query_fp: Fingerprint,
        query_embedding: Any | None = None,
        top_k: int = 50,
    ) -> CandidateSet:
        """RRF-based candidate retrieval combining fingerprint + TF-IDF + semantic.

        Returns a ``CandidateSet`` carrying ranked indices and per-entry
        semantic scores so the engine does not recompute dot products.
        """
        from citeguard.similarity.fusion import reciprocal_rank_fusion

        fp_hits = self.retrieve_fingerprint(query_fp, top_k=top_k)
        tfidf_hits = self.retrieve_tfidf(query_text, top_k=top_k)
        semantic_hits: list[tuple[int, float]] = []
        if query_embedding is not None and self.has_embeddings:
            semantic_hits = self.retrieve_semantic(query_embedding, top_k=top_k)

        ranked_lists = [
            [idx for idx, _ in fp_hits],
            [idx for idx, _ in tfidf_hits],
        ]
        if semantic_hits:
            ranked_lists.append([idx for idx, _ in semantic_hits])

        fused = reciprocal_rank_fusion(ranked_lists, k=60)

        semantic_map = dict(semantic_hits)
        rrf_map = dict(fused)

        return CandidateSet(
            indices=[idx for idx, _score in fused[:top_k]],
            semantic_scores=semantic_map,
            rrf_scores=rrf_map,
        )

    def encode_query(self, text: str) -> Any | None:
        """Encode a query text using the loaded embeddings.

        Returns None when no embeddings are available (fallback to
        lexical-only path).
        """
        if not self.has_embeddings:
            return None
        # The query encoding requires the backend, which lives outside the index.
        # For now, return None — the engine will encode externally.
        return None


def _find_offset(haystack: str, needle: str) -> int:
    """Find the character offset of *needle* within *haystack*."""
    idx = haystack.find(needle)
    return idx if idx >= 0 else 0
