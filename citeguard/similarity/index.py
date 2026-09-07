"""Corpus index for similarity retrieval.

Holds corpus entries, fingerprints, and TF-IDF state in a single
object that the engine can consume.  Semantic fields are placeholders
for a future commit; they are present but not wired up yet.

``SimilarityIndex.build()`` creates an index from the same
``CorpusEntryTuple`` list the engine already uses, so this commit
is a pure refactoring — no algorithmic changes, no score changes.
"""

from __future__ import annotations

from citeguard.similarity.fingerprint import exact_overlap
from citeguard.similarity.lexical import (
    build_tfidf_index,
    compute_cosine_similarity,
)
from citeguard.similarity.models import CorpusEntryTuple, Fingerprint


class SimilarityIndex:
    """Pre-built corpus index for similarity retrieval.

    Created via ``SimilarityIndex.build(corpus_entries)``.

    Properties are read-only.  Retrieval methods expose ranked candidate
    lists but the engine does **not** use ``retrieve_candidates()`` to
    prune the corpus in v0.4.0 — every entry is still evaluated.
    Candidate pruning will be introduced with the semantic / RRF work.
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
        """Placeholder for dense vector embeddings (not yet implemented)."""
        return self._embeddings

    @property
    def embedding_model(self) -> str | None:
        """Placeholder for the model name used to compute *embeddings*."""
        return self._embedding_model

    @property
    def embedding_dim(self) -> int | None:
        """Placeholder for the embedding dimensionality."""
        return self._embedding_dim

    @property
    def size(self) -> int:
        """Number of corpus entries in the index."""
        return len(self._entries)

    # -- Construction --------------------------------------------------------

    @classmethod
    def build(cls, corpus_entries: list[CorpusEntryTuple]) -> SimilarityIndex:
        """Build an index from pre-built corpus entry tuples.

        Constructs the fingerprints list and TF-IDF index from the entries.
        Semantic embeddings are **not** built (placeholder only).
        """
        if not corpus_entries:
            return cls()

        fingerprints = [entry[2] for entry in corpus_entries]
        corpus_texts = [entry[1] for entry in corpus_entries]
        vectorizer, matrix = build_tfidf_index(corpus_texts)

        return cls(
            entries=list(corpus_entries),
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

    def retrieve_candidates(
        self, query_text: str, query_fp: Fingerprint, top_k: int = 50,
    ) -> list[int]:
        """Union of fingerprint + TF-IDF retrieval, deduped, top *top_k*.

        .. note::

           In v0.4.0 this method is exposed for future use and testing.
           The engine does **NOT** use it to prune the corpus yet — it
           still evaluates every entry.  Candidate pruning will be
           introduced with the semantic / RRF implementation.
        """
        fp_hits = self.retrieve_fingerprint(query_fp, top_k=top_k)
        tfidf_hits = self.retrieve_tfidf(query_text, top_k=top_k)

        seen: set[int] = set()
        candidates: list[int] = []
        for idx, _score in fp_hits + tfidf_hits:
            if idx not in seen:
                seen.add(idx)
                candidates.append(idx)
        return candidates[:top_k]
