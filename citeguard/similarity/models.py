"""Similarity data models for Citeguard v0.3.

Defines fingerprint structures, match types, risk levels, result containers,
and the ``CorpusEntryTuple`` type alias shared by the engine and index.
All dataclasses use frozen=True + slots=True for immutability and performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from citeguard.models import Sentence, TextSpan

if TYPE_CHECKING:
    from citeguard.corpus.models import CorpusMetadata


class MatchType(str, Enum):
    """Classification of how closely two texts match."""

    EXACT = "exact"
    NEAR_DUPLICATE = "near_duplicate"
    LEXICAL_OVERLAP = "lexical_overlap"
    SEMANTIC_OVERLAP = "semantic_overlap"
    UNMATCHED = "unmatched"


class RiskLevel(str, Enum):
    """Attribution risk level for a given match."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class Shingle:
    """An ordered k-gram with positional information."""

    hash: int
    start: int
    end: int
    position: int


@dataclass(frozen=True, slots=True)
class FingerprintPoint:
    """A selected shingle from the winnowing process."""

    hash: int
    position: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """Min-hash fingerprint representing a document's content."""

    points: list[FingerprintPoint]
    doc_id: str | None = None


@dataclass(slots=True)
class SimilarityMatch:
    """A single source-to-document match with similarity scores.

    Ordering is intentional: required fields before optional ones
    to avoid Python dataclass field-order errors.
    """

    source_text: str
    source_title: str | None
    source_id: str | None

    exact_overlap: float
    lexical_similarity: float
    combined_score: float

    source_url: str | None = None
    source_authors: list[str] = field(default_factory=list)
    source_year: int | None = None
    source_doi: str | None = None
    matched_source_spans: list[TextSpan] = field(default_factory=list)
    matched_document_spans: list[TextSpan] = field(default_factory=list)
    match_type: MatchType = MatchType.UNMATCHED
    semantic_similarity_raw: float = 0.0
    semantic_rerank_score: float = 0.0


@dataclass(slots=True)
class SimilarityResult:
    """Result of comparing one sentence against the corpus."""

    sentence: Sentence
    matches: list[SimilarityMatch] = field(default_factory=list)
    best_match: SimilarityMatch | None = None
    attribution_risk: RiskLevel = RiskLevel.NONE
    attribution_reason: str = ""


@dataclass(slots=True)
class SimilarityEngineResult:
    """Aggregated results for a full-document similarity analysis."""

    results: list[SimilarityResult]
    overall_similarity_pct: float
    high_risk_count: int
    medium_risk_count: int
    total_sentences: int
    matched_sentences: int
    unique_matched_chars: int
    eligible_chars: int


@dataclass(frozen=True, slots=True)
class SimilarityConfig:
    """Configuration for similarity computation thresholds.

    Fields are divided into two categories:

    **Artifact-producing** (changing these invalidates a persisted index):
    ``shingle_size``, ``winnow_window``, ``ngram_range``,
    ``max_passage_chars``, ``embedding_model``, ``embedding_dimension``.

    **Runtime** (changing these does NOT invalidate a persisted index):
    ``exact_threshold``, ``near_duplicate_threshold``,
    ``lexical_overlap_threshold``, ``max_results_per_sentence``,
    ``combined_weight_exact``, ``combined_weight_lexical``,
    ``quotation_coverage_threshold``, ``enable_semantic``,
    ``semantic_threshold``, ``allow_model_download``,
    ``rrf_k``, ``weight_fingerprint``, ``weight_tfidf``,
    ``weight_semantic``.
    """

    shingle_size: int = 5
    winnow_window: int = 4
    ngram_range: tuple[int, int] = (1, 3)
    exact_threshold: float = 0.95
    near_duplicate_threshold: float = 0.70
    lexical_overlap_threshold: float = 0.40
    max_results_per_sentence: int = 5
    min_match_length: int = 20
    combined_weight_exact: float = 0.6
    combined_weight_lexical: float = 0.4
    quotation_coverage_threshold: float = 0.95

    # Semantic configuration (runtime)
    enable_semantic: bool = False
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384
    allow_model_download: bool = True
    semantic_threshold: float | None = None  # uncalibrated until benchmark

    # Passage segmentation (artifact-producing)
    max_passage_chars: int = 500

    # RRF / reranker (runtime)
    rrf_k: int = 60
    weight_fingerprint: float = 0.3
    weight_tfidf: float = 0.3
    weight_semantic: float = 0.4


@dataclass(frozen=True, slots=True)
class SimilarityMetrics:
    """Post-hoc metrics snapshot for a similarity run."""

    overall_pct: float
    high_risk_count: int
    medium_risk_count: int
    matched_sentences: int
    total_sentences: int
    avg_overlap: float
    unique_matched_chars: int
    eligible_chars: int


# ---------------------------------------------------------------------------
# Corpus entry tuple type alias
# ---------------------------------------------------------------------------

# Lives here (neutral module) to avoid circular imports between engine.py
# and index.py.  Both import from this module.
#
# Fields:
#   0: doc_id        – str
#   1: normalized_text – str (lowercased / normalized)
#   2: Fingerprint
#   3: CorpusMetadata | None
#   4: entry_index   – int (position within original document)
#   5: CorpusEntry   – object (raw corpus entry for offset_map etc.)
CorpusEntryTuple = tuple[str, str, Fingerprint, "CorpusMetadata | None", int, object]


# ---------------------------------------------------------------------------
# Indexed passage (Checkpoint 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndexedPassage:
    """A passage within or as a whole corpus entry, with full provenance.

    Created by ``SimilarityIndex.build()`` when passage segmentation is
    enabled (``max_passage_chars``).  Each passage carries its own
    ``offset_map`` so that normalised spans can be remapped to original
    passage coordinates, and ``offset_in_parent`` connects back to the
    parent entry's original text.

    Segmentation order (strict):

    1. Parent ORIGINAL text
    2. Sentence / passage segmentation
    3. Passage ORIGINAL text
    4. ``normalize_corpus_text_with_map()``
    5. ``normalized_text`` + ``offset_map``
    6. Fingerprint from ``normalized_text``
    """

    passage_id: str              # "doc-1#p0"
    parent_entry_id: str         # "doc-1"

    original_text: str           # orijinal passage (kesilmemiş)
    normalized_text: str         # normalize edilmiş

    # normalized passage index → original passage index
    offset_map: list[int]

    # original passage start inside parent source
    offset_in_parent: int

    fingerprint: Fingerprint
    metadata: object             # CorpusMetadata | None
    source_entry_index: int      # parent'ın entry_index'i
    corpus_entry: object         # orijinal CorpusEntry referansı
