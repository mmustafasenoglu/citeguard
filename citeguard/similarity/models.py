"""Similarity data models for Citeguard v0.3.

Defines fingerprint structures, match types, risk levels, and result containers.
All dataclasses use frozen=True + slots=True for immutability and performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from citeguard.models import Sentence, TextSpan


class MatchType(str, Enum):
    """Classification of how closely two texts match."""

    EXACT = "exact"
    NEAR_DUPLICATE = "near_duplicate"
    LEXICAL_OVERLAP = "lexical_overlap"
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
    """Configuration for similarity computation thresholds."""

    shingle_size: int = 5
    winnow_window: int = 4
    exact_threshold: float = 0.95
    near_duplicate_threshold: float = 0.70
    lexical_overlap_threshold: float = 0.40
    max_results_per_sentence: int = 5
    min_match_length: int = 20
    combined_weight_exact: float = 0.6
    combined_weight_lexical: float = 0.4


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
