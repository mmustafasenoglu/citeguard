"""Stable models and centralized calibration for plagiarism review."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class AttributionStatus(str, Enum):
    ATTRIBUTED = "attributed"
    QUOTED_AND_ATTRIBUTED = "quoted_and_attributed"
    QUOTED_NOT_ATTRIBUTED = "quoted_not_attributed"
    PARAPHRASE_ATTRIBUTED = "paraphrase_attributed"
    UNATTRIBUTED_EXACT = "unattributed_exact"
    UNATTRIBUTED_NEAR_EXACT = "unattributed_near_exact"
    UNATTRIBUTED_SEMANTIC = "unattributed_semantic"
    CITED_BUT_VERBATIM = "cited_but_verbatim"
    POSSIBLE_COMMON_PHRASE = "possible_common_phrase"
    REVIEW_REQUIRED = "review_required"


class MatchSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class PlagiarismConfig:
    """All runtime thresholds used by the review pipeline."""

    exact_threshold: float = 0.90
    near_exact_threshold: float = 0.70
    lexical_threshold: float = 0.55
    semantic_threshold: float = 0.82
    min_match_words: int = 6
    critical_exact_words: int = 20
    common_phrase_max_words: int = 12
    common_phrase_source_count: int = 3
    max_sources: int = 100
    max_matches: int = 200
    max_candidates: int = 50
    include_quotes: bool = False
    include_bibliography: bool = False
    semantic: bool = False
    discover_academic: bool = False
    offline: bool = True
    no_cache: bool = False
    url_timeout: float = 10.0
    url_max_bytes: int = 5_000_000
    local_source_max_bytes: int = 20_000_000
    url_max_redirects: int = 3
    semantic_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    def __post_init__(self) -> None:
        for name in (
            "exact_threshold",
            "near_exact_threshold",
            "lexical_threshold",
            "semantic_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.min_match_words < 1:
            raise ValueError("min_match_words must be positive")
        for name in (
            "critical_exact_words",
            "common_phrase_max_words",
            "common_phrase_source_count",
            "max_sources",
            "max_matches",
            "max_candidates",
            "url_max_bytes",
            "url_max_redirects",
            "local_source_max_bytes",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.url_timeout <= 0:
            raise ValueError("url_timeout must be positive")


@dataclass(frozen=True, slots=True)
class PlagiarismSource:
    source_id: str
    title: str
    source_type: str
    path: str | None
    url: str | None
    authors: tuple[str, ...]
    year: int | None
    content_hash: str
    normalized_hash: str
    textual_content_compared: bool = True
    source_discovered: bool = False


@dataclass(frozen=True, slots=True)
class PlagiarismMatch:
    match_id: str
    document_start: int
    document_end: int
    document_paragraph: int
    document_paragraph_start: int
    document_paragraph_end: int
    document_text: str
    source_id: str
    source_start: int
    source_end: int
    source_text: str
    match_type: str
    exact_score: float
    lexical_score: float
    semantic_score: float
    combined_score: float
    matched_words: int
    document_words: int
    source_words: int
    quoted: bool
    citation_present: bool
    bibliography_region: bool
    attribution_status: AttributionStatus
    severity: MatchSeverity
    confidence: float
    explanation: str
    recommendation: str


@dataclass(frozen=True, slots=True)
class SourceContribution:
    source_id: str
    title: str
    unique_words: int
    percent: float
    match_count: int


@dataclass(frozen=True, slots=True)
class PlagiarismSummary:
    raw_similarity_percent: float
    review_similarity_percent: float
    exact_similarity_percent: float
    lexical_similarity_percent: float
    semantic_similarity_percent: float
    quoted_similarity_percent: float
    bibliography_similarity_percent: float
    attributed_similarity_percent: float
    unattributed_similarity_percent: float
    eligible_words: int
    total_words: int
    matched_words: int
    review_matched_words: int
    high_risk_matches: int


@dataclass(slots=True)
class PlagiarismResult:
    document: Path
    sources: list[PlagiarismSource]
    matches: list[PlagiarismMatch]
    source_contributions: list[SourceContribution]
    summary: PlagiarismSummary
    configuration: PlagiarismConfig
    warnings: list[str] = field(default_factory=list)
    index: object | None = field(default=None, repr=False)


SCHEMA_VERSION = "1.0"
