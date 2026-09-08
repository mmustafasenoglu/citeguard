"""Models for attribution-risk reduction and candidate validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from ..models import Verdict
from ..similarity.models import RiskLevel


class ReductionRiskType(str, Enum):
    """Reason a passage may need attribution improvement."""

    EXACT_COPY = "exact_copy"
    TOO_CLOSE_PARAPHRASE = "too_close_paraphrase"
    MISSING_CITATION = "missing_citation"
    QUOTE_NEEDED = "quote_needed"
    ATTRIBUTION_MISMATCH = "attribution_mismatch"
    COMMON_PHRASE = "common_phrase"
    TECHNICAL_TERM_OVERLAP = "technical_term_overlap"
    ACCEPTABLE_OVERLAP = "acceptable_overlap"
    UNKNOWN = "unknown"


class FixAction(str, Enum):
    """Conservative action selected by the reduction planner."""

    LEAVE = "leave"
    ADD_CITATION = "add_citation"
    ADD_QUOTATION = "add_quotation"
    PARAPHRASE = "paraphrase"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True, slots=True)
class PassageRisk:
    """Similarity and attribution evidence for one document sentence."""

    passage_id: str
    text: str
    paragraph_index: int
    start_offset: int
    end_offset: int
    exact_overlap: float
    lexical_similarity: float
    semantic_similarity_raw: float | None
    attribution_risk: RiskLevel
    has_citation: bool
    citation_verified: bool | None
    citation_support: Verdict | None
    citation_texts: tuple[str, ...]
    risk_type: ReductionRiskType
    recommended_action: FixAction
    confidence: float
    source_title: str | None = None
    source_id: str | None = None
    attribution_reason: str = ""


@dataclass(frozen=True, slots=True)
class FixPlan:
    """A reviewable plan for one risky passage."""

    passage_id: str
    action: FixAction
    reason: str
    preserve_citations: tuple[str, ...] = ()
    must_preserve_numbers: tuple[str, ...] = ()
    rewrite_allowed: bool = False


@dataclass(slots=True)
class RewriteCandidate:
    """A proposed replacement and its validation measurements."""

    text: str
    generator: str
    citations_preserved: bool | None = None
    meaning_score: float | None = None
    lexical_overlap: float | None = None
    exact_overlap: float | None = None
    semantic_similarity_to_original: float | None = None
    source_support_score: float | None = None
    factual_integrity: bool | None = None
    numeric_integrity: bool | None = None
    introduced_claims: list[str] = field(default_factory=list)
    verdict: Verdict | None = None
    rejection_reasons: list[str] = field(default_factory=list)
    score: float | None = None


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Hard-gate result for a rewrite candidate."""

    accepted: bool
    citations_preserved: bool
    numeric_integrity: bool
    factual_integrity: bool
    unsupported_new_claims: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RewriteRequest:
    """Backend-neutral request for candidate generation."""

    original_text: str
    plan: FixPlan
    candidate_count: int


class RewriteBackend(Protocol):
    """Provider-neutral rewrite backend contract."""

    def generate(self, request: RewriteRequest) -> list[RewriteCandidate]:
        """Generate reviewable candidates without applying them."""
