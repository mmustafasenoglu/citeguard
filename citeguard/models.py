"""Shared data models, enums, and value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    """Claim severity level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ClaimType(str, Enum):
    """Semantic classification of a citation-worthy claim."""

    STATISTIC = "statistic"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    HISTORICAL = "historical"
    DEFINITION = "definition"
    PRIOR_WORK = "prior_work"
    QUOTATION = "quotation"
    GENERAL_FACT = "general_fact"


class Verdict(str, Enum):
    """Semantic relationship between a claim and a candidate source."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    UNRELATED = "unrelated"
    INSUFFICIENT_INFORMATION = "insufficient_information"


class VerificationStatus(str, Enum):
    """Citation resolution status (metadata-level, not semantic support)."""

    VERIFIED = "verified"
    UNRESOLVED = "unresolved"
    NOT_FOUND = "not_found"
    SUGGESTED = "suggested"


class BibliographyIssueKind(str, Enum):
    """Type of bibliography consistency issue."""

    CITED_NOT_LISTED = "cited_not_listed"
    LISTED_NOT_CITED = "listed_not_cited"
    DUPLICATE_DOI = "duplicate_doi"


class EvidenceType(str, Enum):
    """Source of the evidence passage."""

    ABSTRACT = "abstract"
    FULL_TEXT = "full_text"


@dataclass(slots=True)
class ExistingCitation:
    """A citation detected in the document text."""

    raw_text: str
    authors: str | None
    year: int | None
    doi: str | None
    numbered_ref: int | None
    paragraph_index: int
    char_offset: int
    no_date: bool = False


@dataclass(slots=True)
class Claim:
    """A citation-worthy claim extracted from the document."""

    text: str
    search_query: str
    claim_type: ClaimType
    severity: Severity
    paragraph_index: int
    has_existing_citation: bool
    linked_citation: ExistingCitation | None = None
    linked_citations: list[ExistingCitation] = field(default_factory=list)
    link_confidence: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SourceCandidate:
    """A metadata record returned by an academic provider."""

    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    doi: str | None
    url: str | None
    abstract: str | None
    source_api: str
    arxiv_id: str | None = None


@dataclass(slots=True)
class Evidence:
    """A passage from a source that may support or contradict a claim."""

    text: str
    source_title: str
    source_api: str
    evidence_type: EvidenceType
    section: str | None = None
    page: int | None = None
    lexical_score: int = 0
    semantic_score: int | None = None
    entailment_score: int | None = None
    verdict: Verdict = Verdict.INSUFFICIENT_INFORMATION


@dataclass(slots=True)
class MatchResult:
    """Deterministic matching result between a claim and a source candidate."""

    candidate: SourceCandidate
    source_exists: bool
    metadata_match_score: int
    claim_support_score: int
    overall_confidence: int
    verdict: Verdict
    reasoning: str
    warnings: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    entailment_score: int | None = None
    entailment_verdict: Verdict | None = None


@dataclass(slots=True)
class VerificationResult:
    """Verification outcome for a single claim."""

    claim: Claim
    status: VerificationStatus
    citation: ExistingCitation | None
    matched: MatchResult | None
    suggestions: list[MatchResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BibliographyEntry:
    """A parsed entry from the bibliography section."""

    raw_text: str
    authors: str | None
    year: int | None
    title: str | None
    doi: str | None
    numbered_ref: int | None = None
    no_date: bool = False


@dataclass(slots=True)
class BibliographyIssue:
    """A consistency issue detected in the bibliography."""

    kind: BibliographyIssueKind
    detail: str
    paragraph_index: int | None = None


@dataclass(slots=True)
class ParsedDocument:
    """Complete parsing result for an input document."""

    path: str
    paragraphs: list[str]
    citations: list[ExistingCitation]
    bibliography_entries: list[BibliographyEntry]
    bibliography_start_index: int | None


@dataclass(slots=True)
class CitationContext:
    """A citation linked to its containing sentence and bibliography entries."""

    citation: ExistingCitation
    sentence: str
    bibliography_entry_indexes: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MetadataScores:
    """Deterministic metadata match scores between two records."""

    author: int | None
    year: int | None
    title: int | None
    doi: int | None
    overall: int


@dataclass(slots=True)
class ReferenceVerification:
    """Verification result for a single bibliography entry."""

    entry_index: int
    entry: BibliographyEntry
    query: str
    status: VerificationStatus
    candidate: SourceCandidate | None
    scores: MetadataScores | None
    warnings: list[str] = field(default_factory=list)
