"""Domain models for optional, evidence-grounded rewrite suggestions.

Rewrite suggestions are generated text only.  They never constitute
citation verification — verification results remain authoritative.

``rewrite != verification``, ``rewrite != evidence``,
``rewrite != citation repair``.  This layer only proposes text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class RewriteMode(str, Enum):
    """Explicit rewrite goal selected by the user."""

    CLARIFY = "clarify"
    HEDGE = "hedge"
    ALIGN_WITH_EVIDENCE = "align_with_evidence"
    REMOVE_UNSUPPORTED_DETAIL = "remove_unsupported_detail"
    CITATION_SAFE = "citation_safe"


class RewriteStatus(str, Enum):
    """Deterministic outcome of a rewrite request."""

    SUCCESS = "success"
    UNAVAILABLE = "unavailable"
    REFUSED = "refused"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PROVIDER_ERROR = "provider_error"
    INVALID_RESPONSE = "invalid_response"


@dataclass(frozen=True, slots=True)
class RewriteContext:
    """Minimum verified context sent to a rewrite provider.

    Built exclusively from CiteGuard verification output: the claim text,
    its existing citation, the verification verdict, the matched source
    metadata, and evidence passages.  Never the full document, never the
    full bibliography, never unrelated claims.
    """

    original_text: str
    claim_text: str
    claim_type: str
    citation_raw: str | None
    verification_status: str
    verdict: str
    source_title: str | None = None
    source_authors: tuple[str, ...] = ()
    source_year: int | None = None
    source_doi: str | None = None
    evidence_texts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RewriteRequest:
    """A single evidence-grounded rewrite request."""

    context: RewriteContext
    mode: RewriteMode = RewriteMode.CLARIFY
    style_constraints: tuple[str, ...] = ()
    max_context_chars: int = 2000


@dataclass(frozen=True, slots=True)
class RewriteResult:
    """Outcome of a rewrite request (suggestion only, never a verdict)."""

    original_text: str
    rewritten_text: str | None
    status: RewriteStatus
    provider: str
    model: str
    warnings: tuple[str, ...] = ()
    evidence_used: tuple[str, ...] = ()
    citation_preserved: bool | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class RewriteProvider(Protocol):
    """Provider-neutral rewrite backend contract.

    Implementations must never silently substitute a local/template
    fallback for a failed remote call.  Any fallback requires explicit
    configuration and must be declared in ``RewriteResult.metadata``.
    """

    def rewrite(self, request: RewriteRequest) -> RewriteResult:
        """Generate a rewrite suggestion for *request*."""
