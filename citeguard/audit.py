"""Product-level audit orchestration for citeguard.

This module owns the end-to-end audit workflow as a library API independent
from Click/Rich, so tests and future consumers can run audits directly:

    document parsing
        -> claim extraction (LLM hybrid, or deterministic when offline)
        -> claim <-> citation linking (already on Claim models)
        -> bibliography consistency
        -> bibliography verification (resolution signal)
        -> cited-claim support assessment (support signal, kept separate)
        -> uncited-claim source suggestions (suggestion signal, thresholded)
        -> optional similarity analysis
        -> diagnostics + execution context
        -> audit metrics + priority review queue
        -> one structured AuditResult

Product invariant: citation presence, bibliographic resolution, claim
support, source suggestion, textual similarity, and semantic similarity are
distinct signals and are never collapsed into one ambiguous "verified" or
"confidence" concept.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .bibliography import (
    bibliography_issues,
    citation_matches_entry,
    resolve_numbered_citations,
)
from .cache import FileCache
from .claims import extract_claims
from .config import (
    DEFAULT_MAX_RESULTS,
    DEFAULT_THRESHOLD,
    HEALTH_PASS_THRESHOLD,
    WEAK_MATCH_THRESHOLD,
    Settings,
)
from .extractor import parse_document
from .matcher import match_claim_to_source
from .models import (
    BibliographyIssue,
    Claim,
    ExistingCitation,
    MatchResult,
    ParsedDocument,
    ReferenceVerification,
    Severity,
    SourceCandidate,
    Verdict,
    VerificationStatus,
)
from .providers.arxiv import ArxivProvider
from .providers.crossref import CrossrefProvider
from .providers.openalex import OpenAlexProvider
from .providers.semantic_scholar import SemanticScholarProvider
from .retrieval import RetrievalEngine, RetrievalResult
from .scoring import (
    SEVERITY_WEIGHT,
    ProductMetrics,
    compute_product_metrics,
    overall_confidence,
)
from .verification import verify_bibliography

# Suggested (never verified) vs verified statuses stay distinct.
_EVALUABLE_BIB_STATUSES = frozenset(
    {
        VerificationStatus.VERIFIED,
        VerificationStatus.PARTIALLY_VERIFIED,
    }
)

CITED_SUMMARY_SUPPORTED = "supported"
CITED_SUMMARY_CONTRADICTED = "contradicted"
CITED_SUMMARY_MIXED = "mixed"
CITED_SUMMARY_UNRESOLVED = "unresolved"
CITED_SUMMARY_UNAVAILABLE = "unavailable"


@dataclass(slots=True)
class AuditOptions:
    """Product-level run configuration (no CLI objects)."""

    threshold: int = DEFAULT_THRESHOLD
    max_results: int = DEFAULT_MAX_RESULTS
    max_claims: int | None = None
    severity: str | None = None
    offline: bool = False
    no_cache: bool = False
    cache_dir: Path | None = None
    recency_max_age: int = 25
    require_evidence: bool = False
    show_similarity: bool = False
    corpus: Path | None = None
    corpus_license: str | None = None
    enable_semantic: bool = False
    bib_providers: Sequence[Any] | None = None
    search_providers: Sequence[Any] | None = None
    on_progress: Callable[[int, int, str], None] | None = None


@dataclass(slots=True)
class ProviderDiagnostic:
    """One structured provider-phase observation (never a parsed string)."""

    provider: str
    phase: str
    kind: str
    message: str
    query: str | None = None
    has_usable_result: bool = False


@dataclass(slots=True)
class ExecutionContext:
    """What the audit actually did (network, cache, LLM, semantic)."""

    offline: bool
    network_allowed: bool
    network_used: bool
    academic_network_used: bool = False
    llm_network_used: bool = False
    providers_queried: list[str] = field(default_factory=list)
    providers_from_cache: list[str] = field(default_factory=list)
    llm_used: bool = False
    llm_tasks: list[str] = field(default_factory=list)
    semantic_backend_used: str | None = None


@dataclass(slots=True)
class CitedSourceAssessment:
    """Support assessment of one cited claim against one cited source."""

    citation_raw: str
    entry_index: int | None
    bib_status: VerificationStatus | None
    candidate: SourceCandidate | None
    evaluated: bool
    reason: str
    verdict: Verdict | None = None
    metadata_score: int | None = None
    support_score: int | None = None
    confidence: int | None = None
    evidence: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class ClaimAssessment:
    """Claim-level rollup over per-source assessments (sources preserved)."""

    claim: Claim
    sources: list[CitedSourceAssessment] = field(default_factory=list)
    summary: str = CITED_SUMMARY_UNAVAILABLE


@dataclass(slots=True)
class SuggestionAssessment:
    """Suggestion outcome for one uncited claim (never a verification)."""

    claim: Claim
    status: VerificationStatus = VerificationStatus.NOT_FOUND
    matched: MatchResult | None = None
    suggestions: list[MatchResult] = field(default_factory=list)


@dataclass(slots=True)
class AuditResult:
    """One structured product-level audit outcome."""

    document: str
    claims: list[Claim]
    bibliography_verification: list[ReferenceVerification]
    claim_assessments: list[ClaimAssessment]
    suggestions: list[SuggestionAssessment]
    bibliography_issues: list[BibliographyIssue]
    similarity: Any | None
    metrics: ProductMetrics
    review_queue: list[dict[str, Any]]
    diagnostics: list[ProviderDiagnostic]
    execution: ExecutionContext
    provider_phase_failed: bool = False


class _PhaseEngine(RetrievalEngine):
    """RetrievalEngine that records every retrieval for diagnostics."""

    def __init__(
        self,
        *args: Any,
        phase: str = "general",
        record: list[tuple[str, str, RetrievalResult]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.phase = phase
        self.record = record if record is not None else []

    def search_with_result(
        self, query: str, max_results: int = DEFAULT_MAX_RESULTS
    ) -> RetrievalResult:
        result = super().search_with_result(query, max_results=max_results)
        self.record.append((self.phase, query, result))
        return result


def _diagnostics_from_record(
    record: list[tuple[str, str, RetrievalResult]],
) -> list[ProviderDiagnostic]:
    diagnostics: list[ProviderDiagnostic] = []
    for phase, query, result in record:
        for error in result.provider_errors:
            name, _, message = error.partition(": ")
            diagnostics.append(
                ProviderDiagnostic(
                    provider=name or "unknown",
                    phase=phase,
                    kind="provider_error",
                    message=message or error,
                    query=query,
                    has_usable_result=bool(result.candidates),
                )
            )
        for name in result.skipped_providers:
            diagnostics.append(
                ProviderDiagnostic(
                    provider=name,
                    phase=phase,
                    kind="skipped_offline",
                    message=(
                        "Provider not queried: offline mode and no cached "
                        "result is available."
                    ),
                    query=query,
                    has_usable_result=False,
                )
            )
    return diagnostics


def select_suggestions(
    matches: list[MatchResult], *, threshold: int
) -> tuple[VerificationStatus, MatchResult | None, list[MatchResult]]:
    """Apply strict threshold semantics shared by suggest/check/API.

    A candidate below ``threshold`` must not produce SUGGESTED, even when it
    is the best retrieval candidate.
    """
    filtered = [m for m in matches if m.overall_confidence >= threshold]
    if not filtered:
        return VerificationStatus.NOT_FOUND, None, []
    return VerificationStatus.SUGGESTED, filtered[0], filtered


def match_candidates(
    claim: Claim,
    candidates: Sequence[SourceCandidate],
    *,
    offline: bool = False,
) -> list[MatchResult]:
    """Evaluate every candidate with the shared evidence pipeline."""
    results: list[MatchResult] = []
    for candidate in candidates:
        metadata_score, support_score, verdict, reasoning, evidence = (
            match_claim_to_source(claim, candidate, offline=offline)
        )
        has_entailment = any(e.entailment_score is not None for e in evidence)
        confidence = overall_confidence(
            metadata_score, support_score, has_entailment=has_entailment
        )
        ent_score = evidence[0].entailment_score if evidence else None
        ent_verdict = (
            evidence[0].verdict if evidence and ent_score is not None else None
        )
        results.append(
            MatchResult(
                candidate=candidate,
                source_exists=True,
                metadata_match_score=metadata_score,
                claim_support_score=support_score,
                overall_confidence=confidence,
                verdict=verdict,
                reasoning=reasoning,
                evidence=evidence,
                entailment_score=ent_score,
                entailment_verdict=ent_verdict,
            )
        )
    return results


def _extract_claims_product(
    parsed: ParsedDocument, options: AuditOptions
) -> tuple[list[Claim], bool]:
    """Extract claims; deterministic-only when offline.

    Returns ``(claims, llm_claims_used)``.
    """
    from .llm import extract_claims_with_llm

    claims: list[Claim] = []
    llm_claims_used = False
    limit = options.max_claims if options.max_claims is not None else float("inf")
    for index, paragraph in enumerate(parsed.paragraphs):
        if len(claims) >= limit:
            break
        if (
            parsed.bibliography_start_index is not None
            and index >= parsed.bibliography_start_index
        ):
            continue
        paragraph_citations = [
            c for c in parsed.citations if c.paragraph_index == index
        ]
        llm_claims: list[Claim] | None = None
        if not options.offline:
            llm_claims = extract_claims_with_llm(
                paragraph, index, paragraph_citations
            )
        if llm_claims:
            llm_claims_used = True
            claims.extend(llm_claims)
        else:
            claims.extend(
                extract_claims(
                    [paragraph],
                    parsed.citations,
                    bibliography_start=parsed.bibliography_start_index,
                    max_claims=None,
                )
            )
    if options.max_claims is not None and len(claims) > options.max_claims:
        claims = claims[: options.max_claims]
    if options.severity:
        claims = [c for c in claims if c.severity.value == options.severity]
    return claims, llm_claims_used


def _claim_citations(claim: Claim) -> list[ExistingCitation]:
    seen: set[tuple[str, int, int]] = set()
    ordered: list[ExistingCitation] = []
    linked: list[ExistingCitation | None] = [claim.linked_citation]
    linked.extend(claim.linked_citations)
    for citation in linked:
        if citation is None:
            continue
        key = (citation.raw_text, citation.paragraph_index, citation.char_offset)
        if key not in seen:
            seen.add(key)
            ordered.append(citation)
    return ordered


def assess_cited_claims(
    claims: list[Claim],
    parsed: ParsedDocument,
    bib_results: list[ReferenceVerification],
    *,
    offline: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[ClaimAssessment]:
    """Assess every cited claim against its actually cited sources.

    No citation association is invented: only citations linked by the
    claim extractor/linker are followed to bibliography entries, and only
    sufficiently resolved candidates (VERIFIED/PARTIALLY_VERIFIED) are
    evaluated for support.  Exact DOI identity stays valid even when
    provider_conflicts carries an uncertainty warning.
    """
    results_by_entry = {r.entry_index: r for r in bib_results}
    numbered = resolve_numbered_citations(
        parsed.citations, parsed.bibliography_entries
    )
    entry_index_by_id = {id(e): i for i, e in enumerate(parsed.bibliography_entries)}
    assessments: list[ClaimAssessment] = []
    cited = [c for c in claims if c.has_existing_citation]
    for done, claim in enumerate(cited, 1):
        if on_progress is not None:
            on_progress(done, len(cited), f"Assessing: {claim.text[:50]}...")
        sources: list[CitedSourceAssessment] = []
        for citation in _claim_citations(claim):
            entry_indexes: list[int] = []
            if (
                citation.numbered_ref is not None
                and numbered.get(citation.numbered_ref) is not None
            ):
                entry = numbered[citation.numbered_ref]
                assert entry is not None
                entry_indexes.append(entry_index_by_id[id(entry)])
            else:
                for i, entry in enumerate(parsed.bibliography_entries):
                    if citation_matches_entry(citation, entry):
                        entry_indexes.append(i)
            if not entry_indexes:
                sources.append(
                    CitedSourceAssessment(
                        citation_raw=citation.raw_text,
                        entry_index=None,
                        bib_status=None,
                        candidate=None,
                        evaluated=False,
                        reason="Citation has no matching bibliography entry.",
                    )
                )
                continue
            for entry_index in entry_indexes:
                ref = results_by_entry.get(entry_index)
                status = ref.status if ref is not None else None
                candidate = ref.candidate if ref is not None else None
                if (
                    ref is not None
                    and status in _EVALUABLE_BIB_STATUSES
                    and candidate is not None
                ):
                    metadata_score, support_score, verdict, _, evidence = (
                        match_claim_to_source(claim, candidate, offline=offline)
                    )
                    has_entailment = any(
                        e.entailment_score is not None for e in evidence
                    )
                    confidence = overall_confidence(
                        metadata_score, support_score, has_entailment=has_entailment
                    )
                    reason = (
                        "Evaluated against the resolved cited source."
                        if status == VerificationStatus.VERIFIED
                        else (
                            "Evaluated against a partially verified source; "
                            "metadata uncertainty is preserved."
                        )
                    )
                    sources.append(
                        CitedSourceAssessment(
                            citation_raw=citation.raw_text,
                            entry_index=entry_index,
                            bib_status=status,
                            candidate=candidate,
                            evaluated=True,
                            reason=reason,
                            verdict=verdict,
                            metadata_score=metadata_score,
                            support_score=support_score,
                            confidence=confidence,
                            evidence=evidence,
                        )
                    )
                elif status == VerificationStatus.METADATA_MISMATCH:
                    sources.append(
                        CitedSourceAssessment(
                            citation_raw=citation.raw_text,
                            entry_index=entry_index,
                            bib_status=status,
                            candidate=candidate,
                            evaluated=False,
                            reason=(
                                "Bibliography metadata does not match the "
                                "provider candidate; support is not presented "
                                "as support from the cited work."
                            ),
                        )
                    )
                else:
                    label = status.value if status is not None else "missing"
                    sources.append(
                        CitedSourceAssessment(
                            citation_raw=citation.raw_text,
                            entry_index=entry_index,
                            bib_status=status,
                            candidate=candidate,
                            evaluated=False,
                            reason=(
                                f"Cited source could not be resolved "
                                f"(status: {label}); support unavailable."
                            ),
                        )
                    )
        assessments.append(
            ClaimAssessment(
                claim=claim, sources=sources, summary=_claim_summary(sources)
            )
        )
    return assessments


def _claim_summary(sources: list[CitedSourceAssessment]) -> str:
    evaluated = [s for s in sources if s.evaluated]
    if not evaluated:
        return CITED_SUMMARY_UNAVAILABLE
    contradicted = any(s.verdict == Verdict.CONTRADICTED for s in evaluated)
    supported = any(
        s.verdict in (Verdict.SUPPORTED, Verdict.PARTIALLY_SUPPORTED)
        for s in evaluated
    )
    if contradicted and supported:
        return CITED_SUMMARY_MIXED
    if contradicted:
        return CITED_SUMMARY_CONTRADICTED
    if supported:
        return CITED_SUMMARY_SUPPORTED
    return CITED_SUMMARY_UNRESOLVED


def _default_bib_providers() -> list[Any]:
    return [CrossrefProvider(), OpenAlexProvider()]


def _default_search_providers() -> list[Any]:
    return [
        SemanticScholarProvider(),
        CrossrefProvider(),
        OpenAlexProvider(),
        ArxivProvider(),
    ]


def _make_engines(
    options: AuditOptions,
    record: list[tuple[str, str, RetrievalResult]],
    cache: FileCache | None,
) -> tuple[_PhaseEngine, _PhaseEngine]:
    bib_providers = (
        list(options.bib_providers)
        if options.bib_providers is not None
        else _default_bib_providers()
    )
    search_providers = (
        list(options.search_providers)
        if options.search_providers is not None
        else _default_search_providers()
    )
    use_cache = not options.no_cache
    bib_engine = _PhaseEngine(
        bib_providers,
        cache=cache,
        use_cache=use_cache,
        phase="bibliography",
        record=record,
        offline=options.offline,
    )
    search_engine = _PhaseEngine(
        search_providers,
        cache=cache,
        use_cache=use_cache,
        phase="suggestion",
        record=record,
        offline=options.offline,
    )
    return bib_engine, search_engine


def audit_document(
    source: Path | ParsedDocument,
    options: AuditOptions | None = None,
    *,
    offline: bool | None = None,
) -> AuditResult:
    """Run the full product audit and return one structured AuditResult."""
    from .llm import get_audit_log

    options = options or AuditOptions()
    if offline is not None:
        options = replace(options, offline=offline)
    llm_log_before = len(get_audit_log())
    record: list[tuple[str, str, RetrievalResult]] = []

    if isinstance(source, ParsedDocument):
        parsed = source
        document_label = parsed.path
    else:
        parsed = parse_document(Path(source))
        document_label = str(source)
    if not parsed.paragraphs:
        raise ValueError("The document is empty or contains only whitespace.")

    settings = Settings.from_env()
    cache = FileCache(options.cache_dir or settings.cache_dir)
    bib_engine, search_engine = _make_engines(options, record, cache)

    claims, _llm_claims_used = _extract_claims_product(parsed, options)

    bib_retrievals: list[tuple[int, str, RetrievalResult]] = []
    bib_results = verify_bibliography(
        parsed.bibliography_entries,
        bib_engine,
        max_results=options.max_results,
        recency_max_age=options.recency_max_age,
        on_retrieval=lambda i, q, r: bib_retrievals.append((i, q, r)),
    )

    bib_issues = bibliography_issues(parsed.citations, parsed.bibliography_entries)

    assessments = assess_cited_claims(
        claims,
        parsed,
        bib_results,
        offline=options.offline,
        on_progress=options.on_progress,
    )

    suggestions: list[SuggestionAssessment] = []
    uncited = [c for c in claims if not c.has_existing_citation]
    total_uncited = len(uncited)
    for done, claim in enumerate(uncited, 1):
        if options.on_progress is not None:
            options.on_progress(
                done, total_uncited, f"Searching: {claim.text[:50]}..."
            )
        candidates = search_engine.search(
            claim.search_query, max_results=options.max_results
        )
        matches = match_candidates(claim, candidates, offline=options.offline)
        status, matched, filtered = select_suggestions(
            matches, threshold=options.threshold
        )
        suggestions.append(
            SuggestionAssessment(
                claim=claim, status=status, matched=matched, suggestions=filtered
            )
        )

    sim_result: Any | None = None
    semantic_backend: str | None = None
    if options.show_similarity:
        sim_result, semantic_backend = _run_similarity(source, options, record)

    diagnostics = _diagnostics_from_record(record)
    execution = _build_execution(options, record, [bib_engine, search_engine],
                                 get_audit_log()[llm_log_before:])
    execution.semantic_backend_used = semantic_backend
    metrics = _build_metrics(claims, parsed, bib_results, bib_retrievals,
                             assessments, suggestions, bib_issues)
    review_queue = build_review_queue(
        assessments, suggestions, bib_results, bib_issues, sim_result
    )
    provider_phase_failed = _phase_failed(options, record)

    return AuditResult(
        document=document_label,
        claims=claims,
        bibliography_verification=bib_results,
        claim_assessments=assessments,
        suggestions=suggestions,
        bibliography_issues=bib_issues,
        similarity=sim_result,
        metrics=metrics,
        review_queue=review_queue,
        diagnostics=diagnostics,
        execution=execution,
        provider_phase_failed=provider_phase_failed,
    )


def suggest_document(
    source: Path | ParsedDocument,
    options: AuditOptions | None = None,
    *,
    offline: bool | None = None,
) -> AuditResult:
    """Run claim extraction plus uncited suggestions (no bib verification)."""
    from .llm import get_audit_log

    options = options or AuditOptions()
    if offline is not None:
        options = replace(options, offline=offline)
    llm_log_before = len(get_audit_log())
    record: list[tuple[str, str, RetrievalResult]] = []

    if isinstance(source, ParsedDocument):
        parsed = source
        document_label = parsed.path
    else:
        parsed = parse_document(Path(source))
        document_label = str(source)
    if not parsed.paragraphs:
        raise ValueError("The document is empty or contains only whitespace.")

    settings = Settings.from_env()
    cache = FileCache(options.cache_dir or settings.cache_dir)
    _, search_engine = _make_engines(options, record, cache)

    claims, _llm_claims_used = _extract_claims_product(parsed, options)

    suggestions: list[SuggestionAssessment] = []
    uncited = [c for c in claims if not c.has_existing_citation]
    total_uncited = len(uncited)
    for done, claim in enumerate(uncited, 1):
        if options.on_progress is not None:
            options.on_progress(
                done, total_uncited, f"Searching: {claim.text[:50]}..."
            )
        candidates = search_engine.search(
            claim.search_query, max_results=options.max_results
        )
        matches = match_candidates(claim, candidates, offline=options.offline)
        status, matched, filtered = select_suggestions(
            matches, threshold=options.threshold
        )
        suggestions.append(
            SuggestionAssessment(
                claim=claim, status=status, matched=matched, suggestions=filtered
            )
        )

    diagnostics = _diagnostics_from_record(record)
    execution = _build_execution(options, record, [search_engine],
                                 get_audit_log()[llm_log_before:])
    metrics = _build_metrics(claims, parsed, [], [], [], suggestions, [])
    review_queue = build_review_queue([], suggestions, [], [], None)

    return AuditResult(
        document=document_label,
        claims=claims,
        bibliography_verification=[],
        claim_assessments=[],
        suggestions=suggestions,
        bibliography_issues=[],
        similarity=None,
        metrics=metrics,
        review_queue=review_queue,
        diagnostics=diagnostics,
        execution=execution,
        provider_phase_failed=_phase_failed(options, record),
    )


def verify_document(
    source: Path | ParsedDocument,
    options: AuditOptions | None = None,
    *,
    offline: bool | None = None,
) -> AuditResult:
    """Run bibliography verification only (no claim extraction)."""
    from .llm import get_audit_log

    options = options or AuditOptions()
    if offline is not None:
        options = replace(options, offline=offline)
    llm_log_before = len(get_audit_log())
    record: list[tuple[str, str, RetrievalResult]] = []

    if isinstance(source, ParsedDocument):
        parsed = source
        document_label = parsed.path
    else:
        parsed = parse_document(Path(source))
        document_label = str(source)
    if not parsed.paragraphs:
        raise ValueError("The document is empty or contains only whitespace.")

    settings = Settings.from_env()
    cache = FileCache(options.cache_dir or settings.cache_dir)
    bib_engine, _ = _make_engines(options, record, cache)

    bib_retrievals: list[tuple[int, str, RetrievalResult]] = []
    bib_results = verify_bibliography(
        parsed.bibliography_entries,
        bib_engine,
        max_results=options.max_results,
        recency_max_age=options.recency_max_age,
        on_retrieval=lambda i, q, r: bib_retrievals.append((i, q, r)),
    )
    bib_issues = bibliography_issues(parsed.citations, parsed.bibliography_entries)

    diagnostics = _diagnostics_from_record(record)
    execution = _build_execution(options, record, [bib_engine],
                                 get_audit_log()[llm_log_before:])
    metrics = _build_metrics([], parsed, bib_results, bib_retrievals, [], [], bib_issues)
    review_queue = build_review_queue([], [], bib_results, bib_issues, None)

    return AuditResult(
        document=document_label,
        claims=[],
        bibliography_verification=bib_results,
        claim_assessments=[],
        suggestions=[],
        bibliography_issues=bib_issues,
        similarity=None,
        metrics=metrics,
        review_queue=review_queue,
        diagnostics=diagnostics,
        execution=execution,
        provider_phase_failed=_phase_failed(options, record),
    )


def _run_similarity(
    source: Path | ParsedDocument,
    options: AuditOptions,
    record: list[tuple[str, str, RetrievalResult]],
) -> tuple[Any | None, str | None]:
    """Run local similarity; never download models when offline."""
    from .extractor import parse_enriched_document
    from .similarity.engine import SimilarityEngine
    from .similarity.models import SimilarityConfig

    if isinstance(source, ParsedDocument):
        return None, None
    if options.enable_semantic and options.offline:
        record.append((
            "similarity",
            "semantic-backend",
            RetrievalResult(
                [], [], [], offline=True,
                skipped_providers=["semantic-embeddings"],
            ),
        ))
        return None, None
    try:
        enriched = parse_enriched_document(Path(source))
    except Exception:
        return None, None
    from .corpus.loader import load_similarity_corpus

    index = load_similarity_corpus(options.corpus, options.corpus_license)
    engine = SimilarityEngine(config=SimilarityConfig())
    result = engine.analyze_document(
        enriched.sentences,
        index=index,
        bibliography_entries=enriched.bibliography_entries,
    )
    return result, "tfidf-local"


def _build_execution(
    options: AuditOptions,
    record: list[tuple[str, str, RetrievalResult]],
    engines: list[RetrievalEngine],
    new_llm_entries: list[Any],
) -> ExecutionContext:
    queried: set[str] = set()
    from_cache: set[str] = set()
    for _, _, result in record:
        queried.update(result.queried_providers)
        from_cache.update(result.cache_hits)
    remote: set[str] = set()
    for engine in engines:
        remote.update(engine.remote_calls)
    tasks = sorted({e.task for e in new_llm_entries if hasattr(e, "task")})
    academic_remote = bool(remote)
    llm_remote = any(
        hasattr(e, "cached") and not e.cached for e in new_llm_entries
    )
    return ExecutionContext(
        offline=options.offline,
        network_allowed=not options.offline,
        network_used=academic_remote or llm_remote,
        academic_network_used=academic_remote,
        llm_network_used=llm_remote,
        providers_queried=sorted(queried),
        providers_from_cache=sorted(from_cache),
        llm_used=bool(new_llm_entries),
        llm_tasks=tasks,
        semantic_backend_used=None,
    )


def _entry_evaluated(
    retrievals: list[tuple[int, str, RetrievalResult]],
) -> bool:
    """An entry counts as evaluated unless every retrieval was skipped or errored."""
    if not retrievals:
        return False
    for _, _, result in retrievals:
        if result.candidates:
            return True
        if not result.provider_errors and not result.skipped_providers:
            return True
    return False


def _build_metrics(
    claims: list[Claim],
    parsed: ParsedDocument,
    bib_results: list[ReferenceVerification],
    bib_retrievals: list[tuple[int, str, RetrievalResult]],
    assessments: list[ClaimAssessment],
    suggestions: list[SuggestionAssessment],
    bib_issues: list[BibliographyIssue],
) -> ProductMetrics:
    total_claims = len(claims)
    cited_claims = sum(1 for c in claims if c.has_existing_citation)
    verified = sum(
        1 for r in bib_results if r.status == VerificationStatus.VERIFIED
    )
    partially = sum(
        1 for r in bib_results if r.status == VerificationStatus.PARTIALLY_VERIFIED
    )

    retrievals_by_entry: dict[int, list[tuple[int, str, RetrievalResult]]] = {}
    for index, query, result in bib_retrievals:
        retrievals_by_entry.setdefault(index, []).append((index, query, result))
    bib_evaluated = sum(
        1
        for i in range(len(parsed.bibliography_entries))
        if _entry_evaluated(retrievals_by_entry.get(i, []))
    )

    supported_cited = 0
    cited_evaluated = 0
    cited_with_evidence = 0
    contradicted = 0
    weak_cited = 0
    unresolved_high = 0
    for assessment in assessments:
        evaluated_sources = [s for s in assessment.sources if s.evaluated]
        if evaluated_sources:
            cited_evaluated += 1
            if any(s.evidence for s in evaluated_sources):
                cited_with_evidence += 1
        if assessment.summary in (
            CITED_SUMMARY_SUPPORTED,
            CITED_SUMMARY_MIXED,
            CITED_SUMMARY_CONTRADICTED,
        ) and any(
            s.verdict in (Verdict.SUPPORTED, Verdict.PARTIALLY_SUPPORTED)
            for s in evaluated_sources
        ):
            supported_cited += 1
        contradicted += sum(
            1 for s in evaluated_sources if s.verdict == Verdict.CONTRADICTED
        )
        weak_cited += sum(
            1
            for s in evaluated_sources
            if s.confidence is not None and s.confidence < WEAK_MATCH_THRESHOLD
        )
        if (
            assessment.claim.severity == Severity.HIGH
            and assessment.summary
            not in (CITED_SUMMARY_SUPPORTED, CITED_SUMMARY_MIXED)
        ):
            unresolved_high += 1

    weak_sugg = sum(
        1
        for s in suggestions
        for m in s.suggestions
        if m.overall_confidence < WEAK_MATCH_THRESHOLD
    )
    uncited_high = sum(
        1
        for c in claims
        if not c.has_existing_citation and c.severity == Severity.HIGH
    )
    distinct_citations = len(
        {c.raw_text.strip() for c in parsed.citations if c.raw_text.strip()}
    )

    return compute_product_metrics(
        total_claims=total_claims,
        cited_claims=cited_claims,
        verified_citations=verified,
        partially_verified=partially,
        bib_evaluated=bib_evaluated,
        bib_total=len(parsed.bibliography_entries),
        supported_cited_claims=supported_cited,
        contradicted_assessments=contradicted,
        cited_evaluated=cited_evaluated,
        cited_with_evidence=cited_with_evidence,
        weak_cited_source_matches=weak_cited,
        weak_suggestions=weak_sugg,
        uncited_high=uncited_high,
        unresolved_high=unresolved_high,
        bibliography_issues=len(bib_issues),
        bibliography_entries=len(parsed.bibliography_entries),
        distinct_citations=distinct_citations,
    )


def _phase_failed(
    options: AuditOptions,
    record: list[tuple[str, str, RetrievalResult]],
) -> bool:
    """Total provider-phase failure: online, attempted, nothing usable, errors."""
    if options.offline or not record:
        return False
    any_candidates = any(bool(result.candidates) for _, _, result in record)
    any_errors = any(bool(result.provider_errors) for _, _, result in record)
    return (not any_candidates) and any_errors


def build_review_queue(
    assessments: list[ClaimAssessment],
    suggestions: list[SuggestionAssessment],
    bib_results: list[ReferenceVerification],
    bib_issues: list[BibliographyIssue],
    sim_result: Any | None,
) -> list[dict[str, Any]]:
    """Build the deterministic product-level review queue.

    Priority principles (documented order):
    1. contradiction on a cited claim
    2. unresolved/mismatched cited claim
    3. uncited high-severity claim
    4. medium-severity citation/support problems
    5. low-severity findings
    6. informational warnings (metadata conflicts, bib issues, similarity)
    """
    items: list[dict[str, Any]] = []
    seq = 0

    def claim_item(
        kind: str,
        group: int,
        claim: Claim,
        status: str,
        verdict: Verdict | None,
        confidence: int | None,
        matched: dict[str, Any] | None,
        detail: str,
        suggestions_count: int = 0,
    ) -> dict[str, Any]:
        nonlocal seq
        seq += 1
        return {
            "kind": kind,
            "group": group,
            "seq": seq,
            "claim_text": claim.text,
            "claim_type": claim.claim_type.value,
            "severity": claim.severity.value,
            "paragraph": claim.paragraph_index + 1,
            "status": status,
            "verdict": verdict.value if verdict is not None else None,
            "confidence": confidence,
            "matched": matched,
            "suggestions_count": suggestions_count,
            "warnings": [],
            "detail": detail,
        }

    for assessment in assessments:
        claim = assessment.claim
        evaluated = [s for s in assessment.sources if s.evaluated]
        best = max(evaluated, key=lambda s: s.confidence or 0) if evaluated else None
        matched = _support_match_item(best) if best is not None else None
        confidence = best.confidence if best is not None else None
        verdict = best.verdict if best is not None else None
        if assessment.summary in (CITED_SUMMARY_CONTRADICTED, CITED_SUMMARY_MIXED):
            group = 0
            kind = "cited_contradiction"
            status = "contradicted"
            detail = "A resolved cited source contradicts the claim."
        elif assessment.summary == CITED_SUMMARY_SUPPORTED:
            continue
        elif assessment.summary == CITED_SUMMARY_UNRESOLVED:
            high_or_medium = claim.severity in (Severity.HIGH, Severity.MEDIUM)
            group = 1 if high_or_medium else 4
            kind = "cited_unresolved"
            status = "unresolved"
            detail = "Cited sources were evaluated but do not support the claim."
        else:
            high_or_medium = claim.severity in (Severity.HIGH, Severity.MEDIUM)
            group = 1 if high_or_medium else 4
            kind = "cited_unresolved"
            status = "unavailable"
            detail = "Cited sources could not be evaluated for support."
        items.append(
            claim_item(kind, group, claim, status, verdict, confidence, matched, detail)
        )

    for suggestion in suggestions:
        claim = suggestion.claim
        matched = _suggestion_match_item(suggestion.matched)
        confidence = (
            suggestion.matched.overall_confidence if suggestion.matched else None
        )
        verdict = suggestion.matched.verdict if suggestion.matched else None
        if suggestion.status == VerificationStatus.SUGGESTED:
            if claim.severity == Severity.HIGH:
                group = 2
            elif claim.severity == Severity.MEDIUM:
                group = 3
            else:
                group = 4
            kind = "uncited_with_suggestion"
            detail = "Uncited claim has candidate sources needing review."
        else:
            if claim.severity == Severity.HIGH:
                group = 2
            elif claim.severity == Severity.MEDIUM:
                group = 3
            else:
                group = 4
            kind = "uncited_claim"
            detail = "Uncited claim has no adequate source suggestion."
        items.append(
            claim_item(
                kind, group, claim, suggestion.status.value, verdict, confidence,
                matched, detail, suggestions_count=len(suggestion.suggestions),
            )
        )

    for ref in bib_results:
        if ref.provider_conflicts:
            items.append({
                "kind": "metadata_conflict",
                "group": 5,
                "seq": _next_seq(items),
                "claim_text": ref.entry.raw_text,
                "claim_type": "bibliography",
                "severity": "low",
                "paragraph": None,
                "status": ref.status.value,
                "verdict": None,
                "confidence": ref.scores.overall if ref.scores else None,
                "matched": None,
                "suggestions_count": 0,
                "warnings": [],
                "detail": "; ".join(ref.provider_conflicts),
            })

    for issue in bib_issues:
        items.append({
            "kind": "bibliography_issue",
            "group": 5,
            "seq": _next_seq(items),
            "claim_text": issue.detail,
            "claim_type": "bibliography",
            "severity": "low",
            "paragraph": (
                issue.paragraph_index + 1
                if issue.paragraph_index is not None
                else None
            ),
            "status": issue.kind.value,
            "verdict": None,
            "confidence": None,
            "matched": None,
            "suggestions_count": 0,
            "warnings": [],
            "detail": issue.detail,
        })

    if sim_result is not None:
        for row in sim_result.results:
            risk = getattr(row, "attribution_risk", None)
            level = risk.value if risk is not None else "low"
            if level not in ("high", "medium"):
                continue
            best_match = getattr(row, "best_match", None)
            items.append({
                "kind": "similarity_risk",
                "group": 5,
                "seq": _next_seq(items),
                "claim_text": row.sentence.text,
                "claim_type": "similarity",
                "severity": level,
                "paragraph": row.sentence.paragraph_index + 1,
                "status": f"similarity_{level}_risk",
                "verdict": None,
                "confidence": None,
                "matched": {
                    "title": getattr(best_match, "source_title", ""),
                    "authors": [],
                    "year": None,
                    "doi": None,
                    "overall_confidence": None,
                    "verdict": None,
                    "evidence": [],
                } if best_match is not None else None,
                "suggestions_count": 0,
                "warnings": [],
                "detail": (
                    "Textual overlap with corpus text; review attribution. "
                    "Similarity is not a plagiarism verdict."
                ),
            })

    items.sort(key=_queue_sort_key)
    return items


def _queue_sort_key(item: dict[str, Any]) -> tuple[int, int, int, int]:
    """Deterministic review ordering: group, severity, risk, insertion order.

    Within problem groups lower confidence sorts first (higher risk first);
    informational items keep insertion order.
    """
    severity_rank = -SEVERITY_WEIGHT[Severity(item["severity"])]
    if item["group"] < 5:
        confidence = item["confidence"]
        risk_rank = confidence if confidence is not None else -1
        return (item["group"], severity_rank, risk_rank, item["seq"])
    return (item["group"], severity_rank, item["seq"], item["seq"])


def _next_seq(items: list[dict[str, Any]]) -> int:
    return len(items) + 1


def _evidence_item(evidence: Any) -> dict[str, Any]:
    return {
        "text": evidence.text,
        "source_title": evidence.source_title,
        "source_api": evidence.source_api,
        "type": evidence.evidence_type.value,
        "section": evidence.section,
        "page": evidence.page,
        "relevance_score": evidence.lexical_score,
        "semantic_score": evidence.semantic_score,
        "entailment_score": evidence.entailment_score,
        "verdict": evidence.verdict.value,
    }


def _support_match_item(source: CitedSourceAssessment) -> dict[str, Any] | None:
    candidate = source.candidate
    if candidate is None:
        return None
    return {
        "title": candidate.title,
        "authors": candidate.authors,
        "year": candidate.year,
        "doi": candidate.doi,
        "overall_confidence": source.confidence,
        "verdict": source.verdict.value if source.verdict is not None else None,
        "evidence": [_evidence_item(e) for e in source.evidence],
    }


def _suggestion_match_item(match: MatchResult | None) -> dict[str, Any] | None:
    if match is None:
        return None
    return {
        "title": match.candidate.title,
        "authors": match.candidate.authors,
        "year": match.candidate.year,
        "doi": match.candidate.doi,
        "overall_confidence": match.overall_confidence,
        "verdict": match.verdict.value,
        "evidence": [_evidence_item(e) for e in match.evidence],
    }


def privacy_notice(execution: ExecutionContext) -> str:
    """Describe actual data handling based on the recorded execution."""
    if execution.offline:
        base = "Offline mode: no network requests were permitted."
        if execution.providers_from_cache:
            names = ", ".join(execution.providers_from_cache)
            base += (
                f" Bibliographic metadata was resolved from local cache "
                f"({names}); document prose was not sent anywhere."
            )
        else:
            base += (
                " Remote metadata lookup was skipped; document prose was "
                "not sent anywhere."
            )
    else:
        if execution.providers_queried:
            names = ", ".join(execution.providers_queried)
            if execution.academic_network_used:
                base = (
                    f"Bibliographic metadata queries were sent to {names}."
                )
            else:
                base = (
                    f"Bibliographic metadata was resolved from local cache "
                    f"for {names}; no network requests were needed."
                )
            base += " Document prose was not sent to academic providers."
        else:
            base = "No academic provider was queried."
    if execution.llm_used:
        tasks = ", ".join(execution.llm_tasks) or "language-model tasks"
        if execution.llm_network_used:
            base += (
                f" Claim text was sent to the configured LLM provider ({tasks})."
            )
        elif execution.offline or not execution.network_allowed:
            base += (
                f" LLM tasks ({tasks}) were skipped (offline mode)."
            )
        else:
            base += (
                f" LLM results ({tasks}) were served from cache; no LLM "
                "provider was contacted."
            )
    else:
        base += " No document text was sent to any LLM provider."
    base += " Verification does not establish claim support."
    return base


def resolve_exit_code(result: AuditResult, *, command: str) -> int:
    """Map an audit outcome to CLI exit codes.

    Interim policy (Checkpoint 8 owns calibration):
    - required provider phase failed online -> 3
    - check with incomplete health or health below the pass threshold -> 1
    - suggest/verify -> 0 on completion (2/3 handled elsewhere)
    """
    if result.provider_phase_failed:
        return 3
    if command == "check":
        if (
            result.metrics is None
            or
            not result.metrics.health_score_complete
            or result.metrics.health_score is None
            or result.metrics.health_score < HEALTH_PASS_THRESHOLD
        ):
            return 1
        return 0
    return 0
