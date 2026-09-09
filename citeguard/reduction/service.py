"""End-to-end attribution-risk reduction service."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..audit import AuditOptions, AuditResult, audit_document
from ..models import Verdict
from ..rewrite import EvidenceGroundedReductionBackend, LLMRewriteProvider, RewriteMode
from ..rewrite.models import RewriteProvider
from .analyzer import analyze_passage_risks
from .apply import TextReplacement, write_revised_text
from .candidates import generate_candidates
from .evaluator import evaluate_candidate
from .grounding import build_grounded_reduction_requests
from .meaning import (
    MeaningThresholds,
    SentenceTransformerSemanticBackend,
    TransformerNLIBackend,
    validate_meaning,
)
from .metrics import ReductionMetrics, compute_reduction_metrics
from .models import (
    EntailmentBackend,
    FixAction,
    FixPlan,
    PassageRisk,
    RewriteCandidate,
    SemanticBackend,
)
from .models import (
    RewriteRequest as ReductionRewriteRequest,
)
from .planner import build_fix_plans
from .ranker import rank_candidates
from .source_evaluator import evaluate_source_overlap
from .validator import validate_candidate

if TYPE_CHECKING:
    from ..similarity.index import SimilarityIndex
    from ..similarity.models import SimilarityEngineResult

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class ReductionOptions:
    """Programmatic configuration for bounded attribution improvement."""

    corpus: Path | None = None
    corpus_license: str | None = None
    apply: bool = False
    output: Path | None = None
    candidate_count: int = 3
    max_iterations: int = 2
    offline: bool = False
    provider: str | None = None
    model: str | None = None
    severity: str | None = None
    semantic_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    allow_model_download: bool = False
    similarity_index: SimilarityIndex | None = None
    rewrite_provider: RewriteProvider | None = None
    semantic_backend: SemanticBackend | None = None
    entailment_backend: EntailmentBackend | None = None
    audit_runner: Callable[[Path, AuditOptions], AuditResult] | None = None
    meaning_thresholds: MeaningThresholds = field(
        default_factory=lambda: MeaningThresholds(0.70, 0.70)
    )


@dataclass(slots=True)
class ReductionChange:
    """Structured candidate and application audit record."""

    passage_id: str
    paragraph_index: int
    risk_type: str
    original_text: str
    matched_source_text: str | None
    source_id: str | None
    action: str
    provider: str | None = None
    model: str | None = None
    candidate_count: int = 0
    selected_candidate: str | None = None
    source_overlap_before: float | None = None
    source_overlap_after: float | None = None
    meaning_score: float | None = None
    forward_entailment_score: float | None = None
    backward_entailment_score: float | None = None
    meaning_verdict: str | None = None
    citation_preserved: bool | None = None
    numeric_integrity: bool | None = None
    named_entity_integrity: bool | None = None
    factual_integrity: bool | None = None
    unsupported_new_claims: tuple[str, ...] = ()
    application_status: str = "planned"
    start_offset: int | None = None
    end_offset: int | None = None
    affected_runs: tuple[int, ...] = ()
    formatting_preserved: bool | None = None
    rejection_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(slots=True)
class ReductionResult:
    """Complete result including scans, plans, changes, and execution facts."""

    source: Path
    output: Path | None
    dry_run: bool
    applied: bool
    iterations: int
    stop_reason: str
    before: SimilarityEngineResult
    after: SimilarityEngineResult
    risks: list[PassageRisk]
    plans: list[FixPlan]
    changes: list[ReductionChange]
    metrics: ReductionMetrics
    execution: dict[str, Any]
    manual_review: dict[str, str]


def _scan(path: Path, index: SimilarityIndex) -> tuple[Any, SimilarityEngineResult]:
    from ..extractor import parse_enriched_document
    from ..similarity.engine import SimilarityEngine
    from ..similarity.models import SimilarityConfig

    enriched = parse_enriched_document(path)
    result = SimilarityEngine(config=SimilarityConfig()).analyze_document(
        sentences=enriched.sentences,
        index=index,
        bibliography_entries=enriched.bibliography_entries,
    )
    return enriched, result


def _source_texts(result: SimilarityEngineResult) -> dict[str, str]:
    texts: dict[str, str] = {}
    for item in result.results:
        if item.best_match is None or item.sentence.is_bibliography:
            continue
        passage_id = f"p{item.sentence.paragraph_index}s{item.sentence.sentence_index}"
        texts[passage_id] = item.best_match.source_text
    return texts


def _new_assertions(original: str, candidate: str) -> list[str]:
    """Identify obvious appended assertions for source-support checking."""
    original_sentences = _SENTENCE_SPLIT_RE.split(original.strip())
    candidate_sentences = _SENTENCE_SPLIT_RE.split(candidate.strip())
    additions = candidate_sentences[len(original_sentences) :]
    original_and = len(re.findall(r"\band\b", original, re.I))
    candidate_parts = re.split(r"\band\b", candidate, flags=re.I)
    if len(candidate_parts) - 1 > original_and:
        additions.extend(candidate_parts[original_and + 1 :])
    return list(dict.fromkeys(part.strip(" .") for part in additions if part.strip(" .")))


def _unsupported_assertions(
    original: str,
    candidate: str,
    evidence: tuple[str, ...],
    entailment_backend: EntailmentBackend,
) -> list[str]:
    unsupported: list[str] = []
    for assertion in _new_assertions(original, candidate):
        if not evidence:
            unsupported.append(assertion)
            continue
        supported = False
        for text in evidence:
            verdict = entailment_backend.evaluate(text, assertion).verdict
            if verdict in {Verdict.SUPPORTED, Verdict.PARTIALLY_SUPPORTED} or (
                verdict.value == "meaning_preserved"
            ):
                supported = True
                break
        if not supported:
            unsupported.append(assertion)
    return unsupported


def _audit(
    path: Path,
    options: ReductionOptions,
) -> AuditResult:
    audit_options = AuditOptions(
        offline=options.offline,
        no_cache=False,
        bib_providers=None,
        search_providers=None,
    )
    if options.audit_runner is not None:
        return options.audit_runner(path, audit_options)
    return audit_document(path, audit_options)


def _parse_generator(generator: str) -> tuple[str, str]:
    provider, separator, model = generator.partition(":")
    return provider, model if separator else "unknown"


def _change(
    risk: PassageRisk,
    plan: FixPlan,
    candidate: RewriteCandidate | None,
    *,
    source_text: str | None,
    candidate_count: int,
    status: str,
    warnings: tuple[str, ...] = (),
) -> ReductionChange:
    provider = model = None
    if candidate is not None:
        provider, model = _parse_generator(candidate.generator)
    return ReductionChange(
        passage_id=risk.passage_id,
        paragraph_index=risk.paragraph_index,
        risk_type=risk.risk_type.value,
        original_text=risk.text,
        matched_source_text=source_text,
        source_id=risk.source_id,
        action=plan.action.value,
        provider=provider,
        model=model,
        candidate_count=candidate_count,
        selected_candidate=candidate.text if candidate else None,
        source_overlap_before=(
            max(
                candidate.source_exact_overlap_before or 0.0,
                candidate.source_lexical_similarity_before or 0.0,
            )
            if candidate
            else None
        ),
        source_overlap_after=(
            max(
                candidate.source_exact_overlap_after or 0.0,
                candidate.source_lexical_similarity_after or 0.0,
            )
            if candidate
            else None
        ),
        meaning_score=candidate.meaning_score if candidate else None,
        forward_entailment_score=(candidate.forward_entailment_score if candidate else None),
        backward_entailment_score=(candidate.backward_entailment_score if candidate else None),
        meaning_verdict=(
            candidate.meaning_verdict.value if candidate and candidate.meaning_verdict else None
        ),
        citation_preserved=candidate.citations_preserved if candidate else None,
        numeric_integrity=candidate.numeric_integrity if candidate else None,
        named_entity_integrity=(candidate.named_entity_integrity if candidate else None),
        factual_integrity=candidate.factual_integrity if candidate else None,
        unsupported_new_claims=(tuple(candidate.introduced_claims) if candidate else ()),
        application_status=status,
        rejection_reasons=(tuple(candidate.rejection_reasons) if candidate else ()),
        warnings=warnings,
    )


def _validate_options(source: Path, options: ReductionOptions) -> None:
    if not 1 <= options.candidate_count <= 10:
        raise ValueError("candidate_count must be between 1 and 10")
    if not 1 <= options.max_iterations <= 10:
        raise ValueError("max_iterations must be between 1 and 10")
    if options.apply and options.output is None:
        raise ValueError("automatic application requires an output path")
    if options.output is not None and source.resolve() == options.output.resolve():
        raise ValueError("output must differ from the source path")
    if options.apply and options.output is not None and options.output.exists():
        raise ValueError("output already exists; choose a new output path")
    if options.output is not None and source.suffix.lower() != options.output.suffix.lower():
        raise ValueError("output format must match the source format")


def improve_attribution(
    source: str | Path,
    options: ReductionOptions | None = None,
) -> ReductionResult:
    """Plan or apply bounded, verified attribution-risk reduction."""
    source_path = Path(source)
    options = options or ReductionOptions()
    from ..llm import get_audit_log

    llm_log_start = len(get_audit_log())
    from ..corpus.loader import load_similarity_corpus

    if options.corpus is not None:
        options.corpus = Path(options.corpus)
    if options.output is not None:
        options.output = Path(options.output)
    _validate_options(source_path, options)
    index = options.similarity_index or load_similarity_corpus(
        options.corpus, options.corpus_license
    )
    _enriched, baseline = _scan(source_path, index)
    baseline_risks = analyze_passage_risks(baseline)
    baseline_plans = build_fix_plans(baseline_risks)
    if not options.apply:
        metrics = compute_reduction_metrics(
            baseline,
            baseline,
            unchanged_passages=len(baseline_risks),
            manual_review_passages=sum(
                plan.action not in {FixAction.LEAVE, FixAction.PARAPHRASE}
                for plan in baseline_plans
            ),
            stop_reason="preview_only",
            application_format=source_path.suffix.lower().lstrip("."),
        )
        return ReductionResult(
            source_path,
            None,
            True,
            False,
            0,
            "preview_only",
            baseline,
            baseline,
            baseline_risks,
            baseline_plans,
            [],
            metrics,
            {
                "offline": options.offline,
                "network_allowed": not options.offline,
                "network_used": False,
                "academic_network_used": False,
                "llm_network_used": False,
                "semantic_backend": "not_run",
                "nli_backend": "not_run",
                "rewrite_provider": "not_run",
                "rewrite_model": options.model,
            },
            {
                plan.passage_id: plan.action.value
                for plan in baseline_plans
                if plan.action not in {FixAction.LEAVE, FixAction.PARAPHRASE}
            },
        )

    provider = options.rewrite_provider or LLMRewriteProvider(
        model=options.model,
        provider=options.provider,
        offline=options.offline,
    )
    allow_download = options.allow_model_download and not options.offline
    semantic_backend = options.semantic_backend or SentenceTransformerSemanticBackend(
        options.semantic_model, allow_download=allow_download
    )
    entailment_backend = options.entailment_backend or TransformerNLIBackend(
        options.nli_model, allow_download=allow_download
    )
    changes: list[ReductionChange] = []
    manual: dict[str, str] = {}
    generated = rejected = accepted = 0
    iterations = 0
    stop_reason = "max_iterations_reached"
    audits: list[AuditResult] = []

    with tempfile.TemporaryDirectory(prefix="citeguard-reduction-") as temp_dir:
        current_path = source_path
        for iteration in range(options.max_iterations):
            _current_enriched, current_scan = _scan(current_path, index)
            risks = analyze_passage_risks(current_scan)
            if options.severity:
                levels = {"low": 1, "medium": 2, "high": 3}
                risks = [
                    risk
                    for risk in risks
                    if levels[risk.attribution_risk.value] >= levels[options.severity]
                ]
            plans = build_fix_plans(risks)
            eligible = [plan for plan in plans if plan.rewrite_allowed]
            for plan in plans:
                if plan.action not in {FixAction.LEAVE, FixAction.PARAPHRASE}:
                    manual[plan.passage_id] = plan.action.value
            if not eligible:
                stop_reason = "no_eligible_reducible_passages"
                break

            audit = _audit(current_path, options)
            audits.append(audit)
            grounded, grounding_manual = build_grounded_reduction_requests(
                audit, risks, plans, mode=RewriteMode.CLARIFY
            )
            manual.update(grounding_manual)
            backend = EvidenceGroundedReductionBackend(provider, grounded, offline=options.offline)
            risk_map = {risk.passage_id: risk for risk in risks}
            source_texts = _source_texts(current_scan)
            replacements: list[TextReplacement] = []
            selected_changes: list[ReductionChange] = []

            for plan in eligible:
                risk = risk_map[plan.passage_id]
                source_text = source_texts.get(plan.passage_id)
                request = ReductionRewriteRequest(risk.text, plan, options.candidate_count)
                candidates = generate_candidates(backend, request)
                generated += len(candidates)
                grounded_request = grounded.get(plan.passage_id)
                evidence = grounded_request.context.evidence_texts if grounded_request else ()
                supported_verdict = (
                    Verdict(grounded_request.context.verdict) if grounded_request else None
                )
                for candidate in candidates:
                    evaluate_candidate(risk.text, candidate)
                    evaluate_source_overlap(risk.text, candidate, source_text or "")
                    validate_candidate(
                        risk.text,
                        candidate,
                        plan,
                        supported_verdict=supported_verdict,
                    )
                    if not candidate.rejection_reasons:
                        unsupported = _unsupported_assertions(
                            risk.text, candidate.text, evidence, entailment_backend
                        )
                        if unsupported:
                            validate_candidate(
                                risk.text,
                                candidate,
                                plan,
                                supported_verdict=supported_verdict,
                                unsupported_claims=unsupported,
                            )
                    if not candidate.rejection_reasons:
                        validate_meaning(
                            risk.text,
                            candidate,
                            semantic_backend=semantic_backend,
                            entailment_backend=entailment_backend,
                            thresholds=options.meaning_thresholds,
                        )
                    candidate.source_support_score = (
                        1.0 if supported_verdict == Verdict.SUPPORTED else 0.7
                    )
                ranked = rank_candidates(candidates)
                rejected += len(candidates) - len(ranked)
                for candidate in candidates:
                    if candidate not in ranked:
                        changes.append(
                            _change(
                                risk,
                                plan,
                                candidate,
                                source_text=source_text,
                                candidate_count=len(candidates),
                                status="rejected",
                            )
                        )
                if not ranked:
                    manual[plan.passage_id] = (
                        grounding_manual.get(plan.passage_id)
                        or "no candidate passed every safety gate"
                    )
                    continue
                selected = ranked[0]
                replacements.append(TextReplacement(risk.text, selected.text, risk.passage_id))
                selected_changes.append(
                    _change(
                        risk,
                        plan,
                        selected,
                        source_text=source_text,
                        candidate_count=len(candidates),
                        status="selected",
                    )
                )

            if not replacements:
                stop_reason = "no_safe_candidates"
                break

            next_path = Path(temp_dir) / f"iteration-{iteration + 1}{source_path.suffix}"
            try:
                if source_path.suffix.lower() in {".md", ".txt"}:
                    text_records = write_revised_text(current_path, next_path, replacements)
                    text_by_id = {record.passage_id: record for record in text_records}
                    for item in selected_changes:
                        record = text_by_id[item.passage_id]
                        item.start_offset = record.start_offset
                        item.end_offset = record.end_offset
                elif source_path.suffix.lower() == ".docx":
                    from .docx_apply import write_revised_docx

                    docx_records = write_revised_docx(current_path, next_path, replacements)
                    docx_by_id = {record.passage_id: record for record in docx_records}
                    for item in selected_changes:
                        record = docx_by_id[item.passage_id]
                        item.affected_runs = (record.run_index,)
                        item.formatting_preserved = record.formatting_preserved
                else:
                    raise ValueError("automatic application supports .md, .txt, and .docx")
            except ValueError as exc:
                for item in selected_changes:
                    item.application_status = "manual_review"
                    item.warnings = (str(exc),)
                    manual[item.passage_id] = str(exc)
                changes.extend(selected_changes)
                stop_reason = "application_failed"
                break

            _revised_enriched, revised_scan = _scan(next_path, index)
            improved = revised_scan.high_risk_count <= current_scan.high_risk_count and (
                revised_scan.high_risk_count < current_scan.high_risk_count
                or revised_scan.overall_similarity_pct < current_scan.overall_similarity_pct - 1e-9
            )
            if not improved:
                for item in selected_changes:
                    item.application_status = "rolled_back_no_improvement"
                    item.warnings = ("actual same-corpus rescan found no improvement",)
                changes.extend(selected_changes)
                stop_reason = "no_measured_improvement"
                break

            for item in selected_changes:
                item.application_status = "applied"
            changes.extend(selected_changes)
            accepted += len(selected_changes)
            iterations += 1
            current_path = next_path
        else:
            stop_reason = "max_iterations_reached"

        assert options.output is not None
        options.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{options.output.stem}-",
            suffix=options.output.suffix,
            dir=options.output.parent,
        )
        os.close(descriptor)
        temporary_output = Path(temporary_name)
        try:
            shutil.copy2(current_path, temporary_output)
            os.replace(temporary_output, options.output)
        finally:
            if temporary_output.exists():
                temporary_output.unlink()
        _actual_enriched, actual_after = _scan(options.output, index)

    applied_changes = [c for c in changes if c.application_status == "applied"]
    meaning_scores = [c.meaning_score for c in applied_changes if c.meaning_score is not None]
    metrics = compute_reduction_metrics(
        baseline,
        actual_after,
        rewritten_passages=len(applied_changes),
        unchanged_passages=max(len(baseline_risks) - len(applied_changes), 0),
        manual_review_passages=len(manual),
        generated_candidates=generated,
        rejected_candidates=rejected,
        accepted_candidates=accepted,
        meaning_scores=meaning_scores,
        citation_integrity_passed=(
            all(c.citation_preserved is True for c in applied_changes) if applied_changes else None
        ),
        numeric_integrity_passed=(
            all(c.numeric_integrity is True for c in applied_changes) if applied_changes else None
        ),
        named_entity_integrity_passed=(
            all(c.named_entity_integrity is True for c in applied_changes)
            if applied_changes
            else None
        ),
        new_unsupported_claims=sum(len(c.unsupported_new_claims) for c in changes),
        iterations_completed=iterations,
        stop_reason=stop_reason,
        application_format=source_path.suffix.lower().lstrip("."),
    )
    academic_network = any(a.execution.academic_network_used for a in audits)
    llm_entries = get_audit_log()[llm_log_start:]
    llm_network = any(not entry.cached for entry in llm_entries)
    return ReductionResult(
        source_path,
        options.output,
        False,
        bool(applied_changes),
        iterations,
        stop_reason,
        baseline,
        actual_after,
        baseline_risks,
        baseline_plans,
        changes,
        metrics,
        {
            "offline": options.offline,
            "network_allowed": not options.offline,
            "network_used": academic_network or llm_network,
            "academic_network_used": academic_network,
            "llm_network_used": llm_network,
            "semantic_backend": type(semantic_backend).__name__,
            "nli_backend": type(entailment_backend).__name__,
            "rewrite_provider": type(provider).__name__,
            "rewrite_model": options.model,
            "maximum_rewrite_calls": options.max_iterations
            * len(baseline_plans)
            * options.candidate_count,
        },
        manual,
    )
