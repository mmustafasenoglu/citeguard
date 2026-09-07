"""Click-based command-line interface for citeguard."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__
from .bibliography import bibliography_issues, citation_matches_entry
from .cache import FileCache
from .claims import extract_claims
from .config import _PROVIDER_DEFAULT_MODELS, DEFAULT_MAX_RESULTS, DEFAULT_THRESHOLD, Settings
from .extractor import parse_document
from .llm import extract_claims_with_llm
from .matcher import match_claim_to_source
from .models import (
    Claim,
    MatchResult,
    ParsedDocument,
    VerificationResult,
    VerificationStatus,
)
from .providers.arxiv import ArxivProvider
from .providers.crossref import CrossrefProvider
from .providers.semantic_scholar import SemanticScholarProvider
from .report import (
    check_report,
    inspection_report,
    inspection_report_markdown,
    markdown_check_report,
    markdown_suggest_report,
    markdown_verification_report,
    similarity_report,
    suggest_report,
    verification_report,
)
from .retrieval import RetrievalEngine
from .scoring import AuditMetrics, compute_audit_metrics, overall_confidence, priority_list
from .similarity.index import SimilarityIndex
from .similarity.models import SimilarityEngineResult
from .verification import verify_bibliography

console = Console()

EXIT_SUCCESS = 0
EXIT_FINDINGS = 1
EXIT_INVALID_INPUT = 2
EXIT_PROVIDER_FAILURE = 3


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="citeguard")
def main() -> None:
    """Audit citations and citation-worthy claims in academic writing."""
    load_dotenv()


@main.command("init")
@click.option("--force", is_flag=True, help="Overwrite an existing .env file.")
def init_command(force: bool) -> None:
    """Create a local .env template for API keys."""
    path = Path(".env")
    if path.exists() and not force:
        raise click.ClickException(".env already exists. Use --force to overwrite it.")
    path.write_text(
        "# LLM provider selection (optional).\n"
        "# Supported: anthropic, openai, xai, groq, openrouter, nvidia, custom\n"
        "# When omitted, auto-detected from available API keys.\n"
        "# CITEGUARD_LLM_PROVIDER=anthropic\n"
        "\n"
        "# LLM model override (optional, provider-specific default used when omitted).\n"
        "# CITEGUARD_LLM_MODEL=claude-sonnet-4-20250514\n"
        "\n"
        "# Failover chain (optional). Comma-separated list of fallback providers.\n"
        "# When the primary provider fails, citeguard tries each in order.\n"
        "# CITEGUARD_LLM_FALLBACKS=openai,groq,openrouter\n"
        "\n"
        "# Task-specific model overrides (optional).\n"
        "# Use a cheaper/faster model for claim extraction and a stronger model\n"
        "# for entailment classification.\n"
        "# CITEGUARD_CLAIM_PROVIDER=groq\n"
        "# CITEGUARD_CLAIM_MODEL=llama-3.3-70b-versatile\n"
        "# CITEGUARD_ENTAILMENT_PROVIDER=openai\n"
        "# CITEGUARD_ENTAILMENT_MODEL=gpt-4o\n"
        "\n"
        "# Anthropic (default provider)\n"
        "# ANTHROPIC_API_KEY=\n"
        "\n"
        "# OpenAI (Responses API)\n"
        "# OPENAI_API_KEY=\n"
        "\n"
        "# xAI / Grok (Responses API)\n"
        "# XAI_API_KEY=\n"
        "\n"
        "# Groq (OpenAI-compatible)\n"
        "# GROQ_API_KEY=\n"
        "\n"
        "# OpenRouter (OpenAI-compatible)\n"
        "# OPENROUTER_API_KEY=\n"
        "\n"
        "# NVIDIA NIM (OpenAI-compatible)\n"
        "# NVIDIA_API_KEY=\n"
        "\n"
        "# Custom / local endpoint (Ollama, LM Studio, vLLM, LiteLLM, etc.)\n"
        "# CITEGUARD_LLM_BASE_URL=http://localhost:11434/v1\n"
        "# CITEGUARD_LLM_API_KEY=local\n"
        "# CITEGUARD_LLM_MODEL=qwen3\n"
        "\n"
        "# Academic providers\n"
        "# Optional. citeguard can use Semantic Scholar without a key.\n"
        "# SEMANTIC_SCHOLAR_API_KEY=\n",
        encoding="utf-8",
    )
    console.print("[green]Created .env[/green]")


def _default_model_for(provider: str) -> str:
    return _PROVIDER_DEFAULT_MODELS.get(provider, "default")


# ---------------------------------------------------------------------------
# LLM diagnostics subgroup
# ---------------------------------------------------------------------------


@main.group("llm")
def llm_group() -> None:
    """LLM provider diagnostics and configuration."""


@llm_group.command("doctor")
def llm_doctor_command() -> None:
    """Test connectivity to all configured LLM providers."""
    from .config import _resolve_api_key
    from .llm_backends import (
        LLMResponse,
        auto_detect_provider,
        resolve_backend,
    )

    providers = [
        "anthropic", "openai", "xai", "groq", "openrouter", "nvidia",
    ]
    custom_url = os.getenv("CITEGUARD_LLM_BASE_URL", "").strip()

    table = Table(title="LLM Provider Diagnostics")
    table.add_column("Provider")
    table.add_column("Status")
    table.add_column("Latency", justify="right")
    table.add_column("Detail")

    active = auto_detect_provider() or os.getenv(
        "CITEGUARD_LLM_PROVIDER", ""
    ).strip().lower()

    for name in providers:
        key = _resolve_api_key(name)
        if not key:
            table.add_row(name, "[red]no key[/red]", "-", "")
            continue

        backend = resolve_backend(name)
        if backend is None:
            table.add_row(name, "[yellow]unavailable[/yellow]", "-", "")
            continue

        resp: LLMResponse = backend.chat(
            "Reply with exactly: ok",
            "Say ok",
            model=_default_model_for(name),
            max_tokens=8,
            timeout=10,
        )
        latency = f"{resp.latency_ms:.0f} ms"

        if resp.text:
            table.add_row(
                name,
                "[green]ok[/green]",
                latency,
                resp.model,
            )
        elif resp.error and "401" in str(resp.status_code):
            table.add_row(
                name,
                "[red]auth error[/red]",
                latency,
                "Invalid API key",
            )
        elif resp.error and "429" in str(resp.status_code):
            table.add_row(
                name,
                "[yellow]rate limited[/yellow]",
                latency,
                "Try again later",
            )
        else:
            table.add_row(
                name,
                "[red]error[/red]",
                latency,
                (resp.error or "unknown")[:50],
            )

    if custom_url:
        backend = resolve_backend("custom")
        if backend:
            resp = backend.chat(
                "Reply with exactly: ok",
                "Say ok",
                model=os.getenv("CITEGUARD_LLM_MODEL", "default"),
                max_tokens=8,
                timeout=10,
            )
            latency = f"{resp.latency_ms:.0f} ms"
            if resp.text:
                table.add_row(
                    "custom",
                    "[green]ok[/green]",
                    latency,
                    custom_url,
                )
            else:
                table.add_row(
                    "custom",
                    "[red]error[/red]",
                    latency,
                    (resp.error or custom_url)[:50],
                )
        else:
            table.add_row(
                "custom",
                "[yellow]no base url[/yellow]",
                "-",
                custom_url,
            )

    console.print(table)

    if active:
        console.print(f"\n[bold]Active provider:[/bold] {active}")
    fallbacks = os.getenv("CITEGUARD_LLM_FALLBACKS", "").strip()
    if fallbacks:
        console.print(f"[bold]Fallbacks:[/bold] {fallbacks}")
    else:
        console.print("[dim]No fallbacks configured.[/dim]")
    console.print(
        "\n[dim]Only connection status is tested. "
        "No document content is sent.[/dim]"
    )


@llm_group.command("list")
def llm_list_command() -> None:
    """Show current LLM configuration (no secrets)."""
    from .config import LLMProviderSettings, TaskModelSettings, _resolve_api_key

    settings = LLMProviderSettings.from_env()

    table = Table(title="LLM Configuration")
    table.add_column("Setting")
    table.add_column("Value")

    table.add_row("Provider", settings.provider)
    table.add_row("Model", settings.model)

    key = _resolve_api_key(settings.provider)
    if key:
        masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
        table.add_row("API Key", f"[green]{masked}[/green]")
    else:
        table.add_row("API Key", "[red]not set[/red]")

    timeout = os.getenv("CITEGUARD_LLM_TIMEOUT", "30")
    table.add_row("Timeout", f"{timeout}s")

    fallbacks = os.getenv("CITEGUARD_LLM_FALLBACKS", "")
    table.add_row("Fallbacks", fallbacks or "(none)")

    if settings.base_url:
        table.add_row("Base URL", settings.base_url)

    console.print(table)

    # Task-specific models
    claim = TaskModelSettings.from_env("claim")
    entailment = TaskModelSettings.from_env("entailment")

    if claim.provider or claim.model:
        task_table = Table(title="Task-Specific Models")
        task_table.add_column("Task")
        task_table.add_column("Provider")
        task_table.add_column("Model")
        task_table.add_row(
            "Claim extraction",
            claim.provider or "(global)",
            claim.model or "(global)",
        )
        task_table.add_row(
            "Entailment",
            entailment.provider or "(global)",
            entailment.model or "(global)",
        )
        console.print(task_table)



@main.command("inspect")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--details/--no-details",
    default=True,
    help="Show each detected citation and its bibliography match status.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["terminal", "json", "both"]),
    default="terminal",
    show_default=True,
)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path))
def inspect_command(
    file: Path, details: bool, output_format: str, output: Path | None
) -> None:
    """Inspect parsing and bibliography detection without using network APIs."""
    parsed = parse_document(file)
    _validate_parsed(parsed)
    if output_format == "json":
        _write_json(inspection_report(parsed), output)
        return
    if output_format == "both":
        json_path, md_path = _both_paths(output, file)
        _write_json(inspection_report(parsed), json_path)
        _write_text(inspection_report_markdown(parsed), md_path)
        return
    issues = bibliography_issues(parsed.citations, parsed.bibliography_entries)

    table = Table(title="citeguard document inspection")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Paragraphs", str(len(parsed.paragraphs)))
    table.add_row("Detected citations", str(len(parsed.citations)))
    table.add_row("Bibliography entries", str(len(parsed.bibliography_entries)))
    table.add_row("Bibliography issues", str(len(issues)))
    console.print(table)

    if details and parsed.citations:
        citation_table = Table(title="Detected citations")
        citation_table.add_column("Paragraph", justify="right")
        citation_table.add_column("Citation")
        citation_table.add_column("Bibliography")
        for citation in parsed.citations:
            matched = any(
                citation_matches_entry(citation, entry)
                for entry in parsed.bibliography_entries
            )
            if citation.numbered_ref is not None:
                match_status = "not checked"
            elif citation.doi and not matched:
                match_status = "not listed"
            else:
                match_status = "matched" if matched else "not listed"
            citation_table.add_row(
                str(citation.paragraph_index + 1),
                citation.raw_text,
                match_status,
            )
        console.print()
        console.print(citation_table)

    if issues:
        console.print("\n[yellow]Bibliography issues[/yellow]")
        for issue in issues:
            location = (
                f" (paragraph {issue.paragraph_index + 1})"
                if issue.paragraph_index is not None
                else ""
            )
            console.print(f"- {issue.kind.value}{location}: {issue.detail}")


@main.command("verify")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--max-results", type=click.IntRange(1, 20), default=DEFAULT_MAX_RESULTS)
@click.option("--no-cache", is_flag=True, help="Do not read or write provider cache entries.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["terminal", "json", "md", "both"]),
    default="terminal",
    show_default=True,
)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path))
def verify_command(
    file: Path,
    max_results: int,
    no_cache: bool,
    output_format: str,
    output: Path | None,
) -> None:
    """Verify bibliography metadata with Crossref; document prose is not sent."""
    parsed = parse_document(file)
    _validate_parsed(parsed)
    settings = Settings.from_env()
    engine = RetrievalEngine(
        [CrossrefProvider()],
        cache=FileCache(settings.cache_dir),
        use_cache=not no_cache,
    )
    results = verify_bibliography(
        parsed.bibliography_entries,
        engine,
        max_results=max_results,
    )
    if output_format == "json":
        _write_json(verification_report(parsed, results), output)
        return
    if output_format == "md":
        report_text = markdown_verification_report(parsed, results)
        _write_text(report_text, output)
        return
    if output_format == "both":
        json_path, md_path = _both_paths(output, file)
        _write_json(verification_report(parsed, results), json_path)
        report_text = markdown_verification_report(parsed, results)
        _write_text(report_text, md_path)
        return

    table = Table(title="citeguard Crossref verification")
    table.add_column("#", justify="right")
    table.add_column("Bibliography entry")
    table.add_column("Status")
    table.add_column("Score", justify="right")
    table.add_column("Best candidate")
    for result in results:
        entry_label = result.entry.title or result.entry.authors or result.entry.raw_text
        candidate_title = result.candidate.title if result.candidate else "-"
        score = str(result.scores.overall) if result.scores else "-"
        table.add_row(
            str(result.entry_index + 1),
            entry_label,
            result.status.value,
            score,
            candidate_title,
        )
    console.print(table)
    console.print(
        "[dim]Only bibliography DOI/title/author/year metadata was sent to Crossref. "
        "Document prose was not sent. Verification does not establish claim support.[/dim]"
    )
    warnings = [warning for result in results for warning in result.warnings]
    if warnings:
        console.print("\n[yellow]Provider warnings[/yellow]")
        for warning in dict.fromkeys(warnings):
            console.print(f"- {warning}")


@main.command("suggest")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--max-results", type=click.IntRange(1, 20), default=DEFAULT_MAX_RESULTS)
@click.option("--threshold", type=click.IntRange(0, 100), default=DEFAULT_THRESHOLD)
@click.option("--max-claims", type=click.IntRange(1, 500), default=None)
@click.option("--severity", type=click.Choice(["high", "medium", "low"]), default=None)
@click.option("--no-cache", is_flag=True, help="Do not read or write provider cache entries.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["terminal", "json", "md", "both"]),
    default="terminal",
    show_default=True,
)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--verbose", is_flag=True, help="Show detailed progress information.")
def suggest_command(
    file: Path,
    max_results: int,
    threshold: int,
    max_claims: int | None,
    severity: str | None,
    no_cache: bool,
    output_format: str,
    output: Path | None,
    verbose: bool,
) -> None:
    """Suggest academic sources for uncited claims."""
    parsed = parse_document(file)
    _validate_parsed(parsed)
    settings = Settings.from_env()
    cache = FileCache(settings.cache_dir)

    claims = _extract_claims_hybrid(
        parsed, max_claims=max_claims, severity=severity, verbose=verbose,
    )

    uncited_claims = [c for c in claims if not c.has_existing_citation]
    if verbose:
        console.print(f"[dim]Detected {len(claims)} claims, {len(uncited_claims)} uncited.[/dim]")

    engine = RetrievalEngine(
        [SemanticScholarProvider(), CrossrefProvider(), ArxivProvider()],
        cache=cache,
        use_cache=not no_cache,
    )

    suggestion_results: list[VerificationResult] = []
    provider_failed = False

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=not verbose,
    ) as progress:
        task = progress.add_task(
            "Searching providers...",
            total=len(uncited_claims) if uncited_claims else None,
        )
        for claim in uncited_claims:
            progress.update(
                task,
                description=f"[dim]Searching: {claim.text[:50]}...[/dim]",
            )
            try:
                candidates = engine.search(
                    claim.search_query, max_results=max_results
                )
            except Exception as exc:
                console.print(f"[yellow]Provider error for claim: {exc}[/yellow]")
                provider_failed = True
                suggestion_results.append(
                    VerificationResult(
                        claim=claim,
                        status=VerificationStatus.NOT_FOUND,
                        citation=None,
                        matched=None,
                        warnings=[str(exc)],
                    )
                )
                progress.advance(task)
                continue

            match_results: list[MatchResult] = []
            for candidate in candidates:
                metadata_score, support_score, verdict, reasoning, evidence = (
                    match_claim_to_source(claim, candidate)
                )
                has_entailment = any(e.entailment_score is not None for e in evidence)
                confidence = overall_confidence(
                    metadata_score, support_score, has_entailment=has_entailment
                )
                match_results.append(
                    MatchResult(
                        candidate=candidate,
                        source_exists=True,
                        metadata_match_score=metadata_score,
                        claim_support_score=support_score,
                        overall_confidence=confidence,
                        verdict=verdict,
                        reasoning=reasoning,
                        evidence=evidence,
                    )
                )

            filtered = [
                m for m in match_results if m.overall_confidence >= threshold
            ]
            suggestion_results.append(
                VerificationResult(
                    claim=claim,
                    status=(
                        VerificationStatus.SUGGESTED
                        if filtered
                        else VerificationStatus.NOT_FOUND
                    ),
                    citation=None,
                    matched=filtered[0] if filtered else None,
                    suggestions=filtered,
                )
            )
            progress.advance(task)

    report = suggest_report(parsed, claims, suggestion_results)

    if output_format == "json":
        _write_json(report, output)
        return
    if output_format == "md":
        report_text = markdown_suggest_report(parsed, claims, suggestion_results)
        _write_text(report_text, output)
        return
    if output_format == "both":
        json_path, md_path = _both_paths(output, file)
        _write_json(report, json_path)
        report_text = markdown_suggest_report(parsed, claims, suggestion_results)
        _write_text(report_text, md_path)
        return

    _print_suggest_terminal(claims, suggestion_results, threshold, verbose)

    if provider_failed:
        console.print(
            "\n[yellow]Some providers failed. Results may be incomplete.[/yellow]"
        )


@main.command("check")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--max-results", type=click.IntRange(1, 20), default=DEFAULT_MAX_RESULTS)
@click.option("--threshold", type=click.IntRange(0, 100), default=DEFAULT_THRESHOLD)
@click.option("--max-claims", type=click.IntRange(1, 500), default=None)
@click.option("--severity", type=click.Choice(["high", "medium", "low"]), default=None)
@click.option("--no-cache", is_flag=True, help="Do not read or write provider cache entries.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["terminal", "json", "md", "both"]),
    default="terminal",
    show_default=True,
)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--verbose", is_flag=True, help="Show detailed progress information.")
@click.option("--show-evidence", is_flag=True, help="Show evidence passages for matched claims.")
@click.option("--require-evidence", is_flag=True, help="Only show claims that have evidence.")
@click.option("--show-similarity", is_flag=True, help="Run similarity engine (CTAC v1).")
@click.option("--corpus", type=click.Path(exists=True, path_type=Path), help="Similarity corpus.")
@click.option("--corpus-license", default=None, help="Explicit corpus license (CC0, CC-BY).")
def check_command(
    file: Path,
    max_results: int,
    threshold: int,
    max_claims: int | None,
    severity: str | None,
    no_cache: bool,
    output_format: str,
    output: Path | None,
    verbose: bool,
    show_evidence: bool,
    require_evidence: bool,
    show_similarity: bool,
    corpus: Path | None,
    corpus_license: str | None,
) -> None:
    from .extractor import parse_enriched_document
    from .similarity.engine import SimilarityEngine
    from .similarity.models import SimilarityConfig

    parsed = parse_document(file)
    _validate_parsed(parsed)
    settings = Settings.from_env()
    cache = FileCache(settings.cache_dir)
    
    sim_result = None
    if show_similarity:
        console.print("[dim]Running similarity analysis...[/dim]")
        enriched = parse_enriched_document(file)
        index = _load_similarity_corpus(corpus, corpus_license)
        sim_engine = SimilarityEngine(config=SimilarityConfig())
        sim_result = sim_engine.analyze_document(
            enriched.sentences,
            index=index,
            bibliography_entries=enriched.bibliography_entries,
        )

    bib_issues = bibliography_issues(parsed.citations, parsed.bibliography_entries)

    claims = _extract_claims_hybrid(
        parsed, max_claims=max_claims, severity=severity, verbose=verbose,
    )

    if verbose:
        console.print(f"[dim]Extracted {len(claims)} claims from document.[/dim]")

    # Verify bibliography
    bib_engine = RetrievalEngine(
        [CrossrefProvider()],
        cache=cache,
        use_cache=not no_cache,
    )
    bib_results = verify_bibliography(
        parsed.bibliography_entries,
        bib_engine,
        max_results=max_results,
    )

    # Search for uncited claims
    uncited_claims = [c for c in claims if not c.has_existing_citation]
    claim_verification: list[VerificationResult] = []
    provider_failed = False

    if uncited_claims:
        search_engine = RetrievalEngine(
            [SemanticScholarProvider(), CrossrefProvider(), ArxivProvider()],
            cache=cache,
            use_cache=not no_cache,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=not verbose,
        ) as progress:
            task = progress.add_task(
                "Searching providers...",
                total=len(uncited_claims),
            )
            for claim in uncited_claims:
                progress.update(
                    task,
                    description=f"[dim]Searching: {claim.text[:50]}...[/dim]",
                )
                try:
                    candidates = search_engine.search(
                        claim.search_query, max_results=max_results
                    )
                except Exception as exc:
                    provider_failed = True
                    claim_verification.append(
                        VerificationResult(
                            claim=claim,
                            status=VerificationStatus.NOT_FOUND,
                            citation=None,
                            matched=None,
                            warnings=[str(exc)],
                        )
                    )
                    progress.advance(task)
                    continue

                best_match: MatchResult | None = None
                all_matches: list[MatchResult] = []
                for candidate in candidates:
                    metadata_score, support_score, verdict, reasoning, evidence = (
                        match_claim_to_source(claim, candidate)
                    )
                    has_entailment = any(e.entailment_score is not None for e in evidence)
                    confidence = overall_confidence(
                        metadata_score, support_score, has_entailment=has_entailment
                    )
                    ent_score = evidence[0].entailment_score if evidence else None
                    ent_verdict = (
                        evidence[0].verdict
                        if evidence and ent_score is not None
                        else None
                    )
                    match_result = MatchResult(
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
                    all_matches.append(match_result)
                    if best_match is None or confidence > best_match.overall_confidence:
                        best_match = match_result

                claim_verification.append(
                    VerificationResult(
                        claim=claim,
                        status=(
                            VerificationStatus.SUGGESTED
                            if best_match
                            else VerificationStatus.NOT_FOUND
                        ),
                        citation=None,
                        matched=best_match,
                        suggestions=[
                            m
                            for m in all_matches
                            if m.overall_confidence >= threshold
                        ],
                    )
                )
                progress.advance(task)

    metrics = compute_audit_metrics(
        claims=claims,
        verification_results=claim_verification,
        bibliography_issues=bib_issues,
    )

    sorted_claims = priority_list(claim_verification)

    if require_evidence:
        sorted_claims = [
            r for r in sorted_claims
            if r.matched and r.matched.evidence
        ]

    report = check_report(
        parsed,
        bib_results,
        claims,
        sorted_claims,
        metrics,
        bib_issues,
        sim_result,
    )

    if output_format == "json":
        _write_json(report, output)
        exit_code = EXIT_FINDINGS if metrics.health_score < 80 else EXIT_SUCCESS
        raise SystemExit(exit_code)
    if output_format == "md":
        report_text = markdown_check_report(
            parsed, bib_results, claims, sorted_claims, metrics, bib_issues, sim_result
        )
        _write_text(report_text, output)
        exit_code = EXIT_FINDINGS if metrics.health_score < 80 else EXIT_SUCCESS
        raise SystemExit(exit_code)
    if output_format == "both":
        json_path, md_path = _both_paths(output, file)
        _write_json(report, json_path)
        report_text = markdown_check_report(
            parsed, bib_results, claims, sorted_claims, metrics, bib_issues, sim_result
        )
        _write_text(report_text, md_path)
        exit_code = EXIT_FINDINGS if metrics.health_score < 80 else EXIT_SUCCESS
        raise SystemExit(exit_code)

    show_ev = show_evidence or verbose
    _print_check_terminal(metrics, claims, sorted_claims, bib_issues, verbose, show_ev, sim_result)

    if provider_failed:
        console.print(
            "\n[yellow]Some providers failed. Results may be incomplete.[/yellow]"
        )

    exit_code = EXIT_FINDINGS if metrics.health_score < 80 else EXIT_SUCCESS
    raise SystemExit(exit_code)


def _print_check_terminal(
    metrics: AuditMetrics,
    claims: list[Claim],
    sorted_claims: list[VerificationResult],
    bib_issues: list,
    verbose: bool,
    show_evidence: bool,
    sim_result: SimilarityEngineResult | None = None,
) -> None:
    if metrics.health_score >= 80:
        health_color = "green"
    elif metrics.health_score >= 60:
        health_color = "yellow"
    else:
        health_color = "red"
    console.print(
        f"\n[{health_color}]Citation Health Score: "
        f"{metrics.health_score}/100[/{health_color}]"
    )
    console.print(
        "[dim]The health score is a review-prioritization heuristic, "
        "not a measure of scientific correctness.[/dim]\n"
    )

    table = Table(title="Audit Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total claims", str(metrics.total_claims))
    table.add_row("Claims requiring citations", str(metrics.claims_requiring_citations))
    table.add_row("Cited claims", str(metrics.cited_claims))
    table.add_row("Verified citations", str(metrics.verified_citations))
    table.add_row("Weak matches", str(metrics.weak_matches))
    table.add_row("Unresolved citations", str(metrics.unresolved_citations))
    table.add_row("Uncited high-severity claims", str(metrics.uncited_high_severity_claims))
    table.add_row("Contradictions", str(metrics.contradictions))
    table.add_row("Bibliography issues", str(metrics.bibliography_issues))
    table.add_row("Citation coverage", f"{metrics.citation_coverage:.1%}")
    table.add_row("Verification ratio", f"{metrics.verification_ratio:.1%}")
    table.add_row("Support ratio", f"{metrics.support_ratio:.1%}")
    table.add_row("Bibliography consistency", f"{metrics.bibliography_consistency:.1%}")
    
    if sim_result:
        table.add_row("Overall similarity", f"{sim_result.overall_similarity_pct:.1f}%")
        table.add_row("High risk similarity", str(sim_result.high_risk_count))
        
    console.print(table)

    if sorted_claims:
        console.print("\n[yellow]Priority review list[/yellow]")
        for result in sorted_claims[:10]:
            verdict = result.matched.verdict.value if result.matched else "n/a"
            confidence = result.matched.overall_confidence if result.matched else 0
            sev = result.claim.severity.value
            color = _severity_color(sev)
            console.print(
                f"  [{color}]{sev.upper()}[/{color}]"
                f" (confidence: {confidence})"
                f" - {result.claim.text[:70]}"
                f" [verdict: {verdict}]"
            )

    if show_evidence and sorted_claims:
        console.print("\n[cyan]Evidence[/cyan]")
        for result in sorted_claims[:5]:
            matched = result.matched
            if matched and matched.evidence:
                sev = result.claim.severity.value
                color = _severity_color(sev)
                console.print(
                    f"  [{color}]{sev.upper()}[/{color}]"
                    f" [verdict: {matched.verdict.value}]"
                    f" (confidence: {matched.overall_confidence})"
                )
                console.print(f"    Claim: {result.claim.text[:80]}")
                for i, ev in enumerate(matched.evidence[:2], 1):
                    console.print(f"    Evidence {i} (relevance: {ev.lexical_score}/100):")
                    console.print(f"      {ev.text[:120]}")
                    console.print(f"      [dim]Source: {ev.source_title}[/dim]")

    if bib_issues:
        console.print("\n[yellow]Bibliography issues[/yellow]")
        for issue in bib_issues:
            console.print(f"  - {issue.kind.value}: {issue.detail}")


def _print_suggest_terminal(
    claims: list[Claim],
    results: list[VerificationResult],
    threshold: int,
    verbose: bool,
) -> None:
    uncited = [c for c in claims if not c.has_existing_citation]
    with_suggestions = [r for r in results if r.suggestions]

    console.print("\n[bold]Source Suggestions[/bold]")
    console.print(f"Claims: {len(claims)} total, {len(uncited)} uncited, "
                  f"{len(with_suggestions)} with suggestions\n")

    for result in results:
        if not result.suggestions:
            continue
        console.print(f"[bold]Claim:[/bold] {result.claim.text}")
        console.print(
            f"  Type: {result.claim.claim_type.value} | "
            f"Severity: {result.claim.severity.value} | "
            f"Paragraph: {result.claim.paragraph_index + 1}"
        )
        for s in result.suggestions[:3]:
            authors_short = ", ".join(s.candidate.authors[:2])
            if len(s.candidate.authors) > 2:
                authors_short += " et al."
            console.print(
                f"  -> {s.candidate.title[:60]} ({authors_short}, "
                f"{s.candidate.year or 'n.d.'}) "
                f"[score: {s.overall_confidence}, verdict: {s.verdict.value}]"
            )
        console.print()

    console.print(
        "[dim]Suggestions require human review and do not confirm source support.[/dim]"
    )


def _severity_color(severity: str) -> str:
    return {"high": "red", "medium": "yellow", "low": "dim"}.get(severity, "white")


def _both_paths(output: Path | None, source_file: Path) -> tuple[Path, Path]:
    """Derive JSON and Markdown output paths for ``--format both``."""
    if output is not None:
        base = output.with_suffix("")
        return base.with_suffix(".json"), base.with_suffix(".md")
    stem = source_file.with_suffix("")
    return stem.with_suffix(".citeguard.json"), stem.with_suffix(".citeguard.md")


def _write_json(payload: dict[str, object], output: Path | None) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if output is None:
        click.echo(serialized)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{serialized}\n", encoding="utf-8")
    console.print(f"[green]Created {output}[/green]")


def _write_text(text: str, output: Path | None) -> None:
    if output is None:
        click.echo(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    console.print(f"[green]Created {output}[/green]")


def _validate_parsed(parsed: ParsedDocument) -> None:
    """Fail fast when the document has no content."""
    if not parsed.paragraphs:
        raise click.ClickException(
            "The document is empty or contains only whitespace."
        )


def _extract_claims_hybrid(
    parsed: ParsedDocument,
    *,
    max_claims: int | None = None,
    severity: str | None = None,
    verbose: bool = False,
) -> list[Claim]:
    """Extract claims using LLM when available, falling back to deterministic rules."""

    claims: list[Claim] = []
    limit: int | float = max_claims if max_claims is not None else float("inf")

    for index, paragraph in enumerate(parsed.paragraphs):
        if len(claims) >= limit:
            break
        if parsed.bibliography_start_index is not None and index >= parsed.bibliography_start_index:
            continue
        paragraph_citations = [
            c for c in parsed.citations if c.paragraph_index == index
        ]

        llm_claims = extract_claims_with_llm(
            paragraph, index, paragraph_citations
        )
        if llm_claims:
            if verbose:
                n = len(llm_claims)
                console.print(f"[dim]LLM extracted {n} claims from paragraph {index + 1}.[/dim]")
            claims.extend(llm_claims)
        else:
            det_claims = extract_claims(
                [paragraph], parsed.citations,
                bibliography_start=parsed.bibliography_start_index,
                max_claims=None,
            )
            claims.extend(det_claims)

    if max_claims and len(claims) > max_claims:
        claims = claims[:max_claims]
    if severity:
        claims = [c for c in claims if c.severity.value == severity]
    return claims


def _load_similarity_corpus(
    corpus_path: Path | None, license_str: str | None = None
) -> SimilarityIndex:
    """Load and fingerprint a similarity corpus.

    Returns a ``SimilarityIndex`` holding corpus entries, fingerprints,
    and TF-IDF state.
    """
    if not corpus_path:
        return SimilarityIndex()
        
    import sys

    from .corpus import deduplicate_entries, ingest_directory, ingest_file
    from .similarity.fingerprint import generate_shingles, winnow
    from .similarity.models import Fingerprint as _Fingerprint
    
    print(f"Loading corpus from {corpus_path}...", file=sys.stderr)
    if corpus_path.is_dir():
        docs = ingest_directory(corpus_path, recursive=True, license_str=license_str)
    else:
        docs = [ingest_file(corpus_path, license_str=license_str)]
        
    all_entries = []
    for doc in docs:
        all_entries.extend(doc.entries)
        
    deduped = deduplicate_entries(all_entries)
    
    corpus_entries = []
    for entry in deduped:
        shingles = generate_shingles(entry.normalized_text)
        points = winnow(shingles)
        fp = _Fingerprint(points=points, doc_id=entry.doc_id)
        corpus_entries.append(
            (entry.doc_id, entry.normalized_text, fp, entry.metadata, entry.entry_index, entry)
        )
        
    print(f"Corpus loaded: {len(docs)} docs, {len(deduped)} unique segments", file=sys.stderr)
    return SimilarityIndex.build(corpus_entries)


@main.command("similarity")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--corpus",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to corpus file or directory for comparison.",
)
@click.option(
    "--corpus-license",
    default=None,
    help="Explicit corpus license (e.g., CC0, CC-BY).",
)
@click.option(
    "--threshold",
    type=click.FloatRange(0.0, 1.0),
    default=0.7,
    help="Similarity threshold for match classification.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["terminal", "json", "md", "both"]),
    default="terminal",
    show_default=True,
)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--show-sentences", is_flag=True, help="Show sentence-level details.")
def similarity_command(
    file: Path,
    corpus: Path | None,
    corpus_license: str | None,
    threshold: float,
    output_format: str,
    output: Path | None,
    show_sentences: bool,
) -> None:
    """Analyze document similarity against a corpus or built-in index.

    Compares document sentences for exact overlap, lexical similarity,
    and attribution risk, producing a similarity report.
    """
    # Parse the document using the enriched extractor
    from .extractor import parse_enriched_document
    from .similarity.engine import SimilarityEngine
    from .similarity.models import SimilarityConfig
    
    # Corpus ingestion
    index = _load_similarity_corpus(corpus, corpus_license)

    enriched = parse_enriched_document(file)

    # Build similarity engine
    config = SimilarityConfig(exact_threshold=threshold)
    engine = SimilarityEngine(config=config)

    # Run similarity analysis
    result = engine.analyze_document(
        sentences=enriched.sentences,
        index=index,
        bibliography_entries=enriched.bibliography_entries,
    )

    # Output results
    if output_format == "json":
        from .cli import _write_json
        _write_json(similarity_report(result), output)
    elif output_format == "md":
        from .cli import _write_text
        report_text = _similarity_to_markdown(result, show_sentences)
        _write_text(report_text, output)
    elif output_format == "both":
        from .cli import _both_paths
        json_path, md_path = _both_paths(output, file)
        from .cli import _write_json, _write_text
        _write_json(similarity_report(result), json_path)
        report_text = _similarity_to_markdown(result, show_sentences)
        _write_text(report_text, md_path)
    else:
        _print_similarity_terminal(result, show_sentences)


def _similarity_to_markdown(result: SimilarityEngineResult, show_sentences: bool) -> str:
    """Convert similarity result to markdown format."""
    lines = [
        "# Citeguard Similarity Report\n",
        f"- **Overall similarity**: {result.overall_similarity_pct:.1f}%\n",
        f"- **High risk**: {result.high_risk_count}, **Medium risk**: {result.medium_risk_count}\n",
        f"- **Matched sentences**: {result.matched_sentences}/{result.total_sentences}\n",
        f"- **Unique matched chars**: {result.unique_matched_chars}/{result.eligible_chars}\n",
    ]

    if show_sentences:
        lines.append("\n## Sentence-level Details\n")
        for r in result.results:
            lines.append(f"### Sentence {r.sentence.sentence_index + 1}")
            lines.append(f"*Original*: {r.sentence.text[:80]}...")
            lines.append(f"*Normalized*: {r.sentence.normalized_text[:80]}...")
            if r.best_match:
                lines.append(f"- **Best match**: {r.best_match.source_text[:60]}...")
                lines.append(f"  - Exact overlap: {r.best_match.exact_overlap:.2f}")
                lines.append(f"  - Lexical similarity: {r.best_match.lexical_similarity:.2f}")
                lines.append(f"  - Combined score: {r.best_match.combined_score:.2f}")
                lines.append(f"  - Attribution risk: {r.attribution_risk.value}")
                lines.append(f"  - Reason: {r.attribution_reason}")
            else:
                lines.append("- No matches found")
            lines.append("")

    lines.append("## Summary\n")
    lines.append(f"- Overall similarity: {result.overall_similarity_pct:.1f}%\n")
    lines.append(f"- High risk matches: {result.high_risk_count}\n")
    lines.append(f"- Medium risk matches: {result.medium_risk_count}\n")
    lines.append(f"- Matched sentences: {result.matched_sentences}/{result.total_sentences}\n")
    matched_chars = result.unique_matched_chars
    eligible = result.eligible_chars
    lines.append(f"- Unique matched characters: {matched_chars}/{eligible}\n")
    return "".join(lines)


def _print_similarity_terminal(result: SimilarityEngineResult, show_sentences: bool) -> None:
    """Print similarity results to terminal."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(title="Citeguard Similarity Analysis")
    table.add_column("Metric", justify="right")
    table.add_column("Value")

    table.add_row("Overall similarity", f"{result.overall_similarity_pct:.1f}%")
    table.add_row("High risk matches", str(result.high_risk_count))
    table.add_row("Medium risk matches", str(result.medium_risk_count))
    table.add_row("Matched sentences", f"{result.matched_sentences}/{result.total_sentences}")
    table.add_row("Unique matched chars", f"{result.unique_matched_chars}/{result.eligible_chars}")

    console.print(table)

    if show_sentences:
        console.print("\n## Sentence-level Details")
        for r in result.results:
            console.print(f"\n**Sentence {r.sentence.sentence_index + 1}**")
            console.print(f"*: {r.sentence.text[:60]}...*")
            if r.best_match:
                console.print(f"- Best match: {r.best_match.source_text[:50]}...")
                exact = r.best_match.exact_overlap
                lex = r.best_match.lexical_similarity
                console.print(f"  Exact: {exact:.2f}, Lexical: {lex:.2f}")
                console.print(f"  Risk: {r.attribution_risk.value}")
            else:
                console.print("- No matches")


if __name__ == "__main__":
    main()
