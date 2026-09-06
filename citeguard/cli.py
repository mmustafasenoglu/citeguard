"""Click-based command-line interface for citeguard."""

from __future__ import annotations

import json
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
from .config import DEFAULT_MAX_RESULTS, DEFAULT_THRESHOLD, Settings
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
    suggest_report,
    verification_report,
)
from .retrieval import RetrievalEngine
from .scoring import AuditMetrics, compute_audit_metrics, overall_confidence, priority_list
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
) -> None:
    """Run a full citation audit: extract claims, verify references, compute health score."""
    parsed = parse_document(file)
    _validate_parsed(parsed)
    settings = Settings.from_env()
    cache = FileCache(settings.cache_dir)

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
    )

    if output_format == "json":
        _write_json(report, output)
        exit_code = EXIT_FINDINGS if metrics.health_score < 80 else EXIT_SUCCESS
        raise SystemExit(exit_code)
    if output_format == "md":
        report_text = markdown_check_report(
            parsed, bib_results, claims, sorted_claims, metrics, bib_issues
        )
        _write_text(report_text, output)
        exit_code = EXIT_FINDINGS if metrics.health_score < 80 else EXIT_SUCCESS
        raise SystemExit(exit_code)
    if output_format == "both":
        json_path, md_path = _both_paths(output, file)
        _write_json(report, json_path)
        report_text = markdown_check_report(
            parsed, bib_results, claims, sorted_claims, metrics, bib_issues
        )
        _write_text(report_text, md_path)
        exit_code = EXIT_FINDINGS if metrics.health_score < 80 else EXIT_SUCCESS
        raise SystemExit(exit_code)

    show_ev = show_evidence or verbose
    _print_check_terminal(metrics, claims, sorted_claims, bib_issues, verbose, show_ev)

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


if __name__ == "__main__":
    main()
