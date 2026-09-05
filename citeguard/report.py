"""Terminal, Markdown, and JSON report formatters."""

from __future__ import annotations

from typing import Any

from .bibliography import bibliography_issues
from .linking import link_citations_to_contexts
from .models import (
    Claim,
    Evidence,
    ParsedDocument,
    ReferenceVerification,
    VerificationResult,
)
from .scoring import AuditMetrics

SCHEMA_VERSION = "1"


def inspection_report(parsed: ParsedDocument) -> dict[str, Any]:
    issues = bibliography_issues(parsed.citations, parsed.bibliography_entries)
    contexts = link_citations_to_contexts(
        parsed.paragraphs, parsed.citations, parsed.bibliography_entries
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "document": parsed.path,
        "summary": {
            "paragraphs": len(parsed.paragraphs),
            "detected_citations": len(parsed.citations),
            "bibliography_entries": len(parsed.bibliography_entries),
            "bibliography_issues": len(issues),
        },
        "citations": [
            {
                "raw_text": context.citation.raw_text,
                "authors": context.citation.authors,
                "year": context.citation.year,
                "no_date": context.citation.no_date,
                "doi": context.citation.doi,
                "numbered_ref": context.citation.numbered_ref,
                "paragraph": context.citation.paragraph_index + 1,
                "char_offset": context.citation.char_offset,
                "sentence": context.sentence,
                "bibliography_entry_indexes": [
                    index + 1 for index in context.bibliography_entry_indexes
                ],
            }
            for context in contexts
        ],
        "bibliography": [
            {
                "index": index + 1,
                "raw_text": entry.raw_text,
                "authors": entry.authors,
                "year": entry.year,
                "no_date": entry.no_date,
                "title": entry.title,
                "doi": entry.doi,
                "numbered_ref": entry.numbered_ref,
            }
            for index, entry in enumerate(parsed.bibliography_entries)
        ],
        "issues": [
            {
                "kind": issue.kind.value,
                "detail": issue.detail,
                "paragraph": (
                    issue.paragraph_index + 1 if issue.paragraph_index is not None else None
                ),
            }
            for issue in issues
        ],
    }


def verification_report(
    parsed: ParsedDocument, results: list[ReferenceVerification]
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status.value] = status_counts.get(result.status.value, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "document": parsed.path,
        "provider": "crossref",
        "privacy": "Bibliography metadata was sent to Crossref; document prose was not sent.",
        "summary": {
            "bibliography_entries": len(results),
            "status_counts": status_counts,
        },
        "results": [_verification_item(result) for result in results],
    }


def check_report(
    parsed: ParsedDocument,
    verification_results: list[ReferenceVerification],
    claims: list[Claim],
    verification_for_claims: list[VerificationResult],
    metrics: AuditMetrics,
    bib_issues: list[Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "document": parsed.path,
        "privacy": (
            "Academic providers may receive generated claim search queries and bibliography "
            "metadata. When Anthropic integration is configured, paragraphs and claim-source "
            "metadata may also be sent. Verification does not establish claim support."
        ),
        "summary": {
            "health_score": metrics.health_score,
            "total_claims": metrics.total_claims,
            "claims_requiring_citations": metrics.claims_requiring_citations,
            "cited_claims": metrics.cited_claims,
            "verified_citations": metrics.verified_citations,
            "weak_matches": metrics.weak_matches,
            "unresolved_citations": metrics.unresolved_citations,
            "uncited_high_severity_claims": metrics.uncited_high_severity_claims,
            "contradictions": metrics.contradictions,
            "bibliography_issues": metrics.bibliography_issues,
            "citation_coverage": metrics.citation_coverage,
            "verification_ratio": metrics.verification_ratio,
            "support_ratio": metrics.support_ratio,
            "bibliography_consistency": metrics.bibliography_consistency,
            "evidence_coverage": metrics.evidence_coverage,
        },
        "claims": [_claim_item(c) for c in claims],
        "verification_results": [_verification_item(r) for r in verification_results],
        "bibliography_issues": [
            {
                "kind": issue.kind.value,
                "detail": issue.detail,
                "paragraph": (
                    issue.paragraph_index + 1 if issue.paragraph_index is not None else None
                ),
            }
            for issue in bib_issues
        ],
        "priority_review": [
            _verification_result_item(r) for r in verification_for_claims[:20]
        ],
    }


def suggest_report(
    parsed: ParsedDocument,
    claims: list[Claim],
    suggestions: list[VerificationResult],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "document": parsed.path,
        "privacy": (
            "Academic providers receive claim search queries. When Anthropic integration is "
            "configured, paragraphs and claim-source metadata may also be sent."
        ),
        "summary": {
            "total_claims": len(claims),
            "uncited_claims": sum(1 for c in claims if not c.has_existing_citation),
            "suggestions_provided": sum(
                1 for s in suggestions if s.suggestions
            ),
        },
        "claims": [_claim_item(c) for c in claims],
        "suggestions": [_suggestion_item(s) for s in suggestions if s.suggestions],
    }


def _verification_item(result: ReferenceVerification) -> dict[str, Any]:
    candidate = result.candidate
    scores = result.scores
    return {
        "bibliography_entry_index": result.entry_index + 1,
        "bibliography_entry": result.entry.raw_text,
        "query": result.query,
        "status": result.status.value,
        "scores": (
            {
                "author": scores.author,
                "year": scores.year,
                "title": scores.title,
                "doi": scores.doi,
                "overall": scores.overall,
            }
            if scores
            else None
        ),
        "candidate": (
            {
                "title": candidate.title,
                "authors": candidate.authors,
                "year": candidate.year,
                "venue": candidate.venue,
                "doi": candidate.doi,
                "url": candidate.url,
                "source_api": candidate.source_api,
            }
            if candidate
            else None
        ),
        "warnings": result.warnings,
    }


def _evidence_item(evidence: Evidence) -> dict[str, Any]:
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


def _claim_item(claim: Claim) -> dict[str, Any]:
    return {
        "text": claim.text,
        "search_query": claim.search_query,
        "claim_type": claim.claim_type.value,
        "severity": claim.severity.value,
        "paragraph": claim.paragraph_index + 1,
        "has_existing_citation": claim.has_existing_citation,
    }


def _suggestion_item(result: VerificationResult) -> dict[str, Any]:
    return {
        "claim_text": result.claim.text,
        "claim_type": result.claim.claim_type.value,
        "severity": result.claim.severity.value,
        "paragraph": result.claim.paragraph_index + 1,
        "suggestions": [
            {
                "title": s.candidate.title,
                "authors": s.candidate.authors,
                "year": s.candidate.year,
                "doi": s.candidate.doi,
                "url": s.candidate.url,
                "source_api": s.candidate.source_api,
                "metadata_match_score": s.metadata_match_score,
                "claim_support_score": s.claim_support_score,
                "overall_confidence": s.overall_confidence,
                "verdict": s.verdict.value,
                "evidence": [_evidence_item(e) for e in s.evidence],
            }
            for s in result.suggestions
        ],
    }


def _verification_result_item(result: VerificationResult) -> dict[str, Any]:
    matched = result.matched
    return {
        "claim_text": result.claim.text,
        "claim_type": result.claim.claim_type.value,
        "severity": result.claim.severity.value,
        "paragraph": result.claim.paragraph_index + 1,
        "status": result.status.value,
        "matched": (
            {
                "title": matched.candidate.title,
                "authors": matched.candidate.authors,
                "year": matched.candidate.year,
                "doi": matched.candidate.doi,
                "overall_confidence": matched.overall_confidence,
                "verdict": matched.verdict.value,
                "evidence": [_evidence_item(e) for e in matched.evidence],
            }
            if matched
            else None
        ),
        "suggestions_count": len(result.suggestions),
        "warnings": result.warnings,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def markdown_check_report(
    parsed: ParsedDocument,
    verification_results: list[ReferenceVerification],
    claims: list[Claim],
    verification_for_claims: list[VerificationResult],
    metrics: AuditMetrics,
    bib_issues: list[Any],
) -> str:
    lines: list[str] = []
    _md = lines.append

    _md("# citeguard Check Report\n")
    _md(f"**Document:** `{parsed.path}`\n")
    _md("")

    _md("## Citation Health Score\n")
    _md(f"**{metrics.health_score}/100**\n")
    _md("")
    _md("> The Citation Health Score is a review-prioritization heuristic. It does **not**")
    _md("> measure scientific or academic correctness.\n")

    _md("## Summary\n")
    _md(f"- Total claims detected: **{metrics.total_claims}**")
    _md(f"- Claims requiring citations: **{metrics.claims_requiring_citations}**")
    _md(f"- Cited claims: **{metrics.cited_claims}**")
    _md(f"- Verified citations: **{metrics.verified_citations}**")
    _md(f"- Unresolved citations: **{metrics.unresolved_citations}**")
    _md(f"- Weak matches: **{metrics.weak_matches}**")
    _md(f"- Uncited high-severity claims: **{metrics.uncited_high_severity_claims}**")
    _md(f"- Contradictions: **{metrics.contradictions}**")
    _md(f"- Bibliography issues: **{metrics.bibliography_issues}**")
    _md("")

    _md("## Ratios\n")
    _md(f"- Citation coverage: **{metrics.citation_coverage:.1%}**")
    _md(f"- Verification ratio: **{metrics.verification_ratio:.1%}**")
    _md(f"- Support ratio: **{metrics.support_ratio:.1%}**")
    _md(f"- Bibliography consistency: **{metrics.bibliography_consistency:.1%}**")
    _md("")

    if claims:
        _md("## Detected Claims\n")
        _md("| # | Claim | Type | Severity | Cited |")
        _md("|---|-------|------|----------|-------|")
        for i, claim in enumerate(claims, 1):
            cited = "Yes" if claim.has_existing_citation else "No"
            ct = claim.claim_type.value
            sev = claim.severity.value
            _md(f"| {i} | {claim.text} | {ct} | {sev} | {cited} |")
        _md("")

    if verification_results:
        _md("## Bibliography Verification\n")
        _md("| # | Entry | Status | Score | Candidate |")
        _md("|---|-------|--------|-------|-----------|")
        for result in verification_results:
            entry_label = (
                result.entry.title or result.entry.authors
                or result.entry.raw_text[:50]
            )
            score = str(result.scores.overall) if result.scores else "-"
            cand_title = result.candidate.title[:40] if result.candidate else "-"
            idx = result.entry_index + 1
            status = result.status.value
            _md(f"| {idx} | {entry_label} | {status} | {score} | {cand_title} |")
        _md("")

    if bib_issues:
        _md("## Bibliography Issues\n")
        for issue in bib_issues:
            _md(f"- **{issue.kind.value}**: {issue.detail}")
        _md("")

    if verification_for_claims:
        _md("## Priority Review List\n")
        _md("Items below are sorted by risk priority (highest first).\n")
        for result in verification_for_claims:
            verdict = result.matched.verdict.value if result.matched else "n/a"
            confidence = result.matched.overall_confidence if result.matched else 0
            sev = result.claim.severity.value.upper()
            txt = result.claim.text
            _md(f"- **{sev}** (confidence: {confidence})"
                f" - {txt} [verdict: {verdict}]")
        _md("")

    # Evidence section for claims with matched sources
    if verification_for_claims:
        _md("## Evidence\n")
        for result in verification_for_claims:
            matched = result.matched
            if matched and matched.evidence:
                sev = result.claim.severity.value.upper()
                txt = result.claim.text
                _md(f"### Claim: {txt}\n")
                _md(f"- Severity: **{sev}** | Verdict: **{matched.verdict.value}**")
                _md(f"- Confidence: **{matched.overall_confidence}/100**\n")
                for i, ev in enumerate(matched.evidence[:3], 1):
                    _md(f"**Evidence {i}** (relevance: {ev.lexical_score}/100)")
                    _md(f"> {ev.text}")
                    _md(f"> *Source: {ev.source_title}*\n")
        _md("")

    _md("---\n")
    _md("*Generated by citeguard. This report is a review-prioritization aid, "
        "not an academic judgment.*")

    return "\n".join(lines)


def markdown_suggest_report(
    parsed: ParsedDocument,
    claims: list[Claim],
    suggestions: list[VerificationResult],
) -> str:
    lines: list[str] = []
    _md = lines.append

    _md("# citeguard Suggest Report\n")
    _md(f"**Document:** `{parsed.path}`\n")

    uncited = [c for c in claims if not c.has_existing_citation]
    _md("## Summary\n")
    _md(f"- Total claims: **{len(claims)}**")
    _md(f"- Uncited claims: **{len(uncited)}**")
    _md(f"- Claims with suggestions: **{sum(1 for s in suggestions if s.suggestions)}**")
    _md("")

    if suggestions:
        _md("## Source Suggestions\n")
        for result in suggestions:
            if not result.suggestions:
                continue
            _md(f"### Claim: {result.claim.text}\n")
            ct = result.claim.claim_type.value
            sev = result.claim.severity.value
            para = result.claim.paragraph_index + 1
            _md(f"- Type: `{ct}` | Severity: `{sev}` | Paragraph: {para}\n")
            _md("| Source | Authors | Year | DOI | Score | Verdict |")
            _md("|--------|---------|------|-----|-------|---------|")
            for s in result.suggestions:
                authors_short = ", ".join(s.candidate.authors[:2])
                if len(s.candidate.authors) > 2:
                    authors_short += " et al."
                doi = s.candidate.doi or "-"
                year = s.candidate.year or "-"
                conf = s.overall_confidence
                verdict = s.verdict.value
                _md(
                    f"| {s.candidate.title[:50]} | {authors_short}"
                    f" | {year} | {doi} | {conf} | {verdict} |"
                )
            _md("")

    _md("---\n")
    _md(
        "*Generated by citeguard. Suggestions require human review"
        " and do not confirm source support.*"
    )

    return "\n".join(lines)


def markdown_verification_report(
    parsed: ParsedDocument, results: list[ReferenceVerification]
) -> str:
    lines: list[str] = []
    _md = lines.append

    _md("# citeguard Verification Report\n")
    _md(f"**Document:** `{parsed.path}`\n")
    _md("Only bibliography metadata was sent to Crossref. Document prose was not sent.\n")

    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status.value] = status_counts.get(result.status.value, 0) + 1

    _md("## Summary\n")
    for status, count in sorted(status_counts.items()):
        _md(f"- {status}: **{count}**")
    _md("")

    if results:
        _md("## Results\n")
        _md("| # | Entry | Status | Score | Candidate |")
        _md("|---|-------|--------|-------|-----------|")
        for result in results:
            entry_label = (
                result.entry.title or result.entry.authors
                or result.entry.raw_text[:50]
            )
            score = str(result.scores.overall) if result.scores else "-"
            cand = result.candidate.title[:40] if result.candidate else "-"
            idx = result.entry_index + 1
            status = result.status.value
            _md(f"| {idx} | {entry_label} | {status} | {score} | {cand} |")
        _md("")

    _md("---\n")
    _md("*Generated by citeguard. Verification does not establish claim support.*")

    return "\n".join(lines)


def inspection_report_markdown(parsed: ParsedDocument) -> str:
    """Markdown version of the inspection report for ``--format both``."""
    issues = bibliography_issues(parsed.citations, parsed.bibliography_entries)
    contexts = link_citations_to_contexts(
        parsed.paragraphs, parsed.citations, parsed.bibliography_entries
    )
    lines: list[str] = []
    _md = lines.append

    _md("# citeguard Inspection Report\n")
    _md(f"**Document:** `{parsed.path}`\n")

    _md("## Summary\n")
    _md(f"- Paragraphs: **{len(parsed.paragraphs)}**")
    _md(f"- Detected citations: **{len(parsed.citations)}**")
    _md(f"- Bibliography entries: **{len(parsed.bibliography_entries)}**")
    _md(f"- Bibliography issues: **{len(issues)}**")
    _md("")

    if contexts:
        _md("## Detected Citations\n")
        _md("| # | Citation | Paragraph | Char Offset | Bibliography Match |")
        _md("|---|----------|-----------|-------------|---------------------|")
        for i, ctx in enumerate(contexts, 1):
            bib_idx = ctx.bibliography_entry_indexes
            bib_str = ", ".join(str(idx + 1) for idx in bib_idx) if bib_idx else "-"
            _md(
                f"| {i} | {ctx.citation.raw_text} "
                f"| {ctx.citation.paragraph_index + 1} "
                f"| {ctx.citation.char_offset} "
                f"| {bib_str} |"
            )
        _md("")

    if parsed.bibliography_entries:
        _md("## Bibliography Entries\n")
        _md("| # | Authors | Year | Title | DOI |")
        _md("|---|---------|------|-------|-----|")
        for i, entry in enumerate(parsed.bibliography_entries, 1):
            authors = entry.authors or "-"
            year = str(entry.year) if entry.year else ("n.d." if entry.no_date else "-")
            title = (entry.title or "-")[:50]
            doi = entry.doi or "-"
            _md(f"| {i} | {authors} | {year} | {title} | {doi} |")
        _md("")

    if issues:
        _md("## Bibliography Issues\n")
        for issue in issues:
            _md(f"- **{issue.kind.value}**: {issue.detail}")
        _md("")

    _md("---\n")
    _md("*Generated by citeguard.*")

    return "\n".join(lines)
