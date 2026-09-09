"""Terminal, JSON, and Markdown plagiarism-review reports."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .models import SCHEMA_VERSION, PlagiarismMatch, PlagiarismResult


def _match_dict(match: PlagiarismMatch) -> dict[str, Any]:
    payload = asdict(match)
    payload["attribution_status"] = match.attribution_status.value
    payload["severity"] = match.severity.value
    return payload


def plagiarism_report(result: PlagiarismResult) -> dict[str, Any]:
    """Return the deterministic versioned machine-readable report."""
    config = asdict(result.configuration)
    return {
        "schema_version": SCHEMA_VERSION,
        "document": {
            "path": str(result.document),
            "supported_format": result.document.suffix.lower().lstrip("."),
            "offset_units": "Unicode code points; end offsets are exclusive",
        },
        "summary": asdict(result.summary),
        "sources": [
            asdict(source) for source in sorted(result.sources, key=lambda item: item.source_id)
        ],
        "source_contributions": [asdict(item) for item in result.source_contributions],
        "matches": [_match_dict(match) for match in result.matches],
        "exclusions": {
            "quotes_excluded_from_review": not result.configuration.include_quotes,
            "bibliography_excluded_from_review": (not result.configuration.include_bibliography),
            "minimum_match_words": result.configuration.min_match_words,
            "common_phrases_excluded_from_review": True,
        },
        "configuration": config,
        "corpus_limitations": (
            "Only textual content in the supplied sources and corpora was compared. "
            "This is not a Turnitin score and does not reproduce a proprietary corpus."
        ),
        "warnings": result.warnings,
    }


def plagiarism_markdown(result: PlagiarismResult, *, show_matches: bool = True) -> str:
    """Render a complete, auditable Markdown report."""
    summary = result.summary
    lines = [
        "# CiteGuard Plagiarism Review",
        "",
        f"- Document: `{result.document}`",
        f"- Raw similarity: **{summary.raw_similarity_percent:.2f}%**",
        f"- Review-relevant similarity: **{summary.review_similarity_percent:.2f}%**",
        f"- Exact overlap: {summary.exact_similarity_percent:.2f}%",
        f"- Near-exact / lexical overlap: {summary.lexical_similarity_percent:.2f}%",
        f"- Semantic overlap: {summary.semantic_similarity_percent:.2f}%",
        f"- Properly attributed or cited: {summary.attributed_similarity_percent:.2f}%",
        f"- Unattributed: {summary.unattributed_similarity_percent:.2f}%",
        f"- Sources compared: {len(result.sources)}",
        f"- Matches: {len(result.matches)}",
        "",
        "## Score methodology",
        "",
        "Percentages use unique covered document words; overlapping source matches are counted "
        "once. Review similarity excludes common phrases and, by default, quotations and the "
        "bibliography. Semantic coverage represents a matched passage, not a word-for-word span.",
        "",
        "## Source contributions",
        "",
    ]
    if result.source_contributions:
        for index, contribution in enumerate(result.source_contributions, 1):
            lines.append(
                f"{index}. **{contribution.title}** — {contribution.percent:.2f}% unique "
                f"contribution ({contribution.match_count} matches)"
            )
    else:
        lines.append("No review-relevant source contribution was measured.")
    if show_matches:
        lines.extend(["", "## Match details", ""])
        for match in result.matches:
            lines.extend(
                [
                    f"### {match.severity.value.upper()} · {match.match_type}",
                    "",
                    f"- Match ID: `{match.match_id}`",
                    f"- Document: paragraph {match.document_paragraph + 1}, characters "
                    f"{match.document_paragraph_start}–{match.document_paragraph_end} "
                    f"(document-wide {match.document_start}–{match.document_end})",
                    f"- Source: `{match.source_id}`",
                    f"- Attribution: `{match.attribution_status.value}`",
                    f"- Exact / lexical / semantic: {match.exact_score:.3f} / "
                    f"{match.lexical_score:.3f} / {match.semantic_score:.3f}",
                    "",
                    f"> {match.document_text}",
                    "",
                    f"Similar source passage: “{match.source_text}”",
                    "",
                    f"{match.explanation} {match.recommendation}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Exclusions and limitations",
            "",
            f"- Quotations included in review score: {result.configuration.include_quotes}",
            f"- Bibliography included in review score: {result.configuration.include_bibliography}",
            f"- Minimum match length: {result.configuration.min_match_words} words",
            "- CiteGuard measures overlap only against supplied textual sources and corpora.",
            "- This is not a Turnitin score and does not match or predict one.",
        ]
    )
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines) + "\n"


def plagiarism_terminal(result: PlagiarismResult, *, show_matches: bool = True) -> str:
    """Render a compact professional terminal report without terminal control codes."""
    summary = result.summary
    lines = [
        "CiteGuard Plagiarism Review",
        "===========================",
        f"File: {result.document}",
        "",
        f"Raw similarity              {summary.raw_similarity_percent:7.2f}%",
        f"Review-relevant similarity  {summary.review_similarity_percent:7.2f}%",
        f"Exact overlap                {summary.exact_similarity_percent:7.2f}%",
        f"Near-exact / lexical         {summary.lexical_similarity_percent:7.2f}%",
        f"Semantic overlap             {summary.semantic_similarity_percent:7.2f}%",
        f"Properly attributed          {summary.attributed_similarity_percent:7.2f}%",
        f"Unattributed                 {summary.unattributed_similarity_percent:7.2f}%",
        "",
        f"Sources: {len(result.sources)}    Matches: {len(result.matches)}",
        "",
        "Source contributions",
        "--------------------",
    ]
    lines.extend(
        f"{index}. {item.title}: {item.percent:.2f}%"
        for index, item in enumerate(result.source_contributions, 1)
    )
    if show_matches and result.matches:
        lines.extend(["", "High-priority matches", "---------------------"])
        for match in result.matches:
            if match.severity.value not in {"critical", "high"}:
                continue
            lines.extend(
                [
                    "",
                    f"{match.severity.value.upper()} · {match.match_type}",
                    f"Document: paragraph {match.document_paragraph + 1}",
                    f"“{match.document_text}”",
                    f"Source: {match.source_id}",
                    f"Attribution: {match.attribution_status.value}",
                    f"Recommendation: {match.recommendation}",
                ]
            )
    lines.extend(
        [
            "",
            "CiteGuard compares only supplied textual sources/corpora. This is not a Turnitin "
            "score and does not predict one.",
        ]
    )
    return "\n".join(lines)
