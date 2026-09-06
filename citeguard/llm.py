"""Optional LLM-backed claim extraction and source matching.

This module wraps Anthropic's Messages API using only stdlib (no SDK dependency).
When ANTHROPIC_API_KEY is not set, all functions return ``None`` so callers
can fall back to the deterministic baseline.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from .models import Claim, ClaimType, ExistingCitation, Severity

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_TIMEOUT = 30

CLAIM_TYPES = [ct.value for ct in ClaimType]
SEVERITIES = [s.value for s in Severity]

_SYSTEM_PROMPT = (
    "You are an academic citation assistant. "
    "You extract citation-worthy claims from academic text. "
    "Return ONLY valid JSON, no markdown fences."
)

_EXTRACT_CLAIMS_PROMPT = """\
Analyze the following paragraph and extract citation-worthy claims.

Paragraph:
\"\"\"
{paragraph}
\"\"\"

Return a JSON array of objects. Each object must have:
- "text": the claim excerpt (max 25 words)
- "search_query": optimized search query for finding a source
- "claim_type": one of {claim_types}
- "severity": one of {severities}
- "has_existing_citation": true if the claim already has a citation in its sentence

Rules:
- Focus on externally verifiable statements: statistics, research findings,
  direct quotations, historical facts, causal claims, comparative claims.
- Skip opinions, instructions, headings, and very short sentences.
- Return [] if no citation-worthy claims are found.
- Return ONLY the JSON array, nothing else.
"""

_MATCH_SOURCE_PROMPT = """\
Evaluate whether the given academic source supports the claim.

Claim: "{claim_text}"
Claim type: {claim_type}

Source:
  Title: {source_title}
  Authors: {source_authors}
  Year: {source_year}
  Abstract: {source_abstract}

Return a JSON object with:
- "metadata_match_score": 0-100 (how well the metadata aligns)
- "claim_support_score": 0-100 (how well the source content supports the claim)
- "verdict": one of "supported", "partially_supported", "contradicted",
  "unrelated", "insufficient_information"
- "reasoning": concise one-sentence explanation

Return ONLY the JSON object, nothing else.
"""


def _api_key() -> str | None:
    return os.getenv("ANTHROPIC_API_KEY")


def _call_anthropic(
    api_key: str,
    system: str,
    user_message: str,
    *,
    model: str = _DEFAULT_MODEL,
    max_tokens: int = 2048,
    timeout: float = _DEFAULT_TIMEOUT,
) -> str | None:
    """Send a Messages API request and return the text content, or None on failure."""
    payload = json.dumps(
        {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user_message}],
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        ANTHROPIC_ENDPOINT,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None

    content = data.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        text = content[0].get("text", "")
        if isinstance(text, str):
            return text
    return None


def _parse_json_response(text: str) -> Any:
    """Best-effort JSON extraction from LLM text that may include fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip().startswith("```"):
                end = i
                break
        cleaned = "\n".join(lines[start:end]).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        first = cleaned.find(start_char)
        last = cleaned.rfind(end_char)
        if first != -1 and last > first:
            try:
                return json.loads(cleaned[first : last + 1])
            except json.JSONDecodeError:
                pass
    return None


def extract_claims_with_llm(
    paragraph: str,
    paragraph_index: int,
    paragraph_citations: list[ExistingCitation],
    *,
    model: str = _DEFAULT_MODEL,
    timeout: float = _DEFAULT_TIMEOUT,
) -> list[Claim] | None:
    """Use Anthropic to extract claims from a single paragraph.

    Returns ``None`` when the API key is absent or the call fails,
    so callers can fall back to the deterministic extractor.
    """
    api_key = _api_key()
    if not api_key:
        return None

    prompt = _EXTRACT_CLAIMS_PROMPT.format(
        paragraph=paragraph,
        claim_types=", ".join(CLAIM_TYPES),
        severities=", ".join(SEVERITIES),
    )
    raw = _call_anthropic(
        api_key, _SYSTEM_PROMPT, prompt, model=model, timeout=timeout
    )
    if not raw:
        return None

    items = _parse_json_response(raw)
    if not isinstance(items, list):
        return None

    citation_texts = {c.raw_text for c in paragraph_citations}
    _sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    claims: list[Claim] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        try:
            claim_type = ClaimType(str(item.get("claim_type", "general_fact")))
        except ValueError:
            claim_type = ClaimType.GENERAL_FACT
        try:
            severity = Severity(str(item.get("severity", "low")))
        except ValueError:
            severity = Severity.LOW
        search_query = str(item.get("search_query", text))[:200]

        has_citation, linked = _map_claim_to_sentence(
            text, _sentences, paragraph_citations, citation_texts,
        )

        claims.append(
            Claim(
                text=text[:200],
                search_query=search_query,
                claim_type=claim_type,
                severity=severity,
                paragraph_index=paragraph_index,
                has_existing_citation=has_citation,
                linked_citations=linked,
                linked_citation=linked[0] if linked else None,
            )
        )
    return claims


def _map_claim_to_sentence(
    claim_text: str,
    sentences: list[str],
    paragraph_citations: list[ExistingCitation],
    citation_texts: set[str],
) -> tuple[bool, list[ExistingCitation]]:
    """Map an LLM-extracted claim back to the original sentence.

    The LLM often strips citation markers from claim text.  To determine
    whether the claim originally had a citation, we find the sentence with
    the highest token overlap to the claim and check whether that sentence
    contains a citation.
    """
    claim_tokens = set(re.findall(r"[a-z0-9]+", claim_text.lower()))
    if not claim_tokens:
        return False, []

    best_overlap = 0.0
    best_sentence = ""
    for sentence in sentences:
        sent_tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        if not sent_tokens:
            continue
        overlap = len(claim_tokens & sent_tokens) / len(claim_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence

    if best_overlap < 0.3 or not best_sentence:
        return False, []

    linked = [
        c for c in paragraph_citations if c.raw_text in best_sentence
    ]
    return bool(linked), linked


def match_source_with_llm(
    claim: Claim,
    source_title: str,
    source_authors: str,
    source_year: int | None,
    source_abstract: str | None,
    *,
    model: str = _DEFAULT_MODEL,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any] | None:
    """Use Anthropic to evaluate whether a source supports a claim.

    Returns ``None`` when the API key is absent or the call fails.
    """
    api_key = _api_key()
    if not api_key:
        return None

    prompt = _MATCH_SOURCE_PROMPT.format(
        claim_text=claim.text,
        claim_type=claim.claim_type.value,
        source_title=source_title,
        source_authors=source_authors,
        source_year=source_year if source_year is not None else "unknown",
        source_abstract=source_abstract or "not available",
    )
    raw = _call_anthropic(
        api_key, _SYSTEM_PROMPT, prompt, model=model, timeout=timeout
    )
    if not raw:
        return None

    result = _parse_json_response(raw)
    if not isinstance(result, dict):
        return None
    return result
