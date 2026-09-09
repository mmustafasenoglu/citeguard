"""Remote LLM rewrite provider with evidence grounding and validation.

Transport reuses the existing resilient LLM router (retry, failover,
caching, audit) via ``_call_llm_detailed`` with ``task="rewrite"``.
No provider SDK code lives in the rewrite business logic.

Failure contract (never a silent fallback):

- no credentials/backend → UNAVAILABLE (zero network calls)
- transport failure → PROVIDER_ERROR
- model refusal → REFUSED
- unparsable or schema-invalid output → INVALID_RESPONSE
- hallucinated citation/DOI/URL/year → INVALID_RESPONSE
"""

from __future__ import annotations

import logging
import re

from .models import (
    RewriteRequest,
    RewriteResult,
    RewriteStatus,
)
from .prompts import (
    REWRITE_JSON_SCHEMA,
    REWRITE_SYSTEM_PROMPT,
    build_rewrite_prompt,
)

log = logging.getLogger(__name__)

MAX_REWRITTEN_CHARS = 1000

_DOI_RE = re.compile(r"\b10\.\d{4,}/[^\s)\"']+")
_URL_RE = re.compile(r"https?://[^\s)\"']+")
_YEAR_RE = re.compile(r"\b\d{4}\b")
_CITATION_SHAPE_RE = re.compile(r"\(([^()]{1,80}?),\s*(\d{4}[a-z]?)\)")

_TRAILING_PUNCT = ".,;:"


def _clean_token(token: str) -> str:
    """Strip sentence punctuation attached to an extracted token."""
    return token.rstrip(_TRAILING_PUNCT)

_REFUSAL_MARKERS = (
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i'm unable",
    "i am unable",
    "cannot help",
    "unable to help",
    "as an ai",
)


def _classify_transport_error(error: str) -> str:
    """Map a provider error string to a human-readable warning."""
    lowered = error.lower()
    if "timeout" in lowered or "timed out" in lowered or "deadline" in lowered:
        return f"provider timeout: {error}"
    if any(
        marker in lowered
        for marker in ("401", "unauthorized", "invalid api key", "authentication")
    ):
        return f"provider authentication failed: {error}"
    if any(
        marker in lowered
        for marker in ("429", "rate limit", "rate-limit", "too many requests")
    ):
        return f"provider rate limited: {error}"
    return f"provider error: {error}"


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def _context_years(request: RewriteRequest) -> set[str]:
    """All 4-digit years appearing anywhere in the verified context."""
    ctx = request.context
    haystacks = [ctx.claim_text, ctx.citation_raw or ""]
    if ctx.source_year is not None:
        haystacks.append(str(ctx.source_year))
    haystacks.extend(ctx.evidence_texts)
    years: set[str] = set()
    for text in haystacks:
        years.update(_YEAR_RE.findall(text))
    return years


def _context_doi_and_urls(request: RewriteRequest) -> set[str]:
    """DOI/URL tokens appearing anywhere in the verified context."""
    ctx = request.context
    haystacks = [ctx.claim_text, ctx.citation_raw or "", ctx.source_doi or ""]
    haystacks.extend(ctx.evidence_texts)
    found: set[str] = set()
    for text in haystacks:
        found.update(_clean_token(t) for t in _DOI_RE.findall(text))
        found.update(_clean_token(t) for t in _URL_RE.findall(text))
    return found


def _detect_hallucinations(
    rewritten: str, request: RewriteRequest
) -> list[str]:
    """Deterministic hallucination checks against verified context."""
    problems: list[str] = []
    allowed_tokens = _context_doi_and_urls(request)
    for raw in _DOI_RE.findall(rewritten):
        token = _clean_token(raw)
        if token not in allowed_tokens:
            problems.append(f"hallucinated DOI rejected: {token}")
    for raw in _URL_RE.findall(rewritten):
        token = _clean_token(raw)
        if token not in allowed_tokens:
            problems.append(f"hallucinated URL rejected: {token}")
    allowed_years = _context_years(request)
    for _authors, year in _CITATION_SHAPE_RE.findall(rewritten):
        digits = year[:4]
        if digits not in allowed_years:
            problems.append(f"hallucinated citation year rejected: ({year})")
    return problems


def _validate_payload(payload: object) -> dict[str, object] | None:
    """Strict schema validation.  Returns the payload or None."""
    if not isinstance(payload, dict):
        return None
    rewritten = payload.get("rewritten_text")
    reason = payload.get("reason")
    citation_preserved = payload.get("citation_preserved")
    warnings = payload.get("warnings")
    if not isinstance(rewritten, str) or not rewritten.strip():
        return None
    if len(rewritten) > MAX_REWRITTEN_CHARS:
        return None
    if not isinstance(reason, str):
        return None
    if not isinstance(citation_preserved, bool):
        return None
    if not isinstance(warnings, list) or not all(
        isinstance(w, str) for w in warnings
    ):
        return None
    return {
        "rewritten_text": rewritten.strip(),
        "reason": reason.strip(),
        "citation_preserved": citation_preserved,
        "warnings": [w for w in warnings if w.strip()],
    }


class LLMRewriteProvider:
    """Remote LLM rewrite provider over the shared LLM router."""

    def __init__(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        timeout: float = 30.0,
        max_tokens: int = 1024,
        offline: bool = False,
    ) -> None:
        self._model = model
        self._provider = provider
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._offline = offline

    def rewrite(self, request: RewriteRequest) -> RewriteResult:
        """Generate a rewrite suggestion, or a deterministic error result."""
        original = request.context.original_text
        if self._offline:
            return RewriteResult(
                original_text=original,
                rewritten_text=None,
                status=RewriteStatus.UNAVAILABLE,
                provider="none",
                model=self._model or "unknown",
                warnings=("rewrite unavailable in offline mode.",),
                evidence_used=request.context.evidence_texts,
                citation_preserved=None,
                metadata={"mode": request.mode.value, "offline": True},
            )

        from ..llm import _call_llm_detailed, _parse_json_response, _resolve
        from ..llm_backends import resolve_backend

        backend = resolve_backend(self._provider) if self._provider else None
        resolved = _resolve(backend=backend, model=self._model, task="rewrite")
        if resolved is None:
            return RewriteResult(
                original_text=original,
                rewritten_text=None,
                status=RewriteStatus.UNAVAILABLE,
                provider="none",
                model=self._model or "unknown",
                warnings=(
                    "rewrite unavailable: no LLM provider credentials configured. "
                    "Set an API key (e.g. ANTHROPIC_API_KEY) or "
                    "CITEGUARD_REWRITE_PROVIDER.",
                ),
                evidence_used=request.context.evidence_texts,
                citation_preserved=None,
                metadata={"mode": request.mode.value},
            )

        prompt = build_rewrite_prompt(
            request.context,
            request.mode,
            request.style_constraints,
            request.max_context_chars,
        )
        try:
            resp = _call_llm_detailed(
                REWRITE_SYSTEM_PROMPT,
                prompt,
                backend=backend,
                model=self._model,
                max_tokens=self._max_tokens,
                timeout=self._timeout,
                task="rewrite",
                json_schema=REWRITE_JSON_SCHEMA,
            )
        except Exception as exc:
            log.debug("Rewrite provider call failed: %s", exc)
            _, model_name = resolved
            return RewriteResult(
                original_text=original,
                rewritten_text=None,
                status=RewriteStatus.PROVIDER_ERROR,
                provider=resolved[0].name,
                model=self._model or model_name,
                warnings=(_classify_transport_error(str(exc)),),
                evidence_used=request.context.evidence_texts,
                citation_preserved=None,
                metadata={"mode": request.mode.value},
            )

        backend, resolved_model = resolved
        provider_name = resp.provider if resp else backend.name
        model_name = self._model or (resp.model if resp else resolved_model)

        if resp is None or resp.text is None:
            error = (resp.error if resp else None) or "all providers exhausted"
            return RewriteResult(
                original_text=original,
                rewritten_text=None,
                status=RewriteStatus.PROVIDER_ERROR,
                provider=provider_name,
                model=model_name,
                warnings=(_classify_transport_error(error),),
                evidence_used=request.context.evidence_texts,
                citation_preserved=None,
                metadata={"mode": request.mode.value},
            )

        if _looks_like_refusal(resp.text):
            return RewriteResult(
                original_text=original,
                rewritten_text=None,
                status=RewriteStatus.REFUSED,
                provider=provider_name,
                model=model_name,
                warnings=("provider refused the rewrite request.",),
                evidence_used=request.context.evidence_texts,
                citation_preserved=None,
                metadata={"mode": request.mode.value},
            )

        validated = _validate_payload(_parse_json_response(resp.text))
        if validated is None:
            return RewriteResult(
                original_text=original,
                rewritten_text=None,
                status=RewriteStatus.INVALID_RESPONSE,
                provider=provider_name,
                model=model_name,
                warnings=("provider response failed schema validation.",),
                evidence_used=request.context.evidence_texts,
                citation_preserved=None,
                metadata={"mode": request.mode.value},
            )

        rewritten = str(validated["rewritten_text"])
        hallucinations = _detect_hallucinations(rewritten, request)
        if hallucinations:
            return RewriteResult(
                original_text=original,
                rewritten_text=None,
                status=RewriteStatus.INVALID_RESPONSE,
                provider=provider_name,
                model=model_name,
                warnings=tuple(hallucinations),
                evidence_used=request.context.evidence_texts,
                citation_preserved=None,
                metadata={"mode": request.mode.value},
            )

        # Deterministic citation check wins over the model's own claim.
        citation_raw = request.context.citation_raw
        if citation_raw:
            actual_preserved: bool | None = citation_raw in rewritten
        else:
            actual_preserved = None
        warnings = list(validated["warnings"])
        if (
            citation_raw
            and bool(validated["citation_preserved"]) != actual_preserved
        ):
            warnings.append(
                "provider citation_preserved flag disagreed with "
                "deterministic check; deterministic check used."
            )
        if rewritten == original:
            warnings.append("provider returned the original text unchanged.")

        return RewriteResult(
            original_text=original,
            rewritten_text=rewritten,
            status=RewriteStatus.SUCCESS,
            provider=provider_name,
            model=model_name,
            warnings=tuple(warnings),
            evidence_used=request.context.evidence_texts,
            citation_preserved=actual_preserved,
            metadata={
                "mode": request.mode.value,
                "reason": validated["reason"],
            },
        )
