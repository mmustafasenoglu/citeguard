"""Entailment evaluation for claim-evidence pairs.

This module provides both an offline contradiction-signal detector and
an optional LLM-backed entailment classifier.  The offline mode flags
negation patterns that suggest a risk of contradiction; it does NOT
make a final supported/contradicted decision on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import _api_key, _call_anthropic, _parse_json_response
from .models import Claim, Evidence, SourceCandidate, Verdict

_NEGATION_SIGNALS = re.compile(
    r"\b(?:not|no|did not|does not|did not|does not|failed to|no significant"
    r"|not associated|not linked|no evidence|no effect|no impact"
    r"|not significantly|not considerably|neither|nor|without"
    r"|lack of|absence of|contrary to|opposite of|refutes?|negates?)\b",
    re.IGNORECASE,
)

_ENTAILMENT_SYSTEM_PROMPT = (
    "You are an academic entailment classifier. "
    "Given a claim and evidence passages, classify the relationship. "
    "Return ONLY valid JSON, no markdown fences."
)

_ENTAILMENT_PROMPT = """\
Classify the relationship between the claim and the evidence.

CLAIM:
{claim_text}

EVIDENCE:
{evidence_text}

Classify ONLY as one of:
- "supported": the evidence directly supports the claim
- "partially_supported": the evidence is related but does not fully support the claim
- "contradicted": the evidence contradicts the claim
- "insufficient_information": not enough information to decide

Return a JSON object with:
- "verdict": one of the four values above
- "confidence": 0-100
- "reasoning": concise one-sentence explanation

Return ONLY the JSON object, nothing else.
"""


@dataclass(slots=True)
class EntailmentResult:
    """Output of an entailment evaluation."""

    verdict: Verdict
    confidence: int
    reasoning: str


def contradiction_risk(
    claim: Claim,
    evidence_text: str,
) -> EntailmentResult:
    """Offline contradiction-signal detector.

    Flags negation patterns in the evidence text relative to the claim.
    This is a heuristic; it does NOT conclude that the evidence
    contradicts the claim.  It only raises the *risk* of contradiction.
    Final supported/contradicted decisions require LLM-backed entailment.
    """
    has_negation = bool(_NEGATION_SIGNALS.search(evidence_text))

    claim_tokens = set(re.findall(r"[a-z0-9]+", claim.text.lower()))
    evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence_text.lower()))

    if not claim_tokens:
        return EntailmentResult(
            verdict=Verdict.INSUFFICIENT_INFORMATION,
            confidence=0,
            reasoning="No claim tokens to compare.",
        )

    overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)

    if has_negation and overlap > 0.3:
        return EntailmentResult(
            verdict=Verdict.INSUFFICIENT_INFORMATION,
            confidence=min(round(50 + overlap * 35), 85),
            reasoning=(
                "Negation signals detected in topically relevant evidence; "
                "contradiction risk is elevated but requires LLM verification."
            ),
        )

    if has_negation:
        return EntailmentResult(
            verdict=Verdict.INSUFFICIENT_INFORMATION,
            confidence=min(round(20 + overlap * 30), 50),
            reasoning="Negation signals present but topical overlap is low.",
        )

    if overlap > 0.5:
        return EntailmentResult(
            verdict=Verdict.PARTIALLY_SUPPORTED,
            confidence=min(round(30 + overlap * 50), 75),
            reasoning="Strong topical overlap without negation signals.",
        )

    return EntailmentResult(
        verdict=Verdict.INSUFFICIENT_INFORMATION,
        confidence=round(overlap * 40),
        reasoning="Insufficient topical overlap for entailment judgment.",
    )


def evaluate_evidence_with_llm(
    claim: Claim,
    evidence: list[Evidence],
    candidate: SourceCandidate,
    *,
    model: str = "claude-sonnet-4-20250514",
    timeout: float = 30,
) -> EntailmentResult | None:
    """LLM-backed entailment classifier.

    Sends only the top evidence passages to the LLM.  Returns ``None``
    when the API key is absent or the call fails, allowing fallback to
    the offline detector.
    """
    api_key = _api_key()
    if not api_key or not evidence:
        return None

    evidence_text = "\n".join(
        f"- {e.text}" for e in evidence[:3]
    )

    prompt = _ENTAILMENT_PROMPT.format(
        claim_text=claim.text,
        evidence_text=evidence_text,
    )
    raw = _call_anthropic(
        api_key, _ENTAILMENT_SYSTEM_PROMPT, prompt,
        model=model, timeout=timeout,
    )
    if not raw:
        return None

    result = _parse_json_response(raw)
    if not isinstance(result, dict):
        return None

    verdict_str = str(result.get("verdict", "insufficient_information"))
    try:
        verdict = Verdict(verdict_str)
    except ValueError:
        verdict = Verdict.INSUFFICIENT_INFORMATION

    confidence = result.get("confidence", 0)
    if not isinstance(confidence, (int, float)):
        confidence = 0
    confidence = max(0, min(round(confidence), 100))

    reasoning = str(result.get("reasoning", ""))[:200]

    return EntailmentResult(
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
    )
