"""Tests for the optional LLM integration module.

All tests run WITHOUT an API key and verify that functions gracefully
return None to allow deterministic fallback.
"""

import json

from citeguard.llm import (
    _map_claim_to_sentence,
    _parse_json_response,
    extract_claims_with_llm,
    match_source_with_llm,
)
from citeguard.models import ClaimType, ExistingCitation, Severity


def test_extract_claims_returns_none_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    citations = [
        ExistingCitation(
            raw_text="(Smith, 2020)", authors="Smith", year=2020,
            doi=None, numbered_ref=None, paragraph_index=0, char_offset=50,
        )
    ]
    result = extract_claims_with_llm(
        "A study shows 80% accuracy.", 0, citations
    )
    assert result is None


def test_match_source_returns_none_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from citeguard.models import Claim
    claim = Claim(
        text="Test claim", search_query="test",
        claim_type=ClaimType.GENERAL_FACT, severity=Severity.LOW,
        paragraph_index=0, has_existing_citation=False,
    )
    result = match_source_with_llm(
        claim, "Title", ["Author"], 2020, "Abstract"
    )
    assert result is None


def test_parse_json_response_clean_json() -> None:
    data = [{"text": "hello"}]
    assert _parse_json_response(json.dumps(data)) == data


def test_parse_json_response_with_fences() -> None:
    data = [{"text": "hello"}]
    fenced = f"```json\n{json.dumps(data)}\n```"
    assert _parse_json_response(fenced) == data


def test_parse_json_response_with_surrounding_text() -> None:
    data = [{"text": "hello"}]
    text = f"Here is the result:\n```json\n{json.dumps(data)}\n```\nDone."
    assert _parse_json_response(text) == data


def test_parse_json_response_invalid() -> None:
    assert _parse_json_response("not json at all") is None


def test_parse_json_response_empty() -> None:
    assert _parse_json_response("") is None


def test_parse_json_response_array() -> None:
    data = [1, 2, 3]
    assert _parse_json_response(json.dumps(data)) == data


def test_parse_json_response_object() -> None:
    data = {"key": "value"}
    assert _parse_json_response(json.dumps(data)) == data


def test_llm_citation_override(monkeypatch) -> None:
    """LLM claims has_existing_citation=True but parser finds none."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    fake_response = json.dumps([
        {
            "text": "Smoking increases cancer risk.",
            "search_query": "smoking cancer risk",
            "claim_type": "causal",
            "severity": "high",
            "has_existing_citation": True,
        }
    ])

    def _fake_call(
        api_key, system, user_message, *, model="m", max_tokens=2048, timeout=30
    ):
        return fake_response

    monkeypatch.setattr("citeguard.llm._call_anthropic", _fake_call)

    result = extract_claims_with_llm(
        "Smoking increases cancer risk.",
        0,
        [],
    )
    assert result is not None
    assert len(result) == 1
    assert result[0].has_existing_citation is False


def test_map_claim_to_sentence_finds_citation() -> None:
    """Claim maps back to a sentence with a citation via token overlap."""
    cit = ExistingCitation(
        raw_text="(Smith, 2020)", authors="Smith", year=2020,
        doi=None, numbered_ref=None, paragraph_index=0, char_offset=50,
    )
    sentences = [
        "Smoking increases cancer risk (Smith, 2020).",
        "Other unrelated sentence.",
    ]
    has_cit, linked = _map_claim_to_sentence(
        "Smoking increases cancer risk", sentences, [cit], {"(Smith, 2020)"}
    )
    assert has_cit is True
    assert len(linked) == 1
    assert linked[0].raw_text == "(Smith, 2020)"


def test_map_claim_to_sentence_no_citation_in_best_match() -> None:
    """Claim maps to a sentence without a citation."""
    cit = ExistingCitation(
        raw_text="(Jones, 2021)", authors="Jones", year=2021,
        doi=None, numbered_ref=None, paragraph_index=0, char_offset=80,
    )
    sentences = [
        "Smoking increases cancer risk.",
        "(Jones, 2021) studied a different topic.",
    ]
    has_cit, linked = _map_claim_to_sentence(
        "Smoking increases cancer risk", sentences, [cit], {"(Jones, 2021)"}
    )
    assert has_cit is False
    assert len(linked) == 0


def test_map_claim_to_sentence_low_overlap_returns_false() -> None:
    """Claim with no overlap returns no citation."""
    sentences = ["Completely unrelated text about flowers."]
    has_cit, linked = _map_claim_to_sentence(
        "Quantum computing breaks encryption", sentences, [], set()
    )
    assert has_cit is False
    assert linked == []


def test_map_claim_to_sentence_paragraph_with_citation(monkeypatch) -> None:
    """Full integration: LLM extracts claim without citation, but sentence has one."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    fake_response = json.dumps([
        {
            "text": "Smoking increases cancer risk.",
            "search_query": "smoking cancer risk",
            "claim_type": "causal",
            "severity": "high",
            "has_existing_citation": False,
        }
    ])

    def _fake_call(
        api_key, system, user_message, *, model="m", max_tokens=2048, timeout=30
    ):
        return fake_response

    monkeypatch.setattr("citeguard.llm._call_anthropic", _fake_call)

    cit = ExistingCitation(
        raw_text="(Smith, 2020)", authors="Smith", year=2020,
        doi=None, numbered_ref=None, paragraph_index=0, char_offset=50,
    )
    result = extract_claims_with_llm(
        "Smoking increases cancer risk (Smith, 2020).",
        0,
        [cit],
    )
    assert result is not None
    assert len(result) == 1
    assert result[0].has_existing_citation is True
    assert result[0].linked_citation is not None
    assert result[0].linked_citation.raw_text == "(Smith, 2020)"
