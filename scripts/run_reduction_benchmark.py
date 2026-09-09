#!/usr/bin/env python3
"""Run the deterministic attribution-reduction engineering benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from citeguard.models import Verdict
from citeguard.reduction.models import (
    FixAction,
    FixPlan,
    MeaningValidation,
    MeaningVerdict,
    RewriteCandidate,
)
from citeguard.reduction.ranker import rank_candidates
from citeguard.reduction.source_evaluator import evaluate_source_overlap
from citeguard.reduction.validator import validate_candidate

FIXTURE = Path(__file__).resolve().parents[1] / "benchmarks/reduction/engineering.json"


def main() -> None:
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))
    true_positive = false_positive = accepted = 0
    citation_pass = numeric_pass = unsupported_introductions = 0
    before_overlap: list[float] = []
    after_overlap: list[float] = []
    for case in cases:
        candidate = RewriteCandidate(case["candidate"], "engineering-fixture")
        evaluate_source_overlap(case["original"], candidate, case["source"])
        verdict = MeaningVerdict(case["meaning"])
        meaning = MeaningValidation(
            semantic_similarity_raw=0.95,
            forward_entailment_score=0.95,
            backward_entailment_score=0.95,
            forward_verdict=verdict,
            backward_verdict=verdict,
            meaning_preservation_score=0.95,
            verdict=verdict,
        )
        plan = FixPlan(
            passage_id=case["name"],
            action=FixAction.PARAPHRASE,
            preserve_citations=("(Smith, 2024)",),
            rewrite_allowed=case.get("rewrite_allowed", True),
            reason="engineering fixture",
        )
        if not plan.rewrite_allowed:
            candidate.rejection_reasons.append("planner does not permit rewriting")
        validate_candidate(
            case["original"],
            candidate,
            plan,
            supported_verdict=Verdict(case["support"]),
            unsupported_claims=case["unsupported"],
            meaning_validation=meaning,
        )
        candidate.meaning_verdict = verdict
        candidate.meaning_score = 0.95
        candidate.source_support_score = 1.0
        is_accepted = bool(rank_candidates([candidate]))
        expected = bool(case["expected_accepted"])
        accepted += int(is_accepted)
        true_positive += int(is_accepted and expected)
        false_positive += int(is_accepted and not expected)
        citation_pass += int(candidate.citations_preserved is True)
        numeric_pass += int(candidate.numeric_integrity is True)
        unsupported_introductions += len(candidate.introduced_claims)
        before_overlap.append(
            max(
                candidate.source_exact_overlap_before or 0.0,
                candidate.source_lexical_similarity_before or 0.0,
            )
        )
        after_overlap.append(
            max(
                candidate.source_exact_overlap_after or 0.0,
                candidate.source_lexical_similarity_after or 0.0,
            )
        )
        if is_accepted != expected:
            raise SystemExit(f"benchmark mismatch: {case['name']}")
    precision = true_positive / (true_positive + false_positive) if accepted else 1.0
    result = {
        "kind": "engineering_regression_not_scientific_validation",
        "cases": len(cases),
        "candidate_acceptance_precision": precision,
        "citation_integrity_passes": citation_pass,
        "numeric_integrity_passes": numeric_pass,
        "unsupported_claim_introductions_detected": unsupported_introductions,
        "mean_source_overlap_before": sum(before_overlap) / len(before_overlap),
        "mean_source_overlap_after": sum(after_overlap) / len(after_overlap),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
