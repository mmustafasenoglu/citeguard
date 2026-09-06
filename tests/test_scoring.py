import pytest

from citeguard.scoring import compute_health_score, overall_confidence


def test_overall_confidence_is_deterministic() -> None:
    # Without entailment: metadata * 0.40 + support * 0.60
    assert overall_confidence(100, 20) == 52
    assert overall_confidence(80, 80) == 80
    # With entailment: metadata * 0.20 + support * 0.80
    assert overall_confidence(100, 20, has_entailment=True) == 36
    assert overall_confidence(80, 80, has_entailment=True) == 80
    assert overall_confidence(0, 75, has_entailment=True) == 60


def test_health_score_is_bounded() -> None:
    assert compute_health_score(
        citation_coverage=1,
        verification_ratio=1,
        support_ratio=1,
        bibliography_consistency=1,
        evidence_coverage=1,
    ) == 100

    assert compute_health_score(
        citation_coverage=0,
        verification_ratio=0,
        support_ratio=0,
        bibliography_consistency=0,
        evidence_coverage=0,
        contradictions=100,
    ) == 0


def test_invalid_ratio_raises() -> None:
    with pytest.raises(ValueError):
        compute_health_score(
            citation_coverage=1.1,
            verification_ratio=1,
            support_ratio=1,
            bibliography_consistency=1,
            evidence_coverage=1,
        )
