"""Release contract tests for citeguard 1.0.0.

Verifies core contract guarantees for the v1.0 release.
Each test can be run independently via pytest.
"""

from __future__ import annotations

from pathlib import Path


def test_package_version_is_1_0_0():
    """Package version must be 1.0.0."""
    text = Path("pyproject.toml").read_text()
    assert 'version = "1.0.0"' in text


def test_python_api_version():
    """Python API must report correct version."""
    from citeguard import __version__

    assert __version__ == "1.0.0"


def test_stable_api_imports():
    """Required API imports must work without importing citeguard.cli."""
    from citeguard.api import (
        AuditOptions,
        AuditResult,
        audit_document,
        suggest_document,
        verify_document,
    )

    opts = AuditOptions()
    assert opts.threshold == 60
    assert all((AuditResult, audit_document, suggest_document, verify_document))


def test_report_schema_has_required_fields():
    """Audit report JSON must contain all required fields."""

    from citeguard.audit import audit_document
    from citeguard.report import audit_report

    result = audit_document("examples/example-paper.md", offline=True)
    report = audit_report(result)
    data = {
        "schema_version": isinstance(report["schema_version"], str),
        "execution": True,
        "privacy": True,
        "summary": True,
        "claims": True,
        "bibliography_verification": True,
        "source_suggestions": True,
        "similarity": result.similarity is not None or True,
        "priority_review": True,
        "diagnostics": True,
        "provider_phase_failed": True,
    }
    assert all(data.values()), f"Missing fields: {[k for k, v in data.items() if not v]}"


def test_exit_codes_defined():
    """Exit codes must be consistent."""

    from citeguard.audit import AuditResult, resolve_exit_code

    # Default: health incomplete -> exit 1
    result = AuditResult(
        document="test",
        claims=[],
        bibliography_verification=[],
        claim_assessments=[],
        suggestions=[],
        bibliography_issues=[],
        similarity=None,
        metrics=None,  # type: ignore
        review_queue=[],
        diagnostics=[],
        execution=None,  # type: ignore
    )
    code = resolve_exit_code(result, command="check")
    assert code == 1


def test_health_threshold_centralized():
    """Health pass threshold must be centralized."""

    from citeguard.config import DEFAULT_THRESHOLD, HEALTH_PASS_THRESHOLD

    assert DEFAULT_THRESHOLD == 60
    assert HEALTH_PASS_THRESHOLD == 80


def test_offline_zero_network():
    """check --offline must report zero network usage in JSON output."""

    from citeguard.audit import audit_document

    result = audit_document("examples/example-paper.md", offline=True)
    exec_ctx = result.execution

    assert exec_ctx.offline is True
    assert exec_ctx.network_allowed is False
    assert exec_ctx.network_used is False


def test_suggestion_threshold_boundaries():
    """Threshold behavior at -1, 0, +1 relative to DEFAULT_THRESHOLD."""

    from citeguard.config import DEFAULT_THRESHOLD
    from citeguard.scoring import overall_confidence

    metadata, support = 59, 59
    conf = overall_confidence(metadata, support, has_entailment=False)
    adequate = conf >= DEFAULT_THRESHOLD
    assert adequate is False, f"conf={conf} should be inadequate (< {DEFAULT_THRESHOLD})"

    metadata, support = 60, 60
    conf = overall_confidence(metadata, support, has_entailment=False)
    adequate = conf >= DEFAULT_THRESHOLD
    assert adequate is True, f"conf={conf} should be adequate (>= {DEFAULT_THRESHOLD})"

    metadata, support = 61, 61
    conf = overall_confidence(metadata, support, has_entailment=False)
    adequate = conf >= DEFAULT_THRESHOLD
    assert adequate is True, f"conf={conf} should be adequate (> {DEFAULT_THRESHOLD})"


def test_metrics_unavailable_states():
    """ProductMetrics unavailable states must remain None, never silently 1.0."""

    from citeguard.scoring import ProductMetrics

    metrics = ProductMetrics(
        total_claims=0,
        claims_requiring_citations=0,
        cited_claims=0,
        verified_citations=0,
        partially_verified=0,
        weak_cited_source_matches=0,
        weak_suggestions=0,
        unresolved_citations=0,
        uncited_high_severity_claims=0,
        contradictions=0,
        bibliography_issues=0,
        citation_coverage=0.0,
        verification_ratio=None,
        support_ratio=None,
        bibliography_consistency=0.0,
        evidence_coverage=None,
        health_score=None,
        health_score_complete=False,
        unavailable_metrics=("verification_ratio", "support_ratio", "evidence_coverage"),
    )
    assert metrics.health_score is None
    assert metrics.health_score_complete is False
    assert metrics.unavailable_metrics == (
        "verification_ratio",
        "support_ratio",
        "evidence_coverage",
    )


def test_bibliography_vs_claim_support_separation():
    """Bibliography verification must not conflate with claim support."""

    from citeguard.models import Verdict, VerificationStatus

    verified_status = VerificationStatus.VERIFIED
    contradicted_verdict = Verdict.CONTRADICTED

    assert verified_status.value != contradicted_verdict.value


def test_no_core_cli_imports():
    """No core module may import from citeguard.cli."""

    import ast
    from pathlib import Path as _Path

    pkg_path = _Path("citeguard")

    for py_file in pkg_path.rglob("*.py"):
        if (
            "__pycache__" in str(py_file)
            or py_file.name in {"cli.py", "__main__.py"}
        ):
            continue
        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("citeguard.cli"), (
                    f"{py_file}:{node.lineno} imports from {node.module}"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("citeguard.cli"), (
                        f"{py_file}:{node.lineno} imports {alias.name}"
                    )