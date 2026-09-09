"""Stable public Python API for citeguard 1.0.

Internal modules remain internal.  Only the symbols re-exported here are
part of the supported programmatic interface.

Usage::

    from citeguard.api import (
        AuditOptions,
        AuditResult,
        audit_document,
        suggest_document,
        verify_document,
    )
"""

from __future__ import annotations

from .audit import (
    AuditOptions,
    AuditResult,
    audit_document,
    suggest_document,
    verify_document,
)
from .plagiarism import PlagiarismConfig, PlagiarismResult, scan_document

__all__ = [
    "AuditOptions",
    "AuditResult",
    "PlagiarismConfig",
    "PlagiarismResult",
    "audit_document",
    "suggest_document",
    "scan_document",
    "verify_document",
]
