"""Source-aware plagiarism review built on CiteGuard's similarity engine."""

from .models import PlagiarismConfig, PlagiarismResult
from .pipeline import scan_document

__all__ = ["PlagiarismConfig", "PlagiarismResult", "scan_document"]
