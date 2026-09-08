"""Backend-neutral candidate generation orchestration."""

from __future__ import annotations

from .models import RewriteBackend, RewriteCandidate, RewriteRequest


def generate_candidates(
    backend: RewriteBackend,
    request: RewriteRequest,
) -> list[RewriteCandidate]:
    """Ask a backend for bounded candidates without accepting any candidate.

    Backends propose text only.  Validation and ranking remain separate so a
    provider cannot bypass the integrity gates by returning a preferred result.
    """
    if request.candidate_count < 1:
        raise ValueError("candidate_count must be at least 1")
    if not request.plan.rewrite_allowed:
        return []
    candidates = backend.generate(request)
    if len(candidates) > request.candidate_count:
        candidates = candidates[: request.candidate_count]
    return candidates
