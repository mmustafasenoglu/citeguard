"""Fail-closed adapter from grounded rewrite providers to reduction candidates."""

from __future__ import annotations

from collections.abc import Mapping

from ..reduction.models import (
    RewriteCandidate,
)
from ..reduction.models import (
    RewriteRequest as ReductionRewriteRequest,
)
from .models import (
    RewriteProvider,
    RewriteStatus,
)
from .models import (
    RewriteRequest as GroundedRewriteRequest,
)


class EvidenceGroundedReductionBackend:
    """Expose verified rewrite proposals through the reduction backend contract.

    Grounded requests must be supplied by the caller and keyed by passage ID.
    A missing mapping, disabled plan, or offline execution fails closed without
    invoking the provider. Provider success creates only an unvalidated
    candidate; the reduction integrity and meaning gates remain authoritative.
    """

    def __init__(
        self,
        provider: RewriteProvider,
        grounded_requests: Mapping[str, GroundedRewriteRequest],
        *,
        offline: bool = False,
    ) -> None:
        self._provider = provider
        self._grounded_requests = grounded_requests
        self._offline = offline

    def generate(self, request: ReductionRewriteRequest) -> list[RewriteCandidate]:
        """Return bounded, unvalidated proposals backed by verified context."""
        if (
            self._offline
            or not request.plan.rewrite_allowed
            or request.candidate_count < 1
        ):
            return []
        grounded_request = self._grounded_requests.get(request.plan.passage_id)
        if grounded_request is None:
            return []

        candidates: list[RewriteCandidate] = []
        for _ in range(request.candidate_count):
            try:
                result = self._provider.rewrite(grounded_request)
            except Exception:
                continue
            if result.status != RewriteStatus.SUCCESS or result.rewritten_text is None:
                continue
            candidates.append(
                RewriteCandidate(
                    text=result.rewritten_text,
                    generator=f"{result.provider}:{result.model}",
                )
            )
        return candidates
