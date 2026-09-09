"""Optional, evidence-grounded rewrite suggestions over verified results.

Rewrite suggestions are generated text only and do not constitute
citation verification.  Verification results remain authoritative.

The rewrite layer never runs during default analysis — it is invoked
only through the explicit ``citeguard rewrite`` command.
"""

from .models import (
    RewriteContext,
    RewriteMode,
    RewriteProvider,
    RewriteRequest,
    RewriteResult,
    RewriteStatus,
)
from .provider import LLMRewriteProvider
from .reduction_adapter import EvidenceGroundedReductionBackend
from .service import (
    build_context,
    collect_rewrite_requests,
    insufficient_result,
)

__all__ = [
    "LLMRewriteProvider",
    "EvidenceGroundedReductionBackend",
    "RewriteContext",
    "RewriteMode",
    "RewriteProvider",
    "RewriteRequest",
    "RewriteResult",
    "RewriteStatus",
    "build_context",
    "collect_rewrite_requests",
    "insufficient_result",
]
