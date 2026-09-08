"""Attribution-risk reduction primitives.

The reduction package is deliberately separate from document auditing.  It
turns similarity evidence into reviewable plans and validates rewrite
candidates without treating lower similarity as proof of academic integrity.
"""

from .analyzer import analyze_passage_risks
from .candidates import generate_candidates
from .evaluator import evaluate_candidate
from .models import (
    FixAction,
    FixPlan,
    PassageRisk,
    ReductionRiskType,
    RewriteCandidate,
)
from .planner import build_fix_plans
from .ranker import rank_candidates
from .validator import validate_candidate

__all__ = [
    "FixAction",
    "FixPlan",
    "PassageRisk",
    "ReductionRiskType",
    "RewriteCandidate",
    "analyze_passage_risks",
    "evaluate_candidate",
    "generate_candidates",
    "build_fix_plans",
    "rank_candidates",
    "validate_candidate",
]
