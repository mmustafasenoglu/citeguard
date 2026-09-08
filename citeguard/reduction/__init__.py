"""Attribution-risk reduction primitives.

The reduction package is deliberately separate from document auditing.  It
turns similarity evidence into reviewable plans and validates rewrite
candidates without treating lower similarity as proof of academic integrity.
"""

from .analyzer import analyze_passage_risks
from .apply import (
    AppliedReplacement,
    TextReplacement,
    apply_text_replacements,
    restore_text,
    write_revised_text,
)
from .candidates import generate_candidates
from .evaluator import evaluate_candidate
from .meaning import (
    HeuristicEntailmentBackend,
    MeaningThresholds,
    SequenceSemanticBackend,
    validate_meaning,
)
from .metrics import ReductionMetrics, compute_reduction_metrics
from .models import (
    EntailmentDirection,
    FixAction,
    FixPlan,
    MeaningValidation,
    MeaningVerdict,
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
    "EntailmentDirection",
    "MeaningValidation",
    "MeaningVerdict",
    "PassageRisk",
    "ReductionRiskType",
    "RewriteCandidate",
    "analyze_passage_risks",
    "evaluate_candidate",
    "generate_candidates",
    "HeuristicEntailmentBackend",
    "MeaningThresholds",
    "SequenceSemanticBackend",
    "validate_meaning",
    "ReductionMetrics",
    "compute_reduction_metrics",
    "build_fix_plans",
    "rank_candidates",
    "validate_candidate",
    "AppliedReplacement",
    "TextReplacement",
    "apply_text_replacements",
    "restore_text",
    "write_revised_text",
]
