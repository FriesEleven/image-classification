"""Budget-aware attention selection utilities."""

from .budget import (
    ATTENTION_CHOICES,
    STAGE_POSITIONS,
    attention_operation_profile,
    candidate_from_positions,
    enumerate_stage_candidates,
    score_candidates,
    select_candidates_for_budgets,
)
from .early_exit import (
    apply_policy,
    policy_metrics,
    select_policy,
    softmax_confidence,
    stratified_calibration_mask,
)

__all__ = [
    "ATTENTION_CHOICES",
    "STAGE_POSITIONS",
    "apply_policy",
    "attention_operation_profile",
    "candidate_from_positions",
    "enumerate_stage_candidates",
    "policy_metrics",
    "score_candidates",
    "select_candidates_for_budgets",
    "select_policy",
    "softmax_confidence",
    "stratified_calibration_mask",
]
