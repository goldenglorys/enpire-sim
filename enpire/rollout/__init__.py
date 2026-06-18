"""Rollout (R) module: evaluate policies and preserve auditable results."""

from .runner import RolloutResult, evaluate_hypothesis

__all__ = ["RolloutResult", "evaluate_hypothesis"]
