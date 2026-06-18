"""Run budgeted robot trials and preserve their results (the R module).

The Rollout module turns a :class:`~enpire.policy.base.Hypothesis` into an
*empirical success rate* by running it over a number of randomized episodes.
This mirrors ENPIRE's real-robot rollouts, except episodes are simulated and
therefore cheap. Each evaluation records enough information (per-episode
success, a sample failure trajectory) for the coding agent to inspect failures
and decide what to change next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..environment.base import BaseEnv
from ..policy.base import Hypothesis, build_policy


@dataclass
class RolloutResult:
    """Empirical outcome of evaluating one hypothesis.

    Attributes:
        hyp_id: The evaluated hypothesis' id.
        n_episodes: Number of episodes run in this evaluation.
        n_success: Number of successful episodes.
        success_rate: ``n_success / n_episodes``.
        mean_pos_error: Mean final block-to-goal distance (diagnostic).
        sample_failure: A short, human-readable summary of one failed episode,
            used by the coding agent to reason about failure modes. ``None`` if
            every episode succeeded.
        episode_seeds: The seeds used (for reproducibility / re-evaluation).
    """

    hyp_id: str
    n_episodes: int
    n_success: int
    success_rate: float
    mean_pos_error: float
    sample_failure: str | None = None
    episode_seeds: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hyp_id": self.hyp_id,
            "n_episodes": self.n_episodes,
            "n_success": self.n_success,
            "success_rate": self.success_rate,
            "mean_pos_error": self.mean_pos_error,
            "sample_failure": self.sample_failure,
        }


def evaluate_hypothesis(
    env: BaseEnv,
    hypothesis: Hypothesis,
    n_episodes: int = 10,
    seed_offset: int = 0,
) -> RolloutResult:
    """Evaluate a hypothesis over ``n_episodes`` randomized episodes.

    Args:
        env: The environment to roll out in.
        hypothesis: The hypothesis to compile and evaluate.
        n_episodes: How many episodes to run (the per-evaluation rollout budget).
        seed_offset: Base seed; episode ``i`` uses ``seed_offset + i`` so that
            different evaluation calls can either reuse or vary the seed set.

    Returns:
        A :class:`RolloutResult` summarising the evaluation.
    """
    policy = build_policy(hypothesis)
    n_success = 0
    pos_errors: list[float] = []
    sample_failure: str | None = None
    seeds: list[int] = []

    for i in range(n_episodes):
        seed = seed_offset + i
        seeds.append(seed)
        success, trajectory = env.rollout(policy, seed=seed)
        final_info = trajectory[-1].info if trajectory else {}
        pos_errors.append(float(final_info.get("pos_error", float("nan"))))
        if success:
            n_success += 1
        elif sample_failure is None:
            sample_failure = (
                f"episode seed={seed} failed after {len(trajectory)} steps; "
                f"final pos_error={final_info.get('pos_error'):.3f}, "
                f"ang_error={final_info.get('ang_error'):.3f}"
            )

    mean_pos_error = float(np.nanmean(pos_errors)) if pos_errors else float("nan")
    return RolloutResult(
        hyp_id=hypothesis.hyp_id,
        n_episodes=n_episodes,
        n_success=n_success,
        success_rate=n_success / n_episodes if n_episodes else 0.0,
        mean_pos_error=mean_pos_error,
        sample_failure=sample_failure,
        episode_seeds=seeds,
    )
