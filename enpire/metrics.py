"""Efficiency metrics for multi-agent physical autoresearch.

ENPIRE proposes two utilization metrics to measure how efficiently a fleet of
agent-robot pairs converts resources into research progress. We adapt them to
the simulation setting:

* **Mean Robot Utilization (MRU)** -- the fraction of total research time spent
  actually running rollouts on the (simulated) robot, as opposed to the agent
  "thinking" (proposing hypotheses). In the paper, MRU decreases as the fleet
  grows because agents spend more time coordinating.
* **Mean Token Utilization (MTU)** -- the mean LLM tokens consumed per unit of
  research progress (or per agent). For non-LLM (mock) agents this is zero.

We also track the headline outcome metrics: best success rate over time and
time-to-success (the research step / wall-clock at which a target success rate
is first reached).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResearchTrace:
    """Time-series record of one autoresearch run, used to compute metrics.

    Attributes:
        target_success: Success-rate threshold defining "solved".
        rollout_seconds: Cumulative wall-clock spent running rollouts.
        thinking_seconds: Cumulative wall-clock spent in agent proposal calls.
        total_tokens: Cumulative LLM tokens consumed across all agents.
        n_agents: Number of agents in the fleet.
        best_success_curve: List of ``(step, wall_clock, best_success_rate)``.
        steps_to_success: Research step at which ``target_success`` was reached
            (``None`` if never).
        seconds_to_success: Wall-clock at which it was reached (``None`` if never).
    """

    target_success: float
    rollout_seconds: float = 0.0
    thinking_seconds: float = 0.0
    total_tokens: int = 0
    n_agents: int = 1
    best_success_curve: list[tuple[int, float, float]] = field(default_factory=list)
    steps_to_success: int | None = None
    seconds_to_success: float | None = None

    def record(self, step: int, wall_clock: float, best_success: float) -> None:
        """Append a point to the best-success curve and note time-to-success."""
        self.best_success_curve.append((step, wall_clock, best_success))
        if self.steps_to_success is None and best_success >= self.target_success:
            self.steps_to_success = step
            self.seconds_to_success = wall_clock

    @property
    def mean_robot_utilization(self) -> float:
        """MRU: rollout time / total research time."""
        total = self.rollout_seconds + self.thinking_seconds
        return self.rollout_seconds / total if total > 0 else 0.0

    @property
    def mean_token_utilization(self) -> float:
        """MTU: mean tokens consumed per agent (0 for non-LLM agents)."""
        return self.total_tokens / self.n_agents if self.n_agents else 0.0

    @property
    def final_best_success(self) -> float:
        """The best success rate achieved by the end of the run."""
        return self.best_success_curve[-1][2] if self.best_success_curve else 0.0

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the headline metrics."""
        return {
            "target_success": self.target_success,
            "final_best_success": self.final_best_success,
            "steps_to_success": self.steps_to_success,
            "seconds_to_success": self.seconds_to_success,
            "mean_robot_utilization": self.mean_robot_utilization,
            "mean_token_utilization": self.mean_token_utilization,
            "total_tokens": self.total_tokens,
            "n_agents": self.n_agents,
            "rollout_seconds": self.rollout_seconds,
            "thinking_seconds": self.thinking_seconds,
        }
