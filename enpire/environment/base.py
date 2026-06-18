"""Base environment interface (the EN module of ENPIRE).

In the original ENPIRE paper the Environment module is responsible for three
things that make a real robot task *agent-operable*:

* **Safety constraints** -- a bounded configuration space; violations trigger
  an immediate failure + reset.
* **Automated verification** -- a reward/success function synthesised by the
  coding agent from a handful of success/failure demonstrations.
* **Automated reset** -- returning the scene to a fresh randomized initial
  state without human intervention.

In simulation, reset and verification are essentially free, so the abstraction
collapses to a familiar Gym-style interface. We keep the interface deliberately
close to ``gymnasium`` so that swapping in a high-fidelity environment such as
``gym-pusht`` or a RoboCasa task is a drop-in change.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class StepResult:
    """Outcome of a single environment step.

    Attributes:
        observation: The observation after taking the action.
        reward: Scalar reward for this step (binary success rewards are common
            in ENPIRE, but we allow dense shaping too).
        terminated: True if the episode ended by reaching a terminal state
            (success or hard-constraint violation).
        truncated: True if the episode ended due to a time limit.
        info: Free-form diagnostics (e.g. coverage, contact, constraint flags).
    """

    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


class BaseEnv(abc.ABC):
    """Abstract Gym-style environment with ENPIRE's reset/verify contract.

    Subclasses must implement :meth:`reset`, :meth:`step`, and
    :meth:`is_success`. The :meth:`rollout` helper runs a full episode given a
    policy callable and returns whether it succeeded along with a trajectory.
    """

    #: Dimensionality of the (flat) observation vector.
    observation_dim: int = 0
    #: Dimensionality of the (flat) action vector.
    action_dim: int = 0
    #: Maximum number of steps before truncation.
    max_steps: int = 200

    @abc.abstractmethod
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset to a randomized initial state and return the observation."""

    @abc.abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        """Advance the simulation by one control step."""

    @abc.abstractmethod
    def is_success(self) -> bool:
        """Return True if the current state satisfies the task goal.

        This is the automated *verification* signal of the EN module.
        """

    def rollout(self, policy, seed: int | None = None, max_steps: int | None = None):
        """Run one full episode with ``policy`` and report success.

        Args:
            policy: A callable mapping an observation array to an action array.
            seed: Optional seed controlling the randomized initial state.
            max_steps: Optional override of :attr:`max_steps`.

        Returns:
            A tuple ``(success, trajectory)`` where ``success`` is a bool and
            ``trajectory`` is a list of ``StepResult`` objects (useful for the
            failure-log inspection the coding agent performs).
        """
        obs = self.reset(seed=seed)
        steps = max_steps if max_steps is not None else self.max_steps
        trajectory: list[StepResult] = []
        for _ in range(steps):
            action = np.asarray(policy(obs), dtype=np.float64)
            result = self.step(action)
            trajectory.append(result)
            obs = result.observation
            if result.terminated or result.truncated:
                break
        return self.is_success(), trajectory
