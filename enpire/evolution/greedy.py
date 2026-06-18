"""Greedy selector -- the ENPIRE baseline.

The original ENPIRE Evolution module is, in effect, greedy hill-climbing:
agents keep and expand whatever currently has the best average success rate.
We model that with an epsilon-greedy policy. With ``epsilon = 0`` it is pure
exploitation (always re-evaluate the current best arm); a small ``epsilon``
adds the light, unstructured exploration that emerges from agents occasionally
trying something off-trend.
"""

from __future__ import annotations

from .base import ArmStats, BaseSelector


class GreedySelector(BaseSelector):
    """Epsilon-greedy hypothesis selection (the paper's default behaviour).

    Args:
        epsilon: Probability of pulling a uniformly random arm instead of the
            current best. ``0.0`` reproduces pure greedy hill-climbing.
        seed: RNG seed for tie-breaking and exploration.
    """

    name = "greedy"

    def __init__(self, epsilon: float = 0.0, seed: int | None = None):
        super().__init__(seed=seed)
        self.epsilon = float(epsilon)

    def select(self, arms: list[ArmStats]) -> int:
        # Always sample any never-evaluated arm first so a fresh proposal gets
        # at least one chance -- this mirrors an agent at least trying its idea.
        for i, arm in enumerate(arms):
            if arm.n_pulls == 0:
                return i
        if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
            return int(self._rng.integers(len(arms)))
        return self._argmax_with_tiebreak([arm.mean for arm in arms])
