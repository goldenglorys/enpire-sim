"""Random selector -- an exploration-only lower bound for the ablation."""

from __future__ import annotations

from .base import ArmStats, BaseSelector


class RandomSelector(BaseSelector):
    """Pull a uniformly random arm every step.

    Useful as a control: any selector worth using should clearly beat random
    allocation of the rollout budget.
    """

    name = "random"

    def select(self, arms: list[ArmStats]) -> int:
        return int(self._rng.integers(len(arms)))
