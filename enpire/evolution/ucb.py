"""UCB1 selector -- principled optimism under uncertainty.

UCB1 (Auer, Cesa-Bianchi & Fischer, 2002) selects the arm maximising

    mean_i + c * sqrt( ln(N) / n_i )

where ``N`` is the total number of pulls, ``n_i`` the pulls of arm ``i``, and
``c`` an exploration coefficient (``c = sqrt(2)`` recovers the classic bound).
The bonus term shrinks as an arm is pulled more, so under-explored hypotheses
are tried before the team commits to an apparent winner. This directly
addresses the ENPIRE Evolution module's lack of an explicit
exploration/exploitation trade-off.
"""

from __future__ import annotations

import math

from .base import ArmStats, BaseSelector


class UCBSelector(BaseSelector):
    """Upper Confidence Bound (UCB1) hypothesis selection.

    Args:
        c: Exploration coefficient. Larger values explore more. ``sqrt(2)`` is
            the textbook default.
        seed: RNG seed (used only for tie-breaking).
    """

    name = "ucb"

    def __init__(self, c: float = math.sqrt(2.0), seed: int | None = None):
        super().__init__(seed=seed)
        self.c = float(c)

    def select(self, arms: list[ArmStats]) -> int:
        # UCB1 requires every arm to be pulled once before the bonus is defined.
        for i, arm in enumerate(arms):
            if arm.n_pulls == 0:
                return i
        total = sum(arm.n_pulls for arm in arms)
        log_total = math.log(total)
        scores = [
            arm.mean + self.c * math.sqrt(log_total / arm.n_pulls) for arm in arms
        ]
        return self._argmax_with_tiebreak(scores)
