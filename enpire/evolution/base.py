"""Evolution (E) module: how the team decides which hypothesis to invest in.

This is the heart of ENPIRE-Sim's novel contribution. In the original ENPIRE,
hypothesis selection is *emergent and greedy*: agents coordinate through Git,
cherry-picking whatever currently has the best average success rate. That is a
sensible engineering default, but it makes no explicit exploration/exploitation
trade-off and provides no theoretical guarantees.

We reframe hypothesis selection as a **multi-armed bandit**: each hypothesis is
an arm, pulling an arm means spending a rollout budget to (re-)evaluate it, and
the reward is the (noisy, Bernoulli) per-episode success. A *selector* decides
which arm to pull next. This lets us compare:

* :class:`~enpire.evolution.greedy.GreedySelector` -- the ENPIRE baseline.
* :class:`~enpire.evolution.ucb.UCBSelector` -- principled optimism (UCB1).
* :class:`~enpire.evolution.thompson.ThompsonSelector` -- Bayesian sampling.
* :class:`~enpire.evolution.random_selector.RandomSelector` -- a lower bound.

All selectors share the :class:`ArmStats` view of an arm and the
:class:`BaseSelector` interface, so the orchestrator is agnostic to which one is
in use -- exactly the kind of fair ablation the ENPIRE paper calls for.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class ArmStats:
    """Sufficient statistics for one hypothesis treated as a bandit arm.

    Attributes:
        hyp_id: The hypothesis identifier.
        n_pulls: Total number of episodes spent evaluating this hypothesis.
        n_success: Total number of successful episodes.
    """

    hyp_id: str
    n_pulls: int = 0
    n_success: int = 0

    @property
    def mean(self) -> float:
        """Empirical success rate; ``0.0`` for an unpulled arm."""
        return self.n_success / self.n_pulls if self.n_pulls > 0 else 0.0

    def update(self, n_episodes: int, n_success: int) -> None:
        """Fold a new evaluation result into the running statistics."""
        self.n_pulls += n_episodes
        self.n_success += n_success


class BaseSelector(abc.ABC):
    """Abstract hypothesis selector (bandit policy).

    Subclasses implement :meth:`select`, which chooses the index of the arm to
    pull next given the current statistics for every live arm.
    """

    #: Human-readable name used in logs and plots.
    name: str = "base"

    def __init__(self, seed: int | None = None):
        import numpy as np

        self._rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def select(self, arms: list[ArmStats]) -> int:
        """Return the index (into ``arms``) of the arm to evaluate next.

        Args:
            arms: The current live arms with their statistics. Never empty.

        Returns:
            An integer index in ``range(len(arms))``.
        """

    def _argmax_with_tiebreak(self, values) -> int:
        """Argmax that breaks ties uniformly at random (avoids index bias)."""
        import numpy as np

        values = np.asarray(values, dtype=float)
        best = np.flatnonzero(values == values.max())
        return int(self._rng.choice(best))
