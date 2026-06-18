"""A deterministic, API-free coding agent.

``MockAgent`` lets the entire ENPIRE-Sim loop run with no API keys and no cost,
which is essential for development, CI, and the bandit-selector ablation (where
we want to study the *selector*, not the quality of an LLM). It performs a
simple but genuine local search over the heuristic parameter space:

* On the first call it proposes the centre of the search space.
* Thereafter it takes the best hypothesis seen so far and mutates each
  parameter with Gaussian noise, clipped to the parameter bounds -- an
  evolutionary "propose a nearby variant" step, analogous to an agent tweaking
  a hyperparameter after reading a failure log.

Mutation scale shrinks as the best success rate rises, mimicking an agent that
makes bolder changes when far from the goal and fine adjustments when close.
"""

from __future__ import annotations

import numpy as np

from ..policy.base import Hypothesis
from ..policy.heuristic import HEURISTIC_PARAM_SPACE, default_heuristic_params
from ..rollout.runner import RolloutResult
from .base import BaseAgent


class MockAgent(BaseAgent):
    """Mutation-based local-search agent (no external API).

    Args:
        agent_id: Identifier / branch name.
        seed: RNG seed for reproducible mutations.
        base_mutation: Mutation standard deviation (as a fraction of each
            parameter's range) when success rate is zero.
    """

    def __init__(self, agent_id: str, seed: int | None = None, base_mutation: float = 0.25):
        super().__init__(agent_id)
        self._rng = np.random.default_rng(seed)
        self.base_mutation = float(base_mutation)

    def propose(
        self,
        task_description: str,
        history: list[tuple[Hypothesis, RolloutResult]],
    ) -> Hypothesis:
        best = self._best_in_history(history)
        if best is None:
            return Hypothesis(
                regime="heuristic",
                params=default_heuristic_params(),
                parent_id=None,
                agent_id=self.agent_id,
                rationale="Initial proposal at the centre of the search space.",
            )

        best_rate = max(r.success_rate for _, r in history)
        # Anneal mutation: bold when far from solved, fine when close.
        scale = self.base_mutation * (1.0 - 0.8 * best_rate)

        new_params: dict[str, float] = {}
        for key, (lo, hi) in HEURISTIC_PARAM_SPACE.items():
            span = hi - lo
            current = float(best.params.get(key, 0.5 * (lo + hi)))
            mutated = current + self._rng.normal(0.0, scale * span)
            new_params[key] = float(np.clip(mutated, lo, hi))

        return Hypothesis(
            regime="heuristic",
            params=new_params,
            parent_id=best.hyp_id,
            agent_id=self.agent_id,
            rationale=(
                f"Mutated best-so-far (success={best_rate:.2f}) with "
                f"scale={scale:.3f} to explore nearby parameters."
            ),
        )
