"""Thompson-sampling selector -- Bayesian hypothesis selection.

For Bernoulli rewards (episode success / failure) the conjugate prior is a
Beta distribution. Thompson sampling maintains a ``Beta(1 + successes,
1 + failures)`` posterior per arm, draws one sample from each, and pulls the
arm with the highest sample. This is a strong, low-variance bandit policy that
naturally balances exploration and exploitation and tends to outperform UCB
empirically on Bernoulli problems -- a good second principled comparator for
the Evolution-module ablation.
"""

from __future__ import annotations

from .base import ArmStats, BaseSelector


class ThompsonSelector(BaseSelector):
    """Thompson sampling with Beta posteriors over per-arm success rates.

    Args:
        prior_alpha: Alpha of the Beta prior (pseudo-successes). Default 1.0
            gives a uniform prior.
        prior_beta: Beta of the Beta prior (pseudo-failures). Default 1.0.
        seed: RNG seed for posterior sampling.
    """

    name = "thompson"

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(seed=seed)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)

    def select(self, arms: list[ArmStats]) -> int:
        samples = []
        for arm in arms:
            alpha = self.prior_alpha + arm.n_success
            beta = self.prior_beta + (arm.n_pulls - arm.n_success)
            samples.append(float(self._rng.beta(alpha, beta)))
        return self._argmax_with_tiebreak(samples)
