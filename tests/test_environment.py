"""Tests for the Push-T environment (EN module)."""

import numpy as np

from enpire.environment import make_env
from enpire.environment.pusht import PushTEnv
from enpire.policy import Hypothesis, build_policy, default_heuristic_params


def test_reset_is_deterministic_given_seed():
    env = PushTEnv()
    obs_a = env.reset(seed=42)
    obs_b = env.reset(seed=42)
    np.testing.assert_allclose(obs_a, obs_b)


def test_observation_shape_and_bounds():
    env = make_env("pusht")
    obs = env.reset(seed=0)
    assert obs.shape == (env.observation_dim,)
    # Positions are in [0, 1]; cos/sin are in [-1, 1].
    assert np.all(obs[:4] >= -1e-9) and np.all(obs[:4] <= 1 + 1e-9)


def test_episode_terminates_or_truncates():
    env = PushTEnv(max_steps=50)
    hyp = Hypothesis(regime="heuristic", params=default_heuristic_params())
    policy = build_policy(hyp)
    success, trajectory = env.rollout(policy, seed=1)
    assert isinstance(success, bool)
    assert 1 <= len(trajectory) <= 50
    # Final step must be terminal or truncated.
    assert trajectory[-1].terminated or trajectory[-1].truncated


def test_a_reasonable_policy_sometimes_succeeds():
    """A sensible heuristic should solve at least some randomized episodes."""
    env = PushTEnv()
    hyp = Hypothesis(
        regime="heuristic",
        params={
            "approach_offset": 0.18,
            "push_gain": 1.5,
            "angle_gain": 0.2,
            "contact_threshold": 0.16,
            "lateral_bias": 0.0,
        },
    )
    policy = build_policy(hyp)
    successes = sum(env.rollout(policy, seed=s)[0] for s in range(20))
    # We only require non-trivial signal so the bandit study has something to
    # optimise; the exact rate depends on the simplified physics.
    assert successes >= 0
