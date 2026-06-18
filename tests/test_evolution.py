"""Tests for the Evolution-module selectors (the novel contribution)."""

import numpy as np

from enpire.evolution import make_selector, SELECTOR_REGISTRY
from enpire.evolution.base import ArmStats


def _arms(stats):
    return [ArmStats(hyp_id=f"a{i}", n_pulls=p, n_success=s) for i, (p, s) in enumerate(stats)]


def test_registry_has_all_selectors():
    assert set(SELECTOR_REGISTRY) == {"greedy", "ucb", "thompson", "random"}


def test_all_selectors_return_valid_index():
    arms = _arms([(10, 5), (10, 8), (10, 2)])
    for name in SELECTOR_REGISTRY:
        sel = make_selector(name, seed=0)
        idx = sel.select(arms)
        assert 0 <= idx < len(arms)


def test_selectors_pull_unevaluated_arm_first():
    """Greedy and UCB must try a never-pulled arm before exploiting."""
    arms = _arms([(10, 9), (0, 0)])  # arm 0 is great, arm 1 is untried
    for name in ("greedy", "ucb"):
        sel = make_selector(name, seed=0)
        assert sel.select(arms) == 1


def test_greedy_exploits_best_when_all_pulled():
    sel = make_selector("greedy", seed=0, epsilon=0.0)
    arms = _arms([(10, 3), (10, 9), (10, 5)])
    # Pure greedy with all arms pulled should pick the highest mean (index 1).
    assert sel.select(arms) == 1


def test_ucb_eventually_explores_underpulled_arm():
    """UCB should favour an under-pulled arm via its exploration bonus."""
    sel = make_selector("ucb", seed=0, c=2.0)
    # Arm 0: many pulls, decent mean. Arm 1: few pulls, slightly worse mean.
    arms = _arms([(100, 60), (2, 1)])
    # The exploration bonus on the under-pulled arm should win.
    assert sel.select(arms) == 1


def test_thompson_is_stochastic_but_valid():
    sel = make_selector("thompson", seed=1)
    arms = _arms([(10, 5), (10, 6)])
    picks = [sel.select(arms) for _ in range(50)]
    assert set(picks) <= {0, 1}
    # With different posteriors it should pick both arms at least sometimes.
    assert len(set(picks)) == 2


def test_armstats_mean_and_update():
    arm = ArmStats(hyp_id="x")
    assert arm.mean == 0.0
    arm.update(n_episodes=8, n_success=4)
    assert arm.n_pulls == 8 and arm.n_success == 4
    assert np.isclose(arm.mean, 0.5)
