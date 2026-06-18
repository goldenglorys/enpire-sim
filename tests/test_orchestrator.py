"""End-to-end smoke tests for the orchestrator with the API-free MockAgent."""

from enpire.agents import make_agent
from enpire.environment import make_env
from enpire.evolution import make_selector
from enpire.orchestrator import Orchestrator, OrchestratorConfig


def _run(selector_name="ucb", n_agents=2, max_steps=20, seed=0):
    env = make_env("pusht")
    agents = [make_agent("mock", agent_id=f"a{i}", seed=seed + i) for i in range(n_agents)]
    selector = make_selector(selector_name, seed=seed)
    config = OrchestratorConfig(max_steps=max_steps, episodes_per_eval=4)
    return Orchestrator(env, agents, selector, config, verbose=False).run()


def test_run_completes_and_produces_trace():
    result = _run()
    assert result.trace.best_success_curve  # non-empty
    assert 0.0 <= result.trace.final_best_success <= 1.0
    assert len(result.arms) >= 1


def test_history_and_arms_consistent():
    result = _run(max_steps=15)
    # Every evaluated arm corresponds to a proposed hypothesis.
    arm_ids = {a.hypothesis.hyp_id for a in result.arms}
    for hyp, _ in result.history:
        assert hyp.hyp_id in arm_ids


def test_all_selectors_run_end_to_end():
    for name in ("greedy", "ucb", "thompson", "random"):
        result = _run(selector_name=name, max_steps=12)
        assert result.config_summary["selector"] == name


def test_metrics_summary_fields_present():
    result = _run()
    summary = result.trace.summary()
    for key in (
        "final_best_success",
        "mean_robot_utilization",
        "mean_token_utilization",
        "n_agents",
    ):
        assert key in summary


def test_mock_agent_improves_over_random_baseline_seeded():
    """Sanity: a multi-agent UCB run should reach a non-zero success rate."""
    result = _run(selector_name="ucb", n_agents=4, max_steps=60, seed=3)
    assert result.trace.final_best_success > 0.0
