"""The ENPIRE autoresearch loop (ties EN, PI, R, and E together).

This orchestrator runs the closed-loop, multi-agent physical-autoresearch
procedure in simulation:

1. A team of coding **agents** (PI) periodically proposes new hypotheses,
   each seeded from the shared pool of results (the simulation analogue of
   ENPIRE's Git-based cross-agent collaboration).
2. The **Evolution** selector (E) decides which hypothesis to spend the next
   rollout budget on, trading off exploration against exploitation. This is
   the component the thesis studies: greedy (the ENPIRE baseline) vs. UCB vs.
   Thompson vs. random.
3. The **Rollout** module (R) evaluates the chosen hypothesis on the
   (self-resetting, self-verifying) **environment** (EN), returning a noisy
   Bernoulli success rate.
4. Statistics are updated, the best success rate is recorded over wall-clock
   time, and the loop repeats until a step budget is exhausted.

The key design choice is the clean separation between *generation* (agents)
and *selection* (the Evolution module): every selector is compared on the same
stream of agent-proposed hypotheses, enabling the fair ablations ENPIRE calls
for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .agents.base import BaseAgent
from .environment.base import BaseEnv
from .evolution.base import ArmStats, BaseSelector
from .metrics import ResearchTrace
from .policy.base import Hypothesis
from .rollout.runner import RolloutResult, evaluate_hypothesis
from .utils.logging import RunLogger, get_console_logger

DEFAULT_TASK_DESCRIPTION = (
    "Solve the Push-T task: a circular pusher must push a T-shaped block to a "
    "goal pose using only non-prehensile contact. Maximise the fraction of "
    "randomized episodes in which the block reaches the goal within tolerance."
)


@dataclass
class OrchestratorConfig:
    """Configuration for one autoresearch run.

    Attributes:
        max_steps: Number of research steps (rollout-budget allocations).
        episodes_per_eval: Episodes run each time a hypothesis is evaluated.
        propose_every: Cadence (in steps) at which each agent proposes a new
            hypothesis. ``1`` means every step.
        target_success: Success-rate threshold counted as "solved".
        eval_seed_stride: Seeds for repeated evaluations of the same arm are
            varied by this stride so re-pulls are not identical (reduces
            correlated noise, closer to real re-evaluation).
        task_description: Natural-language task description handed to agents.
    """

    max_steps: int = 80
    episodes_per_eval: int = 8
    propose_every: int = 1
    target_success: float = 0.95
    eval_seed_stride: int = 1000
    task_description: str = DEFAULT_TASK_DESCRIPTION


@dataclass
class _Arm:
    """Internal bundle linking a hypothesis to its bandit statistics."""

    hypothesis: Hypothesis
    stats: ArmStats
    eval_count: int = 0  # number of times this arm has been (re-)evaluated


@dataclass
class RunResult:
    """Everything produced by a single autoresearch run.

    Attributes:
        trace: The metrics time-series (best-success curve, MRU, MTU, etc.).
        arms: Final arms with their statistics, for inspecting the idea tree.
        history: Full chronological ``(hypothesis, result)`` list.
        config_summary: A dict describing the run configuration.
    """

    trace: ResearchTrace
    arms: list[_Arm]
    history: list[tuple[Hypothesis, RolloutResult]]
    config_summary: dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """Runs the ENPIRE autoresearch loop for one (selector, agents) setup.

    Args:
        env: The environment (EN + R).
        agents: The coding-agent team (PI). One or more.
        selector: The Evolution-module hypothesis selector (E).
        config: Run configuration.
        logger: Optional JSONL run logger for the auditable idea tree.
        verbose: If True, print progress to the console.
    """

    def __init__(
        self,
        env: BaseEnv,
        agents: list[BaseAgent],
        selector: BaseSelector,
        config: OrchestratorConfig | None = None,
        logger: RunLogger | None = None,
        verbose: bool = True,
    ):
        if not agents:
            raise ValueError("At least one agent is required.")
        self.env = env
        self.agents = agents
        self.selector = selector
        self.config = config or OrchestratorConfig()
        self.logger = logger
        self.console = get_console_logger() if verbose else None

        self._arms: list[_Arm] = []
        self._history: list[tuple[Hypothesis, RolloutResult]] = []
        self._trace = ResearchTrace(
            target_success=self.config.target_success, n_agents=len(agents)
        )

    def run(self) -> RunResult:
        """Execute the full autoresearch loop and return the results."""
        cfg = self.config
        wall_clock = 0.0

        for step in range(cfg.max_steps):
            # --- Generation (PI): agents propose new hypotheses on cadence ---
            if step % cfg.propose_every == 0:
                wall_clock += self._propose_round(step)

            if not self._arms:
                continue

            # --- Selection (E): pick which hypothesis to evaluate next ---
            arm_stats = [a.stats for a in self._arms]
            idx = self.selector.select(arm_stats)
            arm = self._arms[idx]

            # --- Rollout (R): evaluate on the (simulated) robot ---
            seed_offset = arm.eval_count * cfg.eval_seed_stride
            t0 = time.perf_counter()
            result = evaluate_hypothesis(
                self.env, arm.hypothesis, cfg.episodes_per_eval, seed_offset
            )
            elapsed = time.perf_counter() - t0
            wall_clock += elapsed
            self._trace.rollout_seconds += elapsed

            arm.stats.update(result.n_episodes, result.n_success)
            arm.eval_count += 1
            self._history.append((arm.hypothesis, result))

            best = self._best_success()
            self._trace.record(step, wall_clock, best)
            self._log_event(
                "rollout",
                step=step,
                wall_clock=wall_clock,
                agent_id=arm.hypothesis.agent_id,
                hyp_id=arm.hypothesis.hyp_id,
                success_rate=result.success_rate,
                arm_mean=arm.stats.mean,
                best_success=best,
            )
            if self.console:
                self.console.info(
                    f"step {step:3d} | eval {arm.hypothesis.hyp_id} "
                    f"(agent {arm.hypothesis.agent_id}) -> "
                    f"sr={result.success_rate:.2f} | best={best:.2f}"
                )

        self._trace.total_tokens = sum(a.tokens_used for a in self.agents)
        return RunResult(
            trace=self._trace,
            arms=self._arms,
            history=self._history,
            config_summary={
                "selector": self.selector.name,
                "n_agents": len(self.agents),
                "max_steps": cfg.max_steps,
                "episodes_per_eval": cfg.episodes_per_eval,
                "target_success": cfg.target_success,
            },
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _propose_round(self, step: int) -> float:
        """Have every agent propose one hypothesis. Returns elapsed seconds."""
        elapsed_total = 0.0
        for agent in self.agents:
            t0 = time.perf_counter()
            hyp = agent.propose(self.config.task_description, self._history)
            elapsed = time.perf_counter() - t0
            elapsed_total += elapsed
            self._trace.thinking_seconds += elapsed
            self._arms.append(_Arm(hypothesis=hyp, stats=ArmStats(hyp_id=hyp.hyp_id)))
            self._log_event(
                "propose",
                step=step,
                agent_id=agent.agent_id,
                hyp_id=hyp.hyp_id,
                parent_id=hyp.parent_id,
                params=hyp.params,
                rationale=hyp.rationale,
            )
        return elapsed_total

    def _best_success(self) -> float:
        """Best empirical success rate among arms that have been evaluated."""
        means = [a.stats.mean for a in self._arms if a.stats.n_pulls > 0]
        return max(means) if means else 0.0

    def _log_event(self, kind: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.log({"event": kind, **fields})
