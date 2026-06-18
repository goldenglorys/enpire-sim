"""Coding-agent abstraction (the researcher that drives Policy Improvement).

In ENPIRE a *coding agent* (Codex, Claude Code, Kimi Code) reads logs, consults
the literature, and writes/revises policy code. In ENPIRE-Sim an agent is any
object that, given the task description and the history of what has been tried,
proposes the next :class:`~enpire.policy.base.Hypothesis`.

We separate the agent (which *generates* hypotheses) from the Evolution module
(which *selects* which hypothesis to invest budget in). This separation is the
key design choice that makes the bandit ablation clean: every selector is
compared on the same stream of agent-proposed hypotheses.

Two families of agents are provided:

* :class:`~enpire.agents.mock.MockAgent` -- a deterministic, API-free local
  search that mutates good hypotheses. Lets the whole system run for free and
  makes tests reproducible.
* LLM-backed agents (:mod:`enpire.agents.claude`, :mod:`enpire.agents.openai`,
  :mod:`enpire.agents.local`) -- real coding agents that emit hypotheses as
  JSON. Used for the frontier-vs-open-source comparison.
"""

from __future__ import annotations

import abc

from ..policy.base import Hypothesis
from ..rollout.runner import RolloutResult


class BaseAgent(abc.ABC):
    """Abstract coding agent that proposes hypotheses from feedback.

    Attributes:
        agent_id: Unique identifier; also names this agent's branch in logs.
        tokens_used: Cumulative LLM tokens consumed (0 for non-LLM agents). Used
            to compute Mean Token Utilization (MTU).
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.tokens_used = 0

    @abc.abstractmethod
    def propose(
        self,
        task_description: str,
        history: list[tuple[Hypothesis, RolloutResult]],
    ) -> Hypothesis:
        """Propose the next hypothesis to try.

        Args:
            task_description: Natural-language description of the task and goal.
            history: Chronological list of ``(hypothesis, result)`` pairs this
                agent (and, in the shared-pool setting, its peers) has observed.
                May be empty on the first call.

        Returns:
            A new :class:`Hypothesis` tagged with this agent's ``agent_id``.
        """

    def _best_in_history(
        self, history: list[tuple[Hypothesis, RolloutResult]]
    ) -> Hypothesis | None:
        """Return the highest-success-rate hypothesis seen so far, if any."""
        if not history:
            return None
        best_hyp, best_res = max(history, key=lambda hr: hr[1].success_rate)
        return best_hyp
