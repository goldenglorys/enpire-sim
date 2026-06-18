"""Policy abstractions and the ``Hypothesis`` representation (the PI module).

In ENPIRE the Policy Improvement module is where a coding agent proposes and
revises *policy code*. The unit of research is a **hypothesis**: a concrete,
testable change to the policy (a learning regime plus its parameters, or a
snippet of code-as-policy).

We make the hypothesis a first-class, serialisable object so that:

* coding agents (mock or LLM-backed) emit hypotheses as structured JSON;
* the Evolution module can treat each hypothesis as a bandit "arm";
* the whole search tree is auditable, exactly like the idea git-tree in the
  ENPIRE paper (one node per idea tried).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Hypothesis:
    """A single, testable policy proposal.

    Attributes:
        regime: The policy-improvement regime, e.g. ``"heuristic"``,
            ``"behavior_cloning"``, ``"code_as_policy"``. Mirrors the PI regimes
            in the paper (heuristic learning, tool calling, BC, offline/online RL).
        params: Regime-specific parameters (the tunable "genome").
        parent_id: ID of the hypothesis this one was derived from, or ``None``
            for a root idea. Enables reconstructing the idea tree.
        agent_id: Identifier of the agent/branch that proposed it.
        rationale: Natural-language justification produced by the agent.
        hyp_id: Unique identifier (auto-generated).
    """

    regime: str
    params: dict[str, Any]
    parent_id: str | None = None
    agent_id: str | None = None
    rationale: str = ""
    hyp_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary for logging."""
        return {
            "hyp_id": self.hyp_id,
            "regime": self.regime,
            "params": self.params,
            "parent_id": self.parent_id,
            "agent_id": self.agent_id,
            "rationale": self.rationale,
        }


class BasePolicy:
    """Callable policy interface: maps an observation to an action."""

    def __call__(self, observation: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


def build_policy(hypothesis: Hypothesis) -> Callable[[np.ndarray], np.ndarray]:
    """Compile a :class:`Hypothesis` into an executable policy callable.

    This is the bridge between an agent's abstract proposal and something the
    Rollout module can run. New regimes are registered here.

    Args:
        hypothesis: The hypothesis to compile.

    Returns:
        A callable mapping observation -> action.

    Raises:
        ValueError: If the regime is not supported.
    """
    # Imported here to avoid a circular import at module load time.
    from .heuristic import HeuristicPushTPolicy

    if hypothesis.regime in ("heuristic", "code_as_policy", "tool_calling"):
        return HeuristicPushTPolicy(**hypothesis.params)
    raise ValueError(
        f"Unsupported policy regime '{hypothesis.regime}'. "
        "Supported: heuristic | code_as_policy | tool_calling."
    )
