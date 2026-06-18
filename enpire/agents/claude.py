"""Anthropic Claude-backed coding agent (the frontier agent in ENPIRE).

This mirrors the role Claude Code plays in the original ENPIRE experiments: a
frontier coding agent that reads feedback and proposes the next policy change.
It uses the official ``anthropic`` SDK with structured outputs so the returned
hypothesis is schema-valid JSON.

The SDK is imported lazily so the package has no hard dependency on
``anthropic`` (the MockAgent path runs without it). Requires the
``ANTHROPIC_API_KEY`` environment variable, or any credential the SDK resolves
by default.
"""

from __future__ import annotations

import json

from ..policy.base import Hypothesis
from ..rollout.runner import RolloutResult
from .base import BaseAgent
from ._prompts import (
    HYPOTHESIS_SCHEMA,
    SYSTEM_PROMPT,
    build_user_prompt,
    clamp_params,
)

#: Default model. Opus 4.8 is Anthropic's most capable model and the closest
#: analogue to the frontier coding agents benchmarked in ENPIRE.
DEFAULT_MODEL = "claude-opus-4-8"


class ClaudeAgent(BaseAgent):
    """Coding agent backed by Anthropic's Claude models.

    Args:
        agent_id: Identifier / branch name.
        model: Claude model ID (defaults to ``claude-opus-4-8``).
        max_tokens: Output token cap for each proposal.
        effort: Reasoning effort level (``low`` | ``medium`` | ``high`` | ``max``).
    """

    def __init__(
        self,
        agent_id: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2048,
        effort: str = "medium",
    ):
        super().__init__(agent_id)
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self._client = self._make_client()

    @staticmethod
    def _make_client():
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "ClaudeAgent requires `pip install anthropic` and an "
                "ANTHROPIC_API_KEY. Use MockAgent for an API-free run."
            ) from exc
        return anthropic.Anthropic()

    def propose(
        self,
        task_description: str,
        history: list[tuple[Hypothesis, RolloutResult]],
    ) -> Hypothesis:
        user_prompt = build_user_prompt(task_description, history)
        # Adaptive thinking + structured outputs: the response is guaranteed to
        # be JSON matching HYPOTHESIS_SCHEMA.
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": HYPOTHESIS_SCHEMA},
            },
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Track token usage for the Mean Token Utilization (MTU) metric.
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.tokens_used += int(getattr(usage, "input_tokens", 0) or 0)
            self.tokens_used += int(getattr(usage, "output_tokens", 0) or 0)

        text = next((b.text for b in response.content if b.type == "text"), "{}")
        data = json.loads(text)
        parent = self._best_in_history(history)
        return Hypothesis(
            regime="heuristic",
            params=clamp_params(data.get("params", {})),
            parent_id=parent.hyp_id if parent else None,
            agent_id=self.agent_id,
            rationale=str(data.get("rationale", "")),
        )
