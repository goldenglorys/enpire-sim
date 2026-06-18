"""OpenAI-backed coding agent (a second frontier comparator).

Plays the role of Codex / GPT in the ENPIRE benchmark. Uses the official
``openai`` SDK in JSON mode. Imported lazily; requires ``pip install openai``
and ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import json

from ..policy.base import Hypothesis
from ..rollout.runner import RolloutResult
from .base import BaseAgent
from ._prompts import SYSTEM_PROMPT, build_user_prompt, clamp_params

DEFAULT_MODEL = "gpt-4o"


class OpenAIAgent(BaseAgent):
    """Coding agent backed by OpenAI chat models.

    Args:
        agent_id: Identifier / branch name.
        model: OpenAI model name.
        max_tokens: Output token cap per proposal.
    """

    def __init__(self, agent_id: str, model: str = DEFAULT_MODEL, max_tokens: int = 1024):
        super().__init__(agent_id)
        self.model = model
        self.max_tokens = max_tokens
        self._client = self._make_client()

    @staticmethod
    def _make_client():
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "OpenAIAgent requires `pip install openai` and OPENAI_API_KEY."
            ) from exc
        return openai.OpenAI()

    def propose(
        self,
        task_description: str,
        history: list[tuple[Hypothesis, RolloutResult]],
    ) -> Hypothesis:
        user_prompt = build_user_prompt(task_description, history)
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.tokens_used += int(getattr(usage, "total_tokens", 0) or 0)

        data = json.loads(response.choices[0].message.content or "{}")
        parent = self._best_in_history(history)
        return Hypothesis(
            regime="heuristic",
            params=clamp_params(data.get("params", {})),
            parent_id=parent.hyp_id if parent else None,
            agent_id=self.agent_id,
            rationale=str(data.get("rationale", "")),
        )
