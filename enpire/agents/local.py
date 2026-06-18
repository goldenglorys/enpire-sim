"""Local open-source coding agent via Ollama (the democratization comparator).

This backs the open-source side of the frontier-vs-open-source study: run the
ENPIRE loop with a locally hosted model (e.g. DeepSeek-Coder, Qwen2.5-Coder,
CodeLlama) served by Ollama, so physical autoresearch can be benchmarked
without paid API access.

Talks to a local Ollama server over HTTP using only the standard library, so
there is no extra dependency. Start a server with ``ollama serve`` and pull a
model with e.g. ``ollama pull qwen2.5-coder``.
"""

from __future__ import annotations

import json
import urllib.request

from ..policy.base import Hypothesis
from ..rollout.runner import RolloutResult
from .base import BaseAgent
from ._prompts import SYSTEM_PROMPT, build_user_prompt, clamp_params

DEFAULT_MODEL = "qwen2.5-coder"
DEFAULT_HOST = "http://localhost:11434"


class LocalAgent(BaseAgent):
    """Coding agent backed by a local Ollama model.

    Args:
        agent_id: Identifier / branch name.
        model: Ollama model tag (must be pulled locally).
        host: Base URL of the Ollama server.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        agent_id: str,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 120.0,
    ):
        super().__init__(agent_id)
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def propose(
        self,
        task_description: str,
        history: list[tuple[Hypothesis, RolloutResult]],
    ) -> Hypothesis:
        user_prompt = build_user_prompt(task_description, history)
        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": user_prompt
            + "\n\nRespond with ONLY a JSON object, no prose.",
            "format": "json",
            "stream": False,
        }
        data = self._post("/api/generate", payload)

        # Ollama reports token counts as eval/prompt_eval counts.
        self.tokens_used += int(data.get("prompt_eval_count", 0) or 0)
        self.tokens_used += int(data.get("eval_count", 0) or 0)

        try:
            parsed = json.loads(data.get("response", "{}"))
        except json.JSONDecodeError:
            parsed = {}
        parent = self._best_in_history(history)
        return Hypothesis(
            regime="heuristic",
            params=clamp_params(parsed.get("params", {})),
            parent_id=parent.hyp_id if parent else None,
            agent_id=self.agent_id,
            rationale=str(parsed.get("rationale", "")),
        )

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
