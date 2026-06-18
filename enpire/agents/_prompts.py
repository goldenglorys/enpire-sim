"""Shared prompt construction and parsing for LLM-backed coding agents.

All LLM agents face the same job: given the task, the tunable parameter space,
and the history of what has been tried (with success rates and failure notes),
emit the next hypothesis as a structured object. Centralising the prompt and
the JSON schema here keeps the Anthropic / OpenAI / local backends thin and
ensures they are compared on identical prompting -- important for the
frontier-vs-open-source study.
"""

from __future__ import annotations

import json

from ..policy.heuristic import HEURISTIC_PARAM_SPACE
from ..policy.base import Hypothesis
from ..rollout.runner import RolloutResult

#: JSON schema describing a hypothesis the agent must return. Used both for
#: Anthropic structured outputs and embedded in prompts for other backends.
HYPOTHESIS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "rationale": {
            "type": "string",
            "description": "One or two sentences explaining the hypothesis.",
        },
        "params": {
            "type": "object",
            "properties": {
                name: {"type": "number"} for name in HEURISTIC_PARAM_SPACE
            },
            "required": list(HEURISTIC_PARAM_SPACE),
            "additionalProperties": False,
        },
    },
    "required": ["rationale", "params"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a robotics research agent performing physical autoresearch in the "
    "style of NVIDIA ENPIRE. You iteratively improve a heuristic Push-T "
    "manipulation policy by proposing new parameter settings, observing their "
    "real-world (simulated) success rates, and refining from failure feedback. "
    "Your objective is to maximise the task success rate. Propose one concrete, "
    "well-reasoned parameter hypothesis at a time. Favour exploring near "
    "high-performing settings while occasionally testing bolder changes when "
    "progress stalls."
)


def build_user_prompt(
    task_description: str,
    history: list[tuple[Hypothesis, RolloutResult]],
    max_history: int = 12,
) -> str:
    """Assemble the user-turn prompt describing the task and recent history.

    Args:
        task_description: Natural-language task description.
        history: Chronological ``(hypothesis, result)`` pairs.
        max_history: How many of the most recent trials to include (keeps the
            prompt bounded and cache-friendly).

    Returns:
        A prompt string for the LLM.
    """
    bounds_lines = [
        f"  - {name}: [{lo}, {hi}]" for name, (lo, hi) in HEURISTIC_PARAM_SPACE.items()
    ]
    parts = [
        task_description.strip(),
        "",
        "Tunable parameters and their valid ranges:",
        *bounds_lines,
        "",
    ]

    if history:
        parts.append("Recent trials (most recent last):")
        for hyp, res in history[-max_history:]:
            params_str = json.dumps({k: round(v, 4) for k, v in hyp.params.items()})
            line = f"  - params={params_str} -> success_rate={res.success_rate:.2f}"
            if res.sample_failure:
                line += f" | failure: {res.sample_failure}"
            parts.append(line)
        best = max(history, key=lambda hr: hr[1].success_rate)
        parts.append("")
        parts.append(
            f"Best so far: success_rate={best[1].success_rate:.2f} with "
            f"params={json.dumps({k: round(v, 4) for k, v in best[0].params.items()})}"
        )
    else:
        parts.append("No trials yet. Propose a sensible starting hypothesis.")

    parts.append("")
    parts.append(
        "Return a single JSON object with a 'rationale' string and a 'params' "
        "object containing every parameter above, each within its range."
    )
    return "\n".join(parts)


def clamp_params(params: dict) -> dict[str, float]:
    """Clamp returned parameters into their valid ranges (a safety constraint).

    Any missing parameter is filled with the midpoint of its range, and any
    out-of-range value is clipped. This makes the agents robust to imperfect
    LLM output.
    """
    out: dict[str, float] = {}
    for name, (lo, hi) in HEURISTIC_PARAM_SPACE.items():
        try:
            val = float(params.get(name, 0.5 * (lo + hi)))
        except (TypeError, ValueError):
            val = 0.5 * (lo + hi)
        out[name] = float(min(max(val, lo), hi))
    return out
