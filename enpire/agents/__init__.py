"""Coding agents that drive Policy Improvement (the researchers).

Only :class:`MockAgent` is imported eagerly because it has no external
dependencies. The LLM-backed agents are imported lazily via :func:`make_agent`
so that importing this package never requires ``anthropic`` / ``openai``.
"""

from .base import BaseAgent
from .mock import MockAgent


def make_agent(kind: str, agent_id: str, **kwargs) -> BaseAgent:
    """Instantiate an agent by kind.

    Args:
        kind: One of ``"mock"``, ``"claude"``, ``"openai"``, ``"local"``.
        agent_id: Identifier / branch name for the agent.
        **kwargs: Backend-specific keyword arguments (e.g. ``model``).

    Returns:
        A :class:`BaseAgent` instance.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    kind = kind.lower()
    if kind == "mock":
        return MockAgent(agent_id, **kwargs)
    if kind == "claude":
        from .claude import ClaudeAgent

        return ClaudeAgent(agent_id, **kwargs)
    if kind == "openai":
        from .openai import OpenAIAgent

        return OpenAIAgent(agent_id, **kwargs)
    if kind == "local":
        from .local import LocalAgent

        return LocalAgent(agent_id, **kwargs)
    raise ValueError(
        f"Unknown agent kind '{kind}'. Available: mock | claude | openai | local."
    )


__all__ = ["BaseAgent", "MockAgent", "make_agent"]
