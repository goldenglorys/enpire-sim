"""ENPIRE-Sim: a simulation-only reproduction and extension of NVIDIA ENPIRE.

ENPIRE (Xiao et al., 2026) is a harness framework in which coding agents
autonomously improve real-world robot manipulation policies through a
closed feedback loop with four modules:

    EN  -- Environment: automatic reset and verification.
    PI  -- Policy Improvement: agents propose and revise policy code.
    R   -- Rollout: policies are evaluated on (real) robots.
    E   -- Evolution: agents compare branches, reuse good recipes, prune bad ones.

This package reproduces that loop entirely in simulation (no robot hardware
required) and adds a *principled* Evolution module: instead of the greedy,
Git-emergent hypothesis selection used by the original work, ENPIRE-Sim
treats hypothesis selection as a multi-armed bandit problem and provides
UCB and Thompson-sampling selectors that can be compared head-to-head
against a greedy baseline.

The package is intentionally runnable with zero API keys and zero heavy
dependencies (only numpy) by using a built-in simplified Push-T environment
and a deterministic ``MockAgent``. Real coding-agent backends (Anthropic
Claude, OpenAI, and local Ollama models) are available when their SDKs and
credentials are configured.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
