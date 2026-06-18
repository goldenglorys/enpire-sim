#!/usr/bin/env python3
"""Run a single ENPIRE-Sim autoresearch experiment.

Example (fully offline, no API keys needed)::

    python scripts/run_experiment.py --agent mock --n-agents 4 --selector ucb \\
        --max-steps 80 --out results/ucb_mock.json

Example (frontier coding agent)::

    export ANTHROPIC_API_KEY=sk-...
    python scripts/run_experiment.py --agent claude --n-agents 1 --selector greedy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Allow running as a plain script without installing the package.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from enpire.agents import make_agent
from enpire.environment import make_env
from enpire.evolution import make_selector
from enpire.orchestrator import Orchestrator, OrchestratorConfig
from enpire.utils.logging import RunLogger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one ENPIRE-Sim experiment.")
    p.add_argument("--env", default="pusht", help="Task name (default: pusht).")
    p.add_argument(
        "--backend",
        default="builtin",
        choices=["builtin", "gym_pusht"],
        help="Environment backend (default: builtin, numpy-only).",
    )
    p.add_argument(
        "--agent",
        default="mock",
        choices=["mock", "claude", "openai", "local"],
        help="Coding-agent backend (default: mock, API-free).",
    )
    p.add_argument("--model", default=None, help="Override the agent's model id.")
    p.add_argument("--n-agents", type=int, default=4, help="Fleet size (default: 4).")
    p.add_argument(
        "--selector",
        default="ucb",
        choices=["greedy", "ucb", "thompson", "random"],
        help="Evolution-module selector (default: ucb).",
    )
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--episodes-per-eval", type=int, default=8)
    p.add_argument("--propose-every", type=int, default=1)
    p.add_argument("--target-success", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0, help="Selector/agent RNG seed.")
    p.add_argument("--out", default=None, help="Path to write the JSON summary.")
    p.add_argument(
        "--log", default=None, help="Optional JSONL path for the full idea tree."
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-step logging.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    env = make_env(args.env, backend=args.backend)

    agent_kwargs = {}
    if args.model:
        agent_kwargs["model"] = args.model
    agents = []
    for i in range(args.n_agents):
        # MockAgent accepts a seed; LLM agents ignore it gracefully.
        kwargs = dict(agent_kwargs)
        if args.agent == "mock":
            kwargs["seed"] = args.seed + i
        agents.append(make_agent(args.agent, agent_id=f"agent{i}", **kwargs))

    selector = make_selector(args.selector, seed=args.seed)

    config = OrchestratorConfig(
        max_steps=args.max_steps,
        episodes_per_eval=args.episodes_per_eval,
        propose_every=args.propose_every,
        target_success=args.target_success,
    )

    logger = RunLogger(args.log) if args.log else None
    orchestrator = Orchestrator(
        env=env,
        agents=agents,
        selector=selector,
        config=config,
        logger=logger,
        verbose=not args.quiet,
    )
    result = orchestrator.run()
    if logger:
        logger.close()

    summary = {
        **result.config_summary,
        "agent_kind": args.agent,
        "seed": args.seed,
        "metrics": result.trace.summary(),
        "best_success_curve": result.trace.best_success_curve,
    }

    print("\n=== Run summary ===")
    for key, value in result.trace.summary().items():
        print(f"  {key}: {value}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote summary to {out_path}")


if __name__ == "__main__":
    main()
