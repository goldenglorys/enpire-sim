#!/usr/bin/env python3
"""Run the Evolution-module ablation: selectors x fleet sizes x seeds.

This is the core experiment of the thesis. It sweeps the hypothesis-selection
strategy (greedy = ENPIRE baseline, vs. UCB / Thompson / random), optionally
across fleet sizes, repeated over several seeds, and writes a tidy results file
plus an aggregated comparison. Use ``scripts/plot_results.py`` to visualise.

Example (fully offline)::

    python scripts/run_ablation.py --selectors greedy ucb thompson random \\
        --n-agents 4 --seeds 5 --max-steps 80 --out results/ablation.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from enpire.agents import make_agent
from enpire.environment import make_env
from enpire.evolution import make_selector
from enpire.orchestrator import Orchestrator, OrchestratorConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ENPIRE-Sim selector ablation.")
    p.add_argument(
        "--selectors",
        nargs="+",
        default=["greedy", "ucb", "thompson", "random"],
        help="Selectors to compare.",
    )
    p.add_argument(
        "--n-agents",
        nargs="+",
        type=int,
        default=[4],
        help="One or more fleet sizes to sweep.",
    )
    p.add_argument("--seeds", type=int, default=5, help="Number of seeds per cell.")
    p.add_argument("--agent", default="mock", help="Agent backend (default: mock).")
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--episodes-per-eval", type=int, default=8)
    p.add_argument("--target-success", type=float, default=0.95)
    p.add_argument("--out", default="results/ablation.json")
    return p


def run_cell(selector_name, n_agents, seed, agent_kind, config) -> dict:
    """Run one (selector, n_agents, seed) configuration and return its summary."""
    env = make_env("pusht", backend="builtin")
    agents = []
    for i in range(n_agents):
        kwargs = {"seed": seed * 100 + i} if agent_kind == "mock" else {}
        agents.append(make_agent(agent_kind, agent_id=f"agent{i}", **kwargs))
    selector = make_selector(selector_name, seed=seed)
    orchestrator = Orchestrator(env, agents, selector, config, verbose=False)
    result = orchestrator.run()
    return {
        "selector": selector_name,
        "n_agents": n_agents,
        "seed": seed,
        "metrics": result.trace.summary(),
        "best_success_curve": result.trace.best_success_curve,
    }


def aggregate(runs: list[dict]) -> list[dict]:
    """Aggregate per-cell runs into mean/std over seeds for each (selector, n)."""
    groups: dict[tuple[str, int], list[dict]] = {}
    for run in runs:
        groups.setdefault((run["selector"], run["n_agents"]), []).append(run)

    summaries = []
    for (selector, n_agents), cell_runs in sorted(groups.items()):
        finals = [r["metrics"]["final_best_success"] for r in cell_runs]
        solved = [
            r["metrics"]["steps_to_success"]
            for r in cell_runs
            if r["metrics"]["steps_to_success"] is not None
        ]
        summaries.append(
            {
                "selector": selector,
                "n_agents": n_agents,
                "n_seeds": len(cell_runs),
                "final_best_success_mean": statistics.mean(finals),
                "final_best_success_std": statistics.pstdev(finals),
                "solve_rate": len(solved) / len(cell_runs),
                "mean_steps_to_success": statistics.mean(solved) if solved else None,
            }
        )
    return summaries


def main() -> None:
    args = build_parser().parse_args()
    config = OrchestratorConfig(
        max_steps=args.max_steps,
        episodes_per_eval=args.episodes_per_eval,
        target_success=args.target_success,
    )

    runs = []
    total = len(args.selectors) * len(args.n_agents) * args.seeds
    done = 0
    for selector_name in args.selectors:
        for n_agents in args.n_agents:
            for seed in range(args.seeds):
                runs.append(
                    run_cell(selector_name, n_agents, seed, args.agent, config)
                )
                done += 1
                print(
                    f"[{done}/{total}] {selector_name} n_agents={n_agents} "
                    f"seed={seed} -> "
                    f"best={runs[-1]['metrics']['final_best_success']:.2f}"
                )

    summary = aggregate(runs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"runs": runs, "aggregate": summary}, indent=2))

    print("\n=== Aggregate (mean over seeds) ===")
    header = f"{'selector':<10} {'n':>3} {'best_mean':>10} {'solve_rate':>11} {'steps':>8}"
    print(header)
    for s in summary:
        steps = f"{s['mean_steps_to_success']:.1f}" if s["mean_steps_to_success"] else "-"
        print(
            f"{s['selector']:<10} {s['n_agents']:>3} "
            f"{s['final_best_success_mean']:>10.3f} {s['solve_rate']:>11.2f} {steps:>8}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
