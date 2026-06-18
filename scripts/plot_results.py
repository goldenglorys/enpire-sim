#!/usr/bin/env python3
"""Plot ablation results: success-rate-vs-time curves per selector.

Reads the JSON produced by ``run_ablation.py`` and renders the headline figure
of the thesis -- team-average best success rate over research wall-clock time,
one line per selector (averaged over seeds) -- mirroring the hill-climb /
scaling figures in the ENPIRE paper.

Example::

    python scripts/plot_results.py results/ablation.json --out results/curves.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot ENPIRE-Sim ablation curves.")
    p.add_argument("results", help="Path to the ablation JSON file.")
    p.add_argument("--out", default="results/curves.png", help="Output image path.")
    p.add_argument(
        "--x",
        default="step",
        choices=["step", "wall_clock"],
        help="X-axis: research step or wall-clock seconds (default: step).",
    )
    p.add_argument(
        "--n-agents",
        type=int,
        default=None,
        help="If set, only plot runs with this fleet size.",
    )
    return p


def interpolate_curve(curve, grid, x_index):
    """Interpolate a (step, wall_clock, success) curve onto a common grid."""
    if not curve:
        return np.zeros_like(grid, dtype=float)
    xs = np.array([row[x_index] for row in curve], dtype=float)
    ys = np.array([row[2] for row in curve], dtype=float)
    # Step-wise hold (success rate is monotonic-ish best-so-far): use np.interp
    # with edge clamping.
    return np.interp(grid, xs, ys, left=ys[0], right=ys[-1])


def main() -> None:
    args = build_parser().parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(Path(args.results).read_text())
    runs = data["runs"]
    x_index = 0 if args.x == "step" else 1

    # Group curves by selector (optionally filtered by fleet size).
    by_selector: dict[str, list] = defaultdict(list)
    for run in runs:
        if args.n_agents is not None and run["n_agents"] != args.n_agents:
            continue
        by_selector[run["selector"]].append(run["best_success_curve"])

    if not by_selector:
        raise SystemExit("No matching runs to plot.")

    # Build a common x-grid spanning all curves.
    all_x = [row[x_index] for curves in by_selector.values() for c in curves for row in c]
    grid = np.linspace(0, max(all_x) if all_x else 1, 200)

    plt.figure(figsize=(8, 5))
    for selector, curves in sorted(by_selector.items()):
        stacked = np.vstack([interpolate_curve(c, grid, x_index) for c in curves])
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        line, = plt.plot(grid, mean, label=selector, linewidth=2)
        plt.fill_between(grid, mean - std, mean + std, alpha=0.15, color=line.get_color())

    xlabel = "research step" if args.x == "step" else "research wall-clock (s)"
    plt.xlabel(xlabel)
    plt.ylabel("team-avg best success rate")
    plt.title("Evolution-module ablation: success rate vs. research time")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
