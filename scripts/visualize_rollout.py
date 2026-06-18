#!/usr/bin/env python3
"""Render a single Push-T episode to an animated GIF (or PNG frames).

This is purely for *seeing* the simulation that the autoresearch loop optimises.
The training/ablation scripts run the same environment headlessly (no graphics)
so they can execute thousands of episodes quickly; this script instead plays one
episode and draws it, so you can watch the circular pusher shove the T-block
toward the goal pose.

Examples::

    # Watch the default heuristic policy on a random seed:
    python scripts/visualize_rollout.py --seed 3 --out results/rollout.gif

    # Visualise the best hypothesis discovered by a run's JSONL idea-tree:
    python scripts/run_experiment.py --agent mock --log results/run.jsonl --quiet
    python scripts/visualize_rollout.py --from-log results/run.jsonl --out results/best.gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from enpire.environment.pusht import (
    PushTEnv,
    _BLOCK_HALF,
    _PUSHER_RADIUS,
    _SUCCESS_POS_TOL,
)
from enpire.policy import Hypothesis, build_policy, default_heuristic_params


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render a Push-T episode to a GIF.")
    p.add_argument("--seed", type=int, default=0, help="Episode seed.")
    p.add_argument("--out", default="results/rollout.gif", help="Output GIF path.")
    p.add_argument("--fps", type=int, default=15, help="Frames per second.")
    p.add_argument(
        "--from-log",
        default=None,
        help="JSONL idea-tree from run_experiment.py --log; uses its best params.",
    )
    return p


def _best_params_from_log(path: str) -> dict:
    """Return the params of the highest-success hypothesis recorded in a JSONL log."""
    proposals: dict[str, dict] = {}
    best_hyp, best_sr = None, -1.0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "propose":
            proposals[event["hyp_id"]] = event.get("params", {})
        elif event.get("event") == "rollout":
            sr = event.get("success_rate", 0.0)
            if sr > best_sr and event["hyp_id"] in proposals:
                best_sr, best_hyp = sr, event["hyp_id"]
    if best_hyp is None:
        raise SystemExit("No usable hypotheses found in the log.")
    print(f"Best logged hypothesis {best_hyp} with success_rate={best_sr:.2f}")
    return proposals[best_hyp]


def main() -> None:
    args = build_parser().parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Circle, Rectangle

    params = (
        _best_params_from_log(args.from_log)
        if args.from_log
        else default_heuristic_params()
    )
    policy = build_policy(Hypothesis(regime="heuristic", params=params))

    # Roll out one episode, recording the state at every step.
    env = PushTEnv()
    obs = env.reset(seed=args.seed)
    frames = [obs.copy()]
    for _ in range(env.max_steps):
        result = env.step(policy(obs))
        obs = result.observation
        frames.append(obs.copy())
        if result.terminated or result.truncated:
            break
    success = env.is_success()
    print(f"Episode finished in {len(frames)} steps | success={success}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"Push-T rollout (seed={args.seed})")

    def draw(i: int):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        o = frames[i]
        pusher, block, goal = o[0:2], o[2:4], o[6:8]
        # Goal region (where the block should end up).
        ax.add_patch(Circle(goal, _SUCCESS_POS_TOL, color="green", alpha=0.2))
        ax.plot(*goal, "g+", markersize=12, label="goal")
        # Block (drawn as its bounding square for the simplified physics).
        ax.add_patch(
            Rectangle(
                block - _BLOCK_HALF, 2 * _BLOCK_HALF, 2 * _BLOCK_HALF,
                color="tab:blue", alpha=0.6, label="T-block",
            )
        )
        # Pusher.
        ax.add_patch(Circle(pusher, _PUSHER_RADIUS, color="tab:red", label="pusher"))
        ax.set_title(f"Push-T  step {i}/{len(frames)-1}  success={env.is_success()}")
        ax.legend(loc="upper right", fontsize=8)

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 // args.fps)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=animation.PillowWriter(fps=args.fps))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
