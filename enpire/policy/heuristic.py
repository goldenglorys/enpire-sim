"""A parametrised heuristic Push-T policy.

ENPIRE's simplest policy-improvement regime is *heuristic learning*: the agent
writes a hand-crafted controller (no neural network) and tunes its parameters
against the real-world success rate. We implement exactly such a controller for
Push-T. The free parameters form the "genome" that coding agents mutate and the
Evolution module selects over.

The control idea is a two-phase pursuit:

1. **Approach** -- drive the pusher to a contact point on the *far* side of the
   block relative to the goal, so that pushing the block moves it toward the
   goal.
2. **Push** -- once in contact, push through the block toward the goal, with a
   correction term that nudges the block's orientation toward the goal angle.

Each parameter has a sensible default; the interesting research question (and
the thing the bandit selectors optimise) is which parameter setting maximises
success rate under the environment's stochastic initial states.
"""

from __future__ import annotations

import numpy as np

from .base import BasePolicy


class HeuristicPushTPolicy(BasePolicy):
    """Parametrised non-prehensile pushing controller for Push-T.

    Args:
        approach_offset: How far behind the block (opposite the goal) the pusher
            aims when not yet in contact. Larger values give a wider wind-up.
        push_gain: Proportional gain on the block-to-goal error during pushing.
        angle_gain: Strength of the orientation-correction term.
        contact_threshold: Distance below which the pusher is considered "in
            contact" and switches from approach to push.
        lateral_bias: Sideways bias applied during pushing to break symmetry and
            help with rotation. Can be negative.
    """

    def __init__(
        self,
        approach_offset: float = 0.18,
        push_gain: float = 1.2,
        angle_gain: float = 0.4,
        contact_threshold: float = 0.16,
        lateral_bias: float = 0.0,
    ):
        self.approach_offset = float(approach_offset)
        self.push_gain = float(push_gain)
        self.angle_gain = float(angle_gain)
        self.contact_threshold = float(contact_threshold)
        self.lateral_bias = float(lateral_bias)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Return a target pusher position given the current observation."""
        pusher = observation[0:2]
        block = observation[2:4]
        theta = float(np.arctan2(observation[5], observation[4]))
        goal = observation[6:8]

        to_goal = goal - block
        goal_dist = float(np.linalg.norm(to_goal))
        goal_dir = to_goal / goal_dist if goal_dist > 1e-8 else np.zeros(2)

        # Contact point sits on the side of the block opposite the goal.
        contact_point = block - goal_dir * self.approach_offset

        pusher_to_block = float(np.linalg.norm(block - pusher))
        if pusher_to_block > self.contact_threshold:
            # Approach phase: move to the wind-up point behind the block.
            target = contact_point
        else:
            # Push phase: drive through the block toward the goal.
            lateral = np.array([-goal_dir[1], goal_dir[0]]) * self.lateral_bias
            angle_term = self.angle_gain * np.array([np.cos(theta), np.sin(theta)])
            target = block + goal_dir * self.push_gain * 0.1 + lateral - angle_term * 0.02

        return np.clip(target, 0.0, 1.0)


#: The mutable search space exposed to coding agents, with (low, high) bounds.
#: Agents propose values inside these bounds; the bounds also define the
#: "safety constraints" of the search for the Evolution module.
HEURISTIC_PARAM_SPACE: dict[str, tuple[float, float]] = {
    "approach_offset": (0.08, 0.30),
    "push_gain": (0.4, 2.5),
    "angle_gain": (0.0, 1.0),
    "contact_threshold": (0.10, 0.22),
    "lateral_bias": (-0.05, 0.05),
}


def default_heuristic_params() -> dict[str, float]:
    """Return the midpoint of the search space as a starting hypothesis."""
    return {k: 0.5 * (lo + hi) for k, (lo, hi) in HEURISTIC_PARAM_SPACE.items()}
