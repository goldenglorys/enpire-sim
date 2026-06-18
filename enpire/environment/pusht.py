"""A self-contained, dependency-light Push-T environment.

Push-T is the canonical task used by ENPIRE to benchmark coding agents
(they clone ``huggingface/gym-pusht``). The real task: a circular pusher must
shove a T-shaped block until it overlaps a fixed goal pose, using only
*non-prehensile* contact (you can push but not grasp).

This module provides a **simplified quasi-static** re-implementation that runs
with only numpy so the whole ENPIRE loop is runnable out of the box. The
physics are approximate -- contact translates the block along the push
direction and rotates it according to the lever arm between the contact point
and the block's centre of mass. This is *not* a faithful rigid-body simulation,
but it is a genuine closed-loop control task whose success rate depends
smoothly on the policy parameters, which is exactly what we need to study
hypothesis-selection (the Evolution module).

For high-fidelity experiments, set ``EnpireConfig.environment.backend =
"gym_pusht"`` (requires ``pip install gym-pusht``); :class:`GymPushTEnv` wraps
the official environment behind the same interface.

Observation layout (length 8), all in normalized [0, 1] world coordinates::

    [pusher_x, pusher_y, block_x, block_y, cos(theta), sin(theta),
     goal_x, goal_y]

Action layout (length 2): the *target* pusher position in [0, 1]^2. The pusher
moves toward the target with a capped speed each step, mimicking a position
controller.
"""

from __future__ import annotations

import numpy as np

from .base import BaseEnv, StepResult

# World is the unit square [0, 1] x [0, 1].
_WORLD = 1.0
_PUSHER_RADIUS = 0.05
_PUSHER_SPEED = 0.06  # max distance the pusher travels per step
_BLOCK_HALF = 0.10  # half-extent of the T bounding region
_SUCCESS_POS_TOL = 0.06  # position tolerance for success (world units)
_SUCCESS_ANG_TOL = 0.35  # orientation tolerance for success (radians)


class PushTEnv(BaseEnv):
    """Simplified quasi-static Push-T task (numpy-only).

    Args:
        max_steps: Episode truncation horizon.
        push_gain: How strongly contact translates the block. Exposed mostly
            for testing; the default approximates a light, slippery block.
    """

    observation_dim = 8
    action_dim = 2

    def __init__(self, max_steps: int = 200, push_gain: float = 0.9):
        self.max_steps = max_steps
        self.push_gain = push_gain
        self._rng = np.random.default_rng(0)
        self._step_count = 0
        self.pusher = np.array([0.5, 0.1])
        self.block = np.array([0.5, 0.5])
        self.theta = 0.0
        self.goal = np.array([0.5, 0.7])
        self.goal_theta = 0.0

    # ------------------------------------------------------------------ #
    # Gym-style API
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset to a randomized initial state (automated reset of EN module)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        # Pusher starts near the bottom edge.
        self.pusher = np.array([self._rng.uniform(0.3, 0.7), 0.1])
        # Block starts somewhere in the lower-middle region.
        self.block = self._rng.uniform([0.3, 0.35], [0.7, 0.55])
        self.theta = float(self._rng.uniform(-0.4, 0.4))
        # Goal is fixed near the top so the task is comparable across seeds.
        self.goal = np.array([0.5, 0.75])
        self.goal_theta = 0.0
        return self._obs()

    def step(self, action: np.ndarray) -> StepResult:
        """Move the pusher toward ``action`` and resolve contact with the block."""
        self._step_count += 1
        target = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)

        # Move pusher toward the commanded target with capped speed.
        delta = target - self.pusher
        dist = float(np.linalg.norm(delta))
        if dist > _PUSHER_SPEED:
            delta = delta / dist * _PUSHER_SPEED
        self.pusher = np.clip(self.pusher + delta, 0.0, 1.0)

        # Resolve contact: if the pusher penetrates the block region, push it.
        self._resolve_contact()

        terminated = self.is_success()
        truncated = self._step_count >= self.max_steps
        reward = 1.0 if terminated else 0.0
        info = {
            "pos_error": float(np.linalg.norm(self.block - self.goal)),
            "ang_error": float(abs(_wrap_angle(self.theta - self.goal_theta))),
            "step": self._step_count,
        }
        return StepResult(self._obs(), reward, terminated, truncated, info)

    def is_success(self) -> bool:
        """Verification: block within position *and* orientation tolerance."""
        pos_ok = float(np.linalg.norm(self.block - self.goal)) <= _SUCCESS_POS_TOL
        ang_ok = abs(_wrap_angle(self.theta - self.goal_theta)) <= _SUCCESS_ANG_TOL
        return bool(pos_ok and ang_ok)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _resolve_contact(self) -> None:
        """Approximate quasi-static pushing.

        If the pusher overlaps the block's bounding circle, translate the block
        along the contact normal and apply a small rotation proportional to the
        lever arm between the contact point and the block centre.
        """
        offset = self.block - self.pusher
        dist = float(np.linalg.norm(offset))
        contact_dist = _PUSHER_RADIUS + _BLOCK_HALF
        if dist >= contact_dist or dist < 1e-8:
            return

        normal = offset / dist
        penetration = contact_dist - dist
        # Translate block out of penetration, scaled by push gain.
        self.block = np.clip(
            self.block + normal * penetration * self.push_gain, 0.0, 1.0
        )
        # Rotate: lever arm is the tangential component of the contact offset.
        tangent = np.array([-normal[1], normal[0]])
        lever = float(np.dot(self.pusher - self.block, tangent))
        self.theta = _wrap_angle(self.theta + 0.5 * lever * self.push_gain)

    def _obs(self) -> np.ndarray:
        return np.array(
            [
                self.pusher[0],
                self.pusher[1],
                self.block[0],
                self.block[1],
                np.cos(self.theta),
                np.sin(self.theta),
                self.goal[0],
                self.goal[1],
            ],
            dtype=np.float64,
        )


def _wrap_angle(a: float) -> float:
    """Wrap an angle to the range (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class GymPushTEnv(BaseEnv):
    """Thin adapter around the official ``gym-pusht`` environment.

    This is provided so high-fidelity experiments share the exact interface of
    the built-in :class:`PushTEnv`. It is imported lazily so that the package
    has no hard dependency on ``gym-pusht`` / ``pygame`` / ``pymunk``.
    """

    observation_dim = 5
    action_dim = 2

    def __init__(self, max_steps: int = 300, render_mode: str | None = None):
        try:
            import gym_pusht  # noqa: F401  (registers the env)
            import gymnasium as gym
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "GymPushTEnv requires `pip install gym-pusht gymnasium`. "
                "Use backend='builtin' for the dependency-free environment."
            ) from exc

        self._gym = gym
        self._env = gym.make("gym_pusht/PushT-v0", render_mode=render_mode)
        self.max_steps = max_steps
        self._last_obs = None
        self._success = False

    def reset(self, seed: int | None = None) -> np.ndarray:  # pragma: no cover
        obs, _ = self._env.reset(seed=seed)
        self._last_obs = np.asarray(obs, dtype=np.float64)
        self._success = False
        return self._last_obs

    def step(self, action: np.ndarray) -> StepResult:  # pragma: no cover
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._last_obs = np.asarray(obs, dtype=np.float64)
        # gym-pusht exposes coverage in info; treat high coverage as success.
        coverage = float(info.get("coverage", reward))
        self._success = coverage >= 0.95
        return StepResult(self._last_obs, float(reward), bool(terminated), bool(truncated), dict(info))

    def is_success(self) -> bool:  # pragma: no cover
        return self._success
