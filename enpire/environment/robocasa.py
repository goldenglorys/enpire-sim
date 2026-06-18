"""RoboCasa environment adapter (the simulation ENPIRE uses for sim results).

ENPIRE reports its simulation results on the **RoboCasa365** benchmark — a
large suite of kitchen manipulation tasks built on `robosuite` + MuJoCo. This
module wraps a RoboCasa task behind ENPIRE-Sim's :class:`BaseEnv` interface so
the *same* autoresearch orchestrator, agents, and bandit-based Evolution module
drive policy improvement on RoboCasa exactly as they do on the built-in Push-T.

Important practical notes
-------------------------
* **Heavy dependencies.** RoboCasa requires ``mujoco``, ``robosuite``, and
  ``robocasa`` plus a multi-GB asset download. It is imported lazily so the rest
  of ENPIRE-Sim has no hard dependency on it. See the README for setup.
* **Policy regime.** Unlike Push-T (a tunable heuristic), RoboCasa tasks need a
  *learned* policy — behavior cloning or RL — which is the regime ENPIRE's
  agents actually tuned in simulation. This adapter exposes the env; the policy
  side plugs in through :mod:`enpire.policy` (see the ``behavior_cloning`` regime
  scaffold). A random/scripted policy is supported only for smoke-testing the
  env wiring.
* **API stability.** RoboCasa/robosuite task names, robot names, and ``make``
  kwargs vary across versions. The constructor forwards ``**make_kwargs`` so you
  can match your installed version; defaults target a common RoboCasa kitchen
  configuration.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEnv, StepResult

#: A few representative RoboCasa365 task names (as used in the ENPIRE paper's
#: simulation figure). Pass any valid task via ``task=``.
ROBOCASA_TASKS = (
    "PnPCounterToCabinet",
    "PnPCabinetToCounter",
    "PnPCounterToSink",
    "PnPSinkToCounter",
    "OpenDrawer",
    "CloseDrawer",
    "OpenSingleDoor",
    "TurnOnSinkFaucet",
    "TurnOffStove",
    "CoffeeSetupMug",
)


class RoboCasaEnv(BaseEnv):
    """Adapter exposing a RoboCasa kitchen task through :class:`BaseEnv`.

    Args:
        task: RoboCasa task / robosuite env name (e.g. ``"PnPCounterToCabinet"``).
        robots: Robot model name passed to ``robosuite.make`` (RoboCasa kitchens
            commonly use ``"PandaOmron"``).
        max_steps: Episode horizon (truncation).
        use_image_obs: If True, request camera observations (needed for visual
            policies / BC); if False, use low-dimensional proprio+object state.
        seed: Base RNG seed for randomized resets.
        **make_kwargs: Extra keyword arguments forwarded to ``robosuite.make`` so
            you can match your installed RoboCasa/robosuite version.
    """

    def __init__(
        self,
        task: str = "PnPCounterToCabinet",
        robots: str = "PandaOmron",
        max_steps: int = 500,
        use_image_obs: bool = False,
        seed: int | None = None,
        **make_kwargs: Any,
    ):
        self.task = task
        self.robots = robots
        self.max_steps = max_steps
        self.use_image_obs = use_image_obs
        self._seed = seed
        self._make_kwargs = make_kwargs
        self._step_count = 0
        self._last_success = False

        self._env = self._make_env()
        # Infer action/observation dimensionality from the live env.
        self.action_dim = int(np.prod(self._env.action_spec[0].shape))
        self._last_obs = self._flatten_obs(self._env.reset())
        self.observation_dim = int(self._last_obs.shape[0])

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _make_env(self):
        try:
            import robocasa  # noqa: F401  (registers RoboCasa tasks)
            import robosuite
        except ImportError as exc:  # pragma: no cover - optional heavy dep
            raise ImportError(
                "RoboCasaEnv requires MuJoCo + robosuite + robocasa and a kitchen "
                "asset download. Install with:\n"
                "    pip install robosuite robocasa mujoco\n"
                "    python -m robocasa.scripts.download_kitchen_assets\n"
                "See the README (RoboCasa setup) for details. For a dependency-free "
                "run, use make_env('pusht')."
            ) from exc

        # Resolve a default arm controller config in a version-tolerant way.
        controller_configs = self._make_kwargs.pop("controller_configs", None)
        if controller_configs is None:
            try:  # newer robosuite
                from robosuite import load_composite_controller_config

                controller_configs = load_composite_controller_config(robot=self.robots)
            except Exception:  # pragma: no cover - fall back to robosuite default
                controller_configs = None

        kwargs = dict(
            env_name=self.task,
            robots=self.robots,
            has_renderer=False,
            has_offscreen_renderer=self.use_image_obs,
            use_camera_obs=self.use_image_obs,
            use_object_obs=True,
            reward_shaping=False,  # sparse, success-based reward (matches ENPIRE)
            ignore_done=False,
            horizon=self.max_steps,
        )
        if controller_configs is not None:
            kwargs["controller_configs"] = controller_configs
        kwargs.update(self._make_kwargs)

        import robosuite

        return robosuite.make(**kwargs)

    # ------------------------------------------------------------------ #
    # BaseEnv API
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
            # robosuite seeds its own RNG via numpy; some versions also accept a
            # deterministic_reset flag. We keep it simple and portable.
        self._step_count = 0
        self._last_success = False
        self._last_obs = self._flatten_obs(self._env.reset())
        return self._last_obs

    def step(self, action: np.ndarray) -> StepResult:
        self._step_count += 1
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        obs, reward, done, info = self._env.step(action)
        self._last_obs = self._flatten_obs(obs)
        self._last_success = bool(self._check_success())
        terminated = self._last_success or bool(done)
        truncated = self._step_count >= self.max_steps
        return StepResult(
            observation=self._last_obs,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info={"success": self._last_success, **(info or {})},
        )

    def is_success(self) -> bool:
        return self._last_success

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _check_success(self) -> bool:
        """Query the underlying robosuite task's success predicate."""
        env = self._env
        # robosuite tasks expose `_check_success`; some wrap it differently.
        if hasattr(env, "_check_success"):
            try:
                return bool(env._check_success())
            except Exception:  # pragma: no cover
                return False
        return False

    def _flatten_obs(self, obs: dict) -> np.ndarray:
        """Flatten a robosuite observation dict into a 1-D float vector.

        Image keys are skipped here (visual policies should consume the raw dict
        directly); we concatenate the low-dimensional proprio/object arrays so
        that simple low-dim policies and BC heads have a usable state vector.
        """
        if not isinstance(obs, dict):
            return np.asarray(obs, dtype=np.float64).reshape(-1)
        parts = []
        for key in sorted(obs):
            if key.endswith("image") or obs[key] is None:
                continue
            arr = np.asarray(obs[key], dtype=np.float64).reshape(-1)
            if arr.ndim == 1 and arr.size and arr.size < 4096:
                parts.append(arr)
        return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float64)
