"""Environment (EN) module: agent-operable, self-resetting, self-verifying tasks."""

from .base import BaseEnv, StepResult
from .pusht import PushTEnv, GymPushTEnv


def make_env(name: str = "pusht", backend: str = "builtin", **kwargs) -> BaseEnv:
    """Factory returning an environment instance by name and backend.

    Args:
        name: Task family. ``"pusht"`` (built-in / gym-pusht) or ``"robocasa"``
            (the RoboCasa kitchen-manipulation suite ENPIRE uses in simulation).
        backend: For ``"pusht"``: ``"builtin"`` (numpy-only) or ``"gym_pusht"``
            (high-fidelity official env). Ignored for RoboCasa.
        **kwargs: Forwarded to the environment constructor. For RoboCasa, pass
            e.g. ``task="OpenDrawer"``, ``robots="PandaOmron"``.

    Returns:
        A :class:`BaseEnv` instance.

    Raises:
        ValueError: If the task name is unknown.
    """
    name = name.lower()
    if name in ("pusht", "push-t", "push_t"):
        if backend == "gym_pusht":
            return GymPushTEnv(**kwargs)
        return PushTEnv(**kwargs)
    if name in ("robocasa", "robo_casa", "robocasa365"):
        # Imported lazily: RoboCasa pulls in MuJoCo + robosuite + assets.
        from .robocasa import RoboCasaEnv

        return RoboCasaEnv(**kwargs)
    raise ValueError(
        f"Unknown environment '{name}'. Available: ['pusht', 'robocasa']."
    )


__all__ = ["BaseEnv", "StepResult", "PushTEnv", "GymPushTEnv", "make_env"]
