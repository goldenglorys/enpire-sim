"""Evolution (E) module: bandit-based hypothesis selectors.

Exposes a registry so experiments can choose a selector by name from config.
"""

from .base import ArmStats, BaseSelector
from .greedy import GreedySelector
from .ucb import UCBSelector
from .thompson import ThompsonSelector
from .random_selector import RandomSelector

#: Maps config-friendly names to selector classes.
SELECTOR_REGISTRY: dict[str, type[BaseSelector]] = {
    GreedySelector.name: GreedySelector,
    UCBSelector.name: UCBSelector,
    ThompsonSelector.name: ThompsonSelector,
    RandomSelector.name: RandomSelector,
}


def make_selector(name: str, seed: int | None = None, **kwargs) -> BaseSelector:
    """Instantiate a selector by name.

    Args:
        name: One of ``"greedy"``, ``"ucb"``, ``"thompson"``, ``"random"``.
        seed: RNG seed passed to the selector.
        **kwargs: Selector-specific keyword arguments (e.g. ``c`` for UCB,
            ``epsilon`` for greedy).

    Returns:
        A :class:`BaseSelector` instance.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    key = name.lower()
    if key not in SELECTOR_REGISTRY:
        raise ValueError(
            f"Unknown selector '{name}'. Available: {sorted(SELECTOR_REGISTRY)}."
        )
    return SELECTOR_REGISTRY[key](seed=seed, **kwargs)


__all__ = [
    "ArmStats",
    "BaseSelector",
    "GreedySelector",
    "UCBSelector",
    "ThompsonSelector",
    "RandomSelector",
    "SELECTOR_REGISTRY",
    "make_selector",
]
