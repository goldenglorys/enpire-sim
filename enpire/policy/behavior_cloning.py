"""Behavior-cloning policy regime (the learned-policy path for RoboCasa).

Push-T can be solved by a hand-tuned heuristic, but RoboCasa kitchen tasks need
a *learned* policy. ENPIRE's coding agents tune learning recipes (behavior
cloning, offline/online RL) there. This module provides the behavior-cloning
regime: a small MLP that maps observations to actions, trained on demonstration
trajectories, with the **hypothesis = training hyperparameters** that the
coding agent proposes and the Evolution module selects over.

``torch`` is imported lazily so the package's offline Push-T path needs no deep
-learning dependency. Install with ``pip install torch`` to use this regime.

Demonstrations are expected as a list of ``(observation, action)`` arrays (e.g.
collected from a scripted policy, teleoperation, or RoboCasa's demo datasets).
Wiring a demo source for a specific RoboCasa task is the main remaining
integration step for the RoboCasa experiments — see the README.
"""

from __future__ import annotations

import numpy as np

from .base import BasePolicy

#: Hyperparameter search space the coding agent tunes for behavior cloning.
#: These are the RoboCasa analogue of HEURISTIC_PARAM_SPACE: the "genome" the
#: Evolution module's bandit selects over. (low, high) bounds; ``hidden_size``
#: and ``n_layers`` are rounded to integers when the net is built.
BC_PARAM_SPACE: dict[str, tuple[float, float]] = {
    "learning_rate": (1e-4, 5e-3),
    "hidden_size": (64, 512),
    "n_layers": (1, 4),
    "weight_decay": (0.0, 1e-3),
    "n_epochs": (5, 60),
    "bc_loss_weight": (0.5, 2.0),  # mirrors ENPIRE's "BC regularization" knob
}


def default_bc_params() -> dict[str, float]:
    """Midpoint of the BC hyperparameter search space."""
    return {k: 0.5 * (lo + hi) for k, (lo, hi) in BC_PARAM_SPACE.items()}


class BehaviorCloningPolicy(BasePolicy):
    """An MLP policy trained by behavior cloning on demonstrations.

    The constructor only stores hyperparameters; call :meth:`fit` with
    demonstrations before using the policy. Compiled from a
    :class:`~enpire.policy.base.Hypothesis` via ``build_policy`` once a demo
    source is wired in.

    Args:
        obs_dim: Observation dimensionality.
        action_dim: Action dimensionality.
        learning_rate: Optimizer learning rate.
        hidden_size: Width of each hidden layer.
        n_layers: Number of hidden layers.
        weight_decay: L2 regularization.
        n_epochs: Training epochs.
        bc_loss_weight: Scale on the BC (MSE) loss term.
        device: Torch device string.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        hidden_size: float = 256,
        n_layers: float = 2,
        weight_decay: float = 0.0,
        n_epochs: float = 30,
        bc_loss_weight: float = 1.0,
        device: str = "cpu",
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.learning_rate = float(learning_rate)
        self.hidden_size = int(round(hidden_size))
        self.n_layers = int(round(n_layers))
        self.weight_decay = float(weight_decay)
        self.n_epochs = int(round(n_epochs))
        self.bc_loss_weight = float(bc_loss_weight)
        self.device = device
        self._net = None
        self._torch = self._import_torch()
        self._build_net()

    @staticmethod
    def _import_torch():
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "BehaviorCloningPolicy requires `pip install torch`. The Push-T "
                "heuristic regime needs no deep-learning dependency."
            ) from exc
        return torch

    def _build_net(self) -> None:
        torch = self._torch
        nn = torch.nn
        layers: list = []
        in_dim = self.obs_dim
        for _ in range(self.n_layers):
            layers += [nn.Linear(in_dim, self.hidden_size), nn.ReLU()]
            in_dim = self.hidden_size
        layers.append(nn.Linear(in_dim, self.action_dim))
        self._net = nn.Sequential(*layers).to(self.device)

    def fit(self, demos: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
        """Train the policy on ``(observation, action)`` demonstration pairs.

        Args:
            demos: List of ``(obs, action)`` numpy arrays.

        Returns:
            A small dict of training diagnostics (final loss).
        """
        torch = self._torch
        if not demos:
            raise ValueError("No demonstrations provided to BehaviorCloningPolicy.fit.")
        obs = torch.tensor(
            np.stack([d[0] for d in demos]), dtype=torch.float32, device=self.device
        )
        act = torch.tensor(
            np.stack([d[1] for d in demos]), dtype=torch.float32, device=self.device
        )
        opt = torch.optim.Adam(
            self._net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = torch.nn.MSELoss()
        final_loss = float("nan")
        self._net.train()
        for _ in range(self.n_epochs):
            opt.zero_grad()
            pred = self._net(obs)
            loss = self.bc_loss_weight * loss_fn(pred, act)
            loss.backward()
            opt.step()
            final_loss = float(loss.item())
        self._net.eval()
        return {"final_loss": final_loss}

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            obs = torch.tensor(
                np.asarray(observation, dtype=np.float32).reshape(1, -1),
                device=self.device,
            )
            return self._net(obs).cpu().numpy().reshape(-1)
