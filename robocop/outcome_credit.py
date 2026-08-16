from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .field import FieldState


@dataclass
class PendingTrajectory:
    state: FieldState
    gradient: np.ndarray
    action_energy: float


@dataclass
class RealizedOutcome:
    state: FieldState
    gradient: np.ndarray
    energy: float
    reward: float
    survival: float


class OutcomeCredit:
    """Assign the *realized next-step outcome* to the trajectory that produced it.

    This prevents trajectory memory from learning the energy of the PD baseline or
    an instantaneous field proxy instead of the actual action/outcome pair.
    """

    def __init__(self) -> None:
        self.pending: Optional[PendingTrajectory] = None

    def reset(self) -> None:
        self.pending = None

    def arm(self, state: FieldState, gradient, action) -> None:
        action = np.asarray(action, dtype=float)
        self.pending = PendingTrajectory(
            state=state,
            gradient=np.asarray(gradient, dtype=float).copy(),
            action_energy=float(np.mean(action ** 2)),
        )

    def resolve(self, reward: float, terminated: bool, truncated: bool = False):
        if self.pending is None:
            return None
        pending = self.pending
        self.pending = None
        return RealizedOutcome(
            state=pending.state,
            gradient=pending.gradient.copy(),
            energy=pending.action_energy,
            reward=float(reward),
            survival=0.0 if (terminated or truncated) else 1.0,
        )
