from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FieldState:
    height: float
    vertical: float
    omega: float
    vel_z: float


class ResolutiveField:
    """Small, simulator-agnostic field score used by the current experiments."""

    def __init__(self, target_height: float = 1.40) -> None:
        self.target_height = target_height

    def score(self, s: FieldState) -> float:
        error_h = s.height - self.target_height
        r_height = math.exp(-5.0 * error_h * error_h)
        r_vertical = (max(-1.0, min(1.0, s.vertical)) + 1.0) / 2.0
        r_omega = math.exp(-0.35 * s.omega * s.omega)
        r_recovery_h = math.tanh(2.0 * (-error_h * s.vel_z))
        r_recovery_vertical = math.tanh(-1.5 * (1.0 - s.vertical) * s.omega)
        vel_total_proxy = math.sqrt(s.vel_z * s.vel_z + s.omega * s.omega)
        r_velocity = math.exp(-0.08 * vel_total_proxy * vel_total_proxy)
        return (
            1.00 * r_height
            + 1.60 * r_vertical
            + 0.30 * r_omega
            + 0.50 * r_recovery_h
            + 0.70 * r_recovery_vertical
            + 0.10 * r_velocity
        )
