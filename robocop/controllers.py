from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return np.zeros_like(v) if n < 1e-12 else v / n


class PDController:
    def __init__(self, kp: float = 0.4, kd: float = 0.03, action_limit: float = 0.4):
        self.kp = kp
        self.kd = kd
        self.action_limit = action_limit

    def action(self, q: np.ndarray, qd: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        q_target = np.asarray(q_target, dtype=float)
        u = self.kp * (q_target - q) - self.kd * qd
        return np.clip(u, -self.action_limit, self.action_limit)


class FieldModulatedController:
    def __init__(self, pd: PDController | None = None, field_gain: float = 0.20):
        self.pd = pd or PDController()
        self.field_gain = field_gain

    def action(self, q, qd, q_target, gradient, confidence: float = 1.0):
        base = self.pd.action(q, qd, q_target)
        correction = self.field_gain * float(np.clip(confidence, 0.0, 1.0)) * normalize(gradient)
        return np.clip(base + correction, -self.pd.action_limit, self.pd.action_limit)
