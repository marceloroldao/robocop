from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .field import FieldState


@dataclass(frozen=True)
class SensorSnapshot:
    height: float
    vertical: float
    omega: float
    vel_z: float
    q: np.ndarray
    qd: np.ndarray

    @staticmethod
    def from_parts(field_state: FieldState, q, qd) -> "SensorSnapshot":
        return SensorSnapshot(
            height=float(field_state.height),
            vertical=float(field_state.vertical),
            omega=float(field_state.omega),
            vel_z=float(field_state.vel_z),
            q=np.asarray(q, dtype=float).copy(),
            qd=np.asarray(qd, dtype=float).copy(),
        )

    def coarse_vector(self) -> np.ndarray:
        return np.asarray([self.height, self.vertical, self.omega, self.vel_z], dtype=float)


@dataclass(frozen=True)
class TransitionSample:
    before: SensorSnapshot
    action: np.ndarray
    after: SensorSnapshot
    reward: float
    energy: float
    terminated: bool
    step: int

    def delta(self) -> np.ndarray:
        return self.after.coarse_vector() - self.before.coarse_vector()

    def stability(self) -> float:
        """Heuristic balance score used only for diagnostics, not control."""
        height_term = np.exp(-5.0 * (self.after.height - 1.40) ** 2)
        vertical_term = (np.clip(self.after.vertical, -1.0, 1.0) + 1.0) / 2.0
        omega_term = np.exp(-0.35 * self.after.omega ** 2)
        velz_term = np.exp(-0.50 * self.after.vel_z ** 2)
        return float(0.30 * height_term + 0.40 * vertical_term + 0.20 * omega_term + 0.10 * velz_term)


class TransitionRecorder:
    """Stores sensor transitions and labels windows preceding a fall."""

    def __init__(self) -> None:
        self.samples: List[TransitionSample] = []

    def add(self, sample: TransitionSample) -> None:
        self.samples.append(sample)

    def clear(self) -> None:
        self.samples.clear()

    def labels(self, prefall_window: int = 12) -> np.ndarray:
        n = len(self.samples)
        labels = np.zeros(n, dtype=int)
        fall_indices = [i for i, s in enumerate(self.samples) if s.terminated]
        for idx in fall_indices:
            start = max(0, idx - int(prefall_window) + 1)
            labels[start:idx + 1] = 1
        return labels

    def feature_matrix(self) -> np.ndarray:
        if not self.samples:
            return np.empty((0, 12), dtype=float)
        rows = []
        for s in self.samples:
            before = s.before.coarse_vector()
            after = s.after.coarse_vector()
            delta = s.delta()
            rows.append(
                np.concatenate(
                    [before, delta, np.asarray([s.energy, s.reward, s.stability(), np.linalg.norm(s.action)], dtype=float)]
                )
            )
        return np.vstack(rows)


def _safe_standardize(x: np.ndarray):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (x - mean) / std, mean, std


def separation_report(recorders: Iterable[TransitionRecorder], prefall_window: int = 12) -> dict:
    matrices = []
    labels = []
    for recorder in recorders:
        x = recorder.feature_matrix()
        y = recorder.labels(prefall_window=prefall_window)
        if len(x):
            matrices.append(x)
            labels.append(y)
    if not matrices:
        return {"n": 0, "prefall": 0, "stable": 0, "centroid_distance": 0.0, "top_features": []}

    x = np.vstack(matrices)
    y = np.concatenate(labels)
    if not np.any(y == 1) or not np.any(y == 0):
        return {"n": int(len(y)), "prefall": int(np.sum(y == 1)), "stable": int(np.sum(y == 0)), "centroid_distance": 0.0, "top_features": []}

    z, _, _ = _safe_standardize(x)
    c_bad = z[y == 1].mean(axis=0)
    c_good = z[y == 0].mean(axis=0)
    diff = np.abs(c_bad - c_good)
    names = [
        "height", "vertical", "omega", "vel_z",
        "d_height", "d_vertical", "d_omega", "d_vel_z",
        "energy", "reward", "stability", "action_norm",
    ]
    order = np.argsort(diff)[::-1]
    top = [(names[i], float(diff[i])) for i in order[:6]]
    return {
        "n": int(len(y)),
        "prefall": int(np.sum(y == 1)),
        "stable": int(np.sum(y == 0)),
        "centroid_distance": float(np.linalg.norm(c_bad - c_good)),
        "top_features": top,
    }
