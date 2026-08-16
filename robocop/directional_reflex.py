from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DirectionalSnapshot:
    height: float
    vertical: float
    roll: float
    pitch: float
    omega_x: float
    omega_y: float
    omega_z: float
    vel_z: float
    q: np.ndarray
    qd: np.ndarray

    def vector(self) -> np.ndarray:
        return np.asarray([
            self.height, self.vertical, self.roll, self.pitch,
            self.omega_x, self.omega_y, self.omega_z, self.vel_z,
        ], dtype=float)


@dataclass(frozen=True)
class DirectionalTransition:
    before: DirectionalSnapshot
    action: np.ndarray
    after: DirectionalSnapshot
    reward: float
    energy: float
    terminated: bool


@dataclass(frozen=True)
class DirectionalReflexPrototype:
    feature: np.ndarray
    action: np.ndarray
    stability_gain: float
    after_stability: float
    energy: float


def quaternion_roll_pitch(q: np.ndarray) -> Tuple[float, float]:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return 0.0, 0.0
    w, x, y, z = q / n
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr, cosr))
    sinp = 2.0 * (w * y - z * x)
    pitch = float(np.arcsin(np.clip(sinp, -1.0, 1.0)))
    return roll, pitch


def snapshot_from_env(env) -> DirectionalSnapshot:
    data = env.unwrapped.data
    action_dim = int(env.action_space.shape[0])
    qpos = np.asarray(data.qpos, dtype=float)
    qvel = np.asarray(data.qvel, dtype=float)
    quat = qpos[3:7]
    roll, pitch = quaternion_roll_pitch(quat)
    w, x, y, z = quat / max(float(np.linalg.norm(quat)), 1e-12)
    vertical = float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
    return DirectionalSnapshot(
        height=float(qpos[2]),
        vertical=vertical,
        roll=roll,
        pitch=pitch,
        omega_x=float(qvel[3]),
        omega_y=float(qvel[4]),
        omega_z=float(qvel[5]),
        vel_z=float(qvel[2]),
        q=np.array(qpos[7:7 + action_dim], copy=True),
        qd=np.array(qvel[6:6 + action_dim], copy=True),
    )


def stability(s: DirectionalSnapshot) -> float:
    height_term = np.exp(-5.0 * (s.height - 1.40) ** 2)
    vertical_term = (np.clip(s.vertical, -1.0, 1.0) + 1.0) / 2.0
    omega2 = s.omega_x*s.omega_x + s.omega_y*s.omega_y + s.omega_z*s.omega_z
    omega_term = np.exp(-0.35 * omega2)
    velz_term = np.exp(-0.50 * s.vel_z ** 2)
    return float(0.30*height_term + 0.40*vertical_term + 0.20*omega_term + 0.10*velz_term)


def trend_feature(previous: DirectionalSnapshot, current: DirectionalSnapshot) -> np.ndarray:
    coarse = current.vector()
    delta = current.vector() - previous.vector()
    q = np.asarray(current.q, dtype=float) * 0.20
    qd = np.asarray(current.qd, dtype=float) * 0.12
    return np.concatenate([coarse, delta, q, qd])


class DirectionalReflexMemory:
    """Balance reflexes that preserve front/back/left/right fall direction."""

    def __init__(self, prototypes: List[DirectionalReflexPrototype], mean: np.ndarray, std: np.ndarray):
        self.prototypes = list(prototypes)
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        if self.prototypes:
            matrix = np.vstack([p.feature for p in self.prototypes])
            self._z = (matrix - self.mean) / self.std
        else:
            self._z = np.empty((0, len(self.mean)), dtype=float)

    @classmethod
    def fit(cls, episodes: Iterable[List[DirectionalTransition]], prefall_window: int = 12,
            min_improvement: float = 0.003, min_after_stability: float = 0.62):
        features = []
        candidates = []
        for episode in episodes:
            n = len(episode)
            fall = bool(n and episode[-1].terminated)
            prefall_start = max(0, n - int(prefall_window)) if fall else n
            for i in range(1, n):
                sample = episode[i]
                feature = trend_feature(episode[i-1].after, sample.before)
                features.append(feature)
                if i >= prefall_start or sample.terminated:
                    continue
                before_s = stability(sample.before)
                after_s = stability(sample.after)
                gain = after_s - before_s
                if gain < min_improvement or after_s < min_after_stability:
                    continue
                candidates.append((feature, sample, gain, after_s))
        if not features:
            return cls([], np.zeros(16), np.ones(16))
        matrix = np.vstack(features)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        prototypes = [DirectionalReflexPrototype(
            feature=np.asarray(f, dtype=float),
            action=np.asarray(s.action, dtype=float).copy(),
            stability_gain=float(g),
            after_stability=float(a),
            energy=float(s.energy),
        ) for f, s, g, a in candidates]
        return cls(prototypes, mean, std)

    def lookup(self, previous: DirectionalSnapshot, current: DirectionalSnapshot, k: int = 12):
        if not self.prototypes:
            return None, 0.0, float('inf')
        z = (trend_feature(previous, current) - self.mean) / self.std
        d = np.sqrt(np.mean((self._z - z) ** 2, axis=1))
        k = max(1, min(int(k), len(d)))
        idxs = np.argpartition(d, k-1)[:k]
        best = max(idxs, key=lambda i: (
            self.prototypes[int(i)].stability_gain
            + 0.25*self.prototypes[int(i)].after_stability
            - 0.20*np.clip(self.prototypes[int(i)].energy/0.02, 0.0, 2.0)
            - 0.18*float(d[int(i)])
        ))
        p = self.prototypes[int(best)]
        return p.action.copy(), p.stability_gain, float(d[int(best)])

    def stats(self):
        if not self.prototypes:
            return {'prototypes': 0, 'mean_gain': 0.0, 'mean_energy': 0.0}
        return {
            'prototypes': len(self.prototypes),
            'mean_gain': float(np.mean([p.stability_gain for p in self.prototypes])),
            'mean_energy': float(np.mean([p.energy for p in self.prototypes])),
        }
