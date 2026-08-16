from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .transition_memory import SensorSnapshot, TransitionRecorder
from .transition_rescue import _stability_of_snapshot


@dataclass(frozen=True)
class BalanceReflexPrototype:
    feature: np.ndarray
    action: np.ndarray
    stability_gain: float
    after_stability: float
    energy: float
    reward: float


def _trend_feature(before: SensorSnapshot, after: SensorSnapshot) -> np.ndarray:
    """Describe posture plus motion tendency.

    The key difference from static-state lookup is that the feature explicitly
    includes the direction and rate of change of the body state, so two similar
    poses moving in opposite directions map to different reflexes.
    """
    coarse = after.coarse_vector()
    delta = after.coarse_vector() - before.coarse_vector()
    q = np.asarray(after.q, dtype=float) * 0.20
    qd = np.asarray(after.qd, dtype=float) * 0.12
    return np.concatenate([coarse, delta, q, qd])


class BalanceReflexMemory:
    """Memory of corrective reflexes indexed by state and motion tendency."""

    def __init__(self, prototypes: List[BalanceReflexPrototype], mean: np.ndarray, std: np.ndarray):
        self.prototypes = list(prototypes)
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        if self.prototypes:
            matrix = np.vstack([p.feature for p in self.prototypes])
            self._z = (matrix - self.mean) / self.std
        else:
            self._z = np.empty((0, len(self.mean)), dtype=float)

    @classmethod
    def fit(
        cls,
        recorders: Iterable[TransitionRecorder],
        prefall_window: int = 12,
        min_improvement: float = 0.003,
        min_after_stability: float = 0.62,
    ) -> "BalanceReflexMemory":
        features = []
        candidates = []
        for recorder in recorders:
            labels = recorder.labels(prefall_window=prefall_window)
            samples = recorder.samples
            for i in range(1, len(samples)):
                prev = samples[i - 1]
                sample = samples[i]
                feature = _trend_feature(prev.after, sample.before)
                features.append(feature)
                if labels[i] == 1 or sample.terminated:
                    continue
                before_s = _stability_of_snapshot(sample.before)
                after_s = _stability_of_snapshot(sample.after)
                gain = after_s - before_s
                if gain < min_improvement or after_s < min_after_stability:
                    continue
                candidates.append((feature, sample, gain, after_s))

        if not features:
            return cls([], np.zeros(8), np.ones(8))

        matrix = np.vstack(features)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        prototypes = [
            BalanceReflexPrototype(
                feature=np.asarray(feature, dtype=float),
                action=np.asarray(sample.action, dtype=float).copy(),
                stability_gain=float(gain),
                after_stability=float(after_s),
                energy=float(sample.energy),
                reward=float(sample.reward),
            )
            for feature, sample, gain, after_s in candidates
        ]
        return cls(prototypes, mean, std)

    def lookup(
        self,
        previous: SensorSnapshot,
        current: SensorSnapshot,
        k: int = 12,
    ) -> Tuple[Optional[np.ndarray], float, float]:
        if not self.prototypes:
            return None, 0.0, float("inf")
        z = (_trend_feature(previous, current) - self.mean) / self.std
        d = np.sqrt(np.mean((self._z - z) ** 2, axis=1))
        k = max(1, min(int(k), len(d)))
        idxs = np.argpartition(d, k - 1)[:k]
        best_idx = max(
            idxs,
            key=lambda i: (
                self.prototypes[int(i)].stability_gain
                + 0.25 * self.prototypes[int(i)].after_stability
                - 0.20 * np.clip(self.prototypes[int(i)].energy / 0.02, 0.0, 2.0)
                - 0.18 * float(d[int(i)])
            ),
        )
        p = self.prototypes[int(best_idx)]
        return p.action.copy(), float(p.stability_gain), float(d[int(best_idx)])

    def stats(self) -> dict:
        if not self.prototypes:
            return {"prototypes": 0, "mean_gain": 0.0, "mean_energy": 0.0}
        return {
            "prototypes": len(self.prototypes),
            "mean_gain": float(np.mean([p.stability_gain for p in self.prototypes])),
            "mean_energy": float(np.mean([p.energy for p in self.prototypes])),
        }
