from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .transition_memory import SensorSnapshot, TransitionRecorder, TransitionSample


@dataclass(frozen=True)
class RescuePrototype:
    feature: np.ndarray
    action: np.ndarray
    quality: float
    stability_before: float
    stability_after: float
    energy: float
    reward: float


def _snapshot_feature(snapshot: SensorSnapshot) -> np.ndarray:
    """Sensor description used to retrieve a previously successful transition.

    Coarse body state is kept explicit while joint positions/velocities are softly
    scaled so the search remains sensitive to pose without letting 34 joint
    coordinates dominate height/vertical/omega/vel_z.
    """
    coarse = snapshot.coarse_vector()
    q = np.asarray(snapshot.q, dtype=float) * 0.25
    qd = np.asarray(snapshot.qd, dtype=float) * 0.10
    return np.concatenate([coarse, q, qd])


def _stability_of_snapshot(snapshot: SensorSnapshot) -> float:
    height_term = np.exp(-5.0 * (snapshot.height - 1.40) ** 2)
    vertical_term = (np.clip(snapshot.vertical, -1.0, 1.0) + 1.0) / 2.0
    omega_term = np.exp(-0.35 * snapshot.omega ** 2)
    velz_term = np.exp(-0.50 * snapshot.vel_z ** 2)
    return float(0.30 * height_term + 0.40 * vertical_term + 0.20 * omega_term + 0.10 * velz_term)


class TransitionRescueMemory:
    """Memory of sensor transitions that historically improved balance.

    The memory does not store isolated 'good states'.  Each prototype is a
    complete transition Z_t --a_t--> Z_{t+1}.  At rescue time it retrieves an
    action that previously moved a similar sensor state toward a more stable
    successor state.
    """

    def __init__(self, prototypes: List[RescuePrototype], mean: np.ndarray, std: np.ndarray):
        self.prototypes = list(prototypes)
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        self._matrix = (
            np.vstack([p.feature for p in self.prototypes]) if self.prototypes else np.empty((0, len(self.mean)))
        )
        self._z = (self._matrix - self.mean) / self.std if len(self._matrix) else self._matrix

    @classmethod
    def fit(
        cls,
        recorders: Iterable[TransitionRecorder],
        prefall_window: int = 12,
        min_improvement: float = 0.002,
        min_after_stability: float = 0.62,
    ) -> "TransitionRescueMemory":
        candidates: List[Tuple[np.ndarray, TransitionSample, float, float, float]] = []
        all_features = []

        for recorder in recorders:
            labels = recorder.labels(prefall_window=prefall_window)
            for idx, sample in enumerate(recorder.samples):
                feature = _snapshot_feature(sample.before)
                all_features.append(feature)
                if labels[idx] == 1 or sample.terminated:
                    continue
                before_s = _stability_of_snapshot(sample.before)
                after_s = _stability_of_snapshot(sample.after)
                improvement = after_s - before_s
                if improvement < min_improvement or after_s < min_after_stability:
                    continue
                # Reward is bounded to keep it from numerically dominating the
                # transition objective.  Energy is explicitly penalized.
                quality = (
                    0.50 * after_s
                    + 0.30 * np.clip(improvement / 0.10, 0.0, 1.0)
                    + 0.20 * np.tanh(sample.reward / 5.0)
                    - 0.20 * np.clip(sample.energy / 0.02, 0.0, 2.0)
                )
                candidates.append((feature, sample, float(quality), before_s, after_s))

        if not all_features:
            return cls([], np.zeros(4), np.ones(4))

        feature_matrix = np.vstack(all_features)
        mean = feature_matrix.mean(axis=0)
        std = feature_matrix.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)

        prototypes = [
            RescuePrototype(
                feature=np.asarray(feature, dtype=float),
                action=np.asarray(sample.action, dtype=float).copy(),
                quality=quality,
                stability_before=before_s,
                stability_after=after_s,
                energy=float(sample.energy),
                reward=float(sample.reward),
            )
            for feature, sample, quality, before_s, after_s in candidates
        ]
        return cls(prototypes, mean, std)

    def lookup(self, snapshot: SensorSnapshot, k: int = 12) -> Tuple[Optional[np.ndarray], float, float]:
        """Return (action, expected_stability_gain, distance).

        Nearest candidates are filtered by state similarity, then the best
        historical transition quality is selected with a distance penalty.
        """
        if not self.prototypes:
            return None, 0.0, float("inf")
        z = (_snapshot_feature(snapshot) - self.mean) / self.std
        d = np.sqrt(np.mean((self._z - z) ** 2, axis=1))
        k = max(1, min(int(k), len(d)))
        idxs = np.argpartition(d, k - 1)[:k]
        best_idx = max(
            idxs,
            key=lambda i: self.prototypes[int(i)].quality - 0.18 * float(d[int(i)]),
        )
        p = self.prototypes[int(best_idx)]
        gain = p.stability_after - p.stability_before
        return p.action.copy(), float(gain), float(d[int(best_idx)])

    def stats(self) -> dict:
        if not self.prototypes:
            return {"prototypes": 0, "mean_gain": 0.0, "mean_energy": 0.0}
        return {
            "prototypes": len(self.prototypes),
            "mean_gain": float(np.mean([p.stability_after - p.stability_before for p in self.prototypes])),
            "mean_energy": float(np.mean([p.energy for p in self.prototypes])),
        }
