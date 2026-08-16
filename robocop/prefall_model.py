from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .transition_memory import SensorSnapshot, TransitionRecorder, TransitionSample


@dataclass
class PrefallRiskModel:
    """Nearest-centroid pre-fall detector trained only from past transitions.

    The model intentionally remains small and auditable: transition features are
    standardized, then compared with stable and pre-fall centroids. A risk near
    1 means the observed transition is closer to the pre-fall regime.
    """

    mean: np.ndarray
    std: np.ndarray
    stable_centroid: np.ndarray
    prefall_centroid: np.ndarray

    @staticmethod
    def feature(sample: TransitionSample) -> np.ndarray:
        before = sample.before.coarse_vector()
        delta = sample.delta()
        return np.concatenate(
            [
                before,
                delta,
                np.asarray(
                    [sample.energy, sample.reward, sample.stability(), np.linalg.norm(sample.action)],
                    dtype=float,
                ),
            ]
        )

    @classmethod
    def fit(cls, recorders: Iterable[TransitionRecorder], prefall_window: int = 12) -> "PrefallRiskModel":
        xs = []
        ys = []
        for recorder in recorders:
            x = recorder.feature_matrix()
            y = recorder.labels(prefall_window=prefall_window)
            if len(x):
                xs.append(x)
                ys.append(y)
        if not xs:
            raise ValueError("no transitions supplied")
        x = np.vstack(xs)
        y = np.concatenate(ys)
        if not np.any(y == 0) or not np.any(y == 1):
            raise ValueError("training requires stable and pre-fall transitions")
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-9, 1.0, std)
        z = (x - mean) / std
        return cls(
            mean=mean,
            std=std,
            stable_centroid=z[y == 0].mean(axis=0),
            prefall_centroid=z[y == 1].mean(axis=0),
        )

    def risk(self, sample: TransitionSample) -> float:
        z = (self.feature(sample) - self.mean) / self.std
        d_good = float(np.linalg.norm(z - self.stable_centroid))
        d_bad = float(np.linalg.norm(z - self.prefall_centroid))
        # Smoothly maps relative proximity into [0, 1]. Positive margin means
        # closer to pre-fall than to stable transitions.
        margin = np.clip(d_good - d_bad, -20.0, 20.0)
        return float(1.0 / (1.0 + np.exp(-margin)))


def make_transition(
    before: SensorSnapshot,
    action: np.ndarray,
    after: SensorSnapshot,
    reward: float,
    terminated: bool = False,
    step: int = 0,
) -> TransitionSample:
    action = np.asarray(action, dtype=float)
    return TransitionSample(
        before=before,
        action=action.copy(),
        after=after,
        reward=float(reward),
        energy=float(np.mean(action ** 2)),
        terminated=bool(terminated),
        step=int(step),
    )
