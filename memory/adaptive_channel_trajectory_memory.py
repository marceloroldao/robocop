from __future__ import annotations

import numpy as np

from memory.indexed_full_body_trajectory_memory import IndexedFullBodyTrajectoryMemory
from memory.full_body_trajectory_memory import FullBodyPrototype


class AdaptiveChannelTrajectoryMemory(IndexedFullBodyTrajectoryMemory):
    """Indexed trajectory memory with learned per-channel discriminative weights.

    The learned weights are fixed before holdout. High-weight channels contribute
    more to RMS distance and receive a tighter hard gate; low-weight channels are
    downweighted and receive a looser gate. Mean weight is normalized to 1.
    """

    def __init__(self, *args, channel_weights=None, weight_floor: float = 0.35,
                 weight_ceiling: float = 2.50, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_floor = float(weight_floor)
        self.weight_ceiling = float(weight_ceiling)
        self.channel_weights = None if channel_weights is None else self._normalize_weights(channel_weights)

    def _normalize_weights(self, w):
        x = np.asarray(w, dtype=float)
        if x.ndim != 1 or not np.all(np.isfinite(x)) or np.any(x <= 0):
            raise ValueError("channel_weights must be a finite positive 1-D vector")
        x = np.clip(x, self.weight_floor, self.weight_ceiling)
        x = x / max(1e-12, float(np.mean(x)))
        return x

    def set_channel_weights(self, w) -> None:
        self.channel_weights = self._normalize_weights(w)
        if self.scales is not None and len(self.channel_weights) != len(self.scales):
            raise ValueError("channel_weights length mismatch")
        if self._records:
            self._rebuild_index()

    def fit_scales(self, vectors) -> None:
        super().fit_scales(vectors)
        if self.channel_weights is None:
            self.channel_weights = np.ones_like(self.scales)
        elif len(self.channel_weights) != len(self.scales):
            raise ValueError("channel_weights length mismatch")

    def _sqrt_w(self):
        if self.channel_weights is None:
            if self.scales is None:
                raise RuntimeError("fit_scales() must be called first")
            return np.ones_like(self.scales)
        return np.sqrt(self.channel_weights)

    def _rms(self, a: np.ndarray, b: np.ndarray) -> float:
        z = self._z(a, b)
        w = self.channel_weights[None, :]
        return float(np.sqrt(np.sum(w * z * z) / (z.shape[0] * np.sum(self.channel_weights))))

    def _descriptor(self, trajectory: np.ndarray) -> np.ndarray:
        if self.scales is None:
            raise RuntimeError("fit_scales() must be called before using V11.10")
        sw = self._sqrt_w()
        current = trajectory[-1] / self.scales * sw
        delta = (trajectory[-1] - trajectory[0]) / self.scales * sw
        raw = np.concatenate([current, delta])
        groups = np.array_split(raw, min(16, raw.size))
        return np.asarray([float(np.mean(g)) for g in groups], dtype=float)

    def _compatible_indexed(self, reference: FullBodyPrototype, query: np.ndarray, density: int):
        factor = self._resolution_factor(density, reference.confirmations)
        z = self._z(reference.trajectory, query)
        sw = self._sqrt_w()[None, :]
        wz = z * sw
        rms = float(np.sqrt(np.sum(wz * wz) / (z.shape[0] * np.sum(self.channel_weights))))
        mx = float(np.max(wz))
        # High-weight channels get tighter admissible raw error; low-weight channels looser.
        raw_gate = (self.base_gate * factor) / np.maximum(sw, 1e-12)
        ok = bool(np.all(z <= raw_gate))
        return rms, mx, density, factor, ok

    def weight_stats(self) -> dict[str, float]:
        w = np.asarray(self.channel_weights, float)
        return {
            "min": float(np.min(w)),
            "mean": float(np.mean(w)),
            "max": float(np.max(w)),
            "std": float(np.std(w)),
        }
