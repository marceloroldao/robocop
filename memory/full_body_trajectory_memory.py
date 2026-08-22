from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class FullBodyPrototype:
    trajectory: np.ndarray
    target_state: np.ndarray
    confirmations: int = 1
    gain_sum: float = 0.0

    @property
    def mean_gain(self) -> float:
        return float(self.gain_sum / max(1, self.confirmations))


@dataclass(frozen=True)
class FullBodyRecall:
    target_state: np.ndarray
    correction_vector: np.ndarray
    confidence: float
    coherence: float
    neighbors: int
    direct: bool
    rms_distance: float
    max_channel_error: float
    local_density: int
    resolution_factor: float


class FullBodyTrajectoryMemory:
    """V11 memory over raw full-body sensor trajectories.

    The memory compares each channel at every time step. Channel scales are fit
    from training data only. Local density tightens the admissible neighborhood
    as a region accumulates experience, with a non-zero resolution floor.
    """

    def __init__(
        self,
        *,
        context: int = 5,
        min_confirmations: int = 3,
        coarse_rms: float = 1.30,
        direct_rms: float = 0.65,
        interpolation_rms: float = 0.95,
        base_gate: float = 2.25,
        resolution_floor: float = 0.32,
        density_alpha: float = 0.23,
        min_coherence: float = 0.72,
        max_records: int = 6000,
    ) -> None:
        self.context = int(context)
        self.min_confirmations = int(min_confirmations)
        self.coarse_rms = float(coarse_rms)
        self.direct_rms = float(direct_rms)
        self.interpolation_rms = float(interpolation_rms)
        self.base_gate = float(base_gate)
        self.resolution_floor = float(resolution_floor)
        self.density_alpha = float(density_alpha)
        self.min_coherence = float(min_coherence)
        self.max_records = int(max_records)
        self.scales: Optional[np.ndarray] = None
        self._records: list[FullBodyPrototype] = []
        self.admitted = 0
        self.merged = 0

    @property
    def size(self) -> int:
        return len(self._records)

    def fit_scales(self, vectors: Iterable[np.ndarray]) -> None:
        x = np.asarray([np.asarray(v, dtype=float) for v in vectors], dtype=float)
        if x.ndim != 2 or len(x) < 8:
            raise ValueError("V11 needs at least eight training sensor vectors")
        med = np.median(x, axis=0)
        mad = np.median(np.abs(x - med), axis=0)
        q25, q75 = np.percentile(x, [25, 75], axis=0)
        robust = np.maximum(1.4826 * mad, (q75 - q25) / 1.349)
        std = np.std(x, axis=0)
        floor = np.maximum(1e-4, 0.05 * np.where(std > 1e-8, std, 1.0))
        self.scales = np.maximum(robust, floor)

    def trajectory(self, history: Iterable[np.ndarray]) -> np.ndarray:
        h = [np.asarray(v, dtype=float) for v in history]
        if len(h) != self.context:
            raise ValueError(f"expected exactly {self.context} full-body states")
        t = np.stack(h, axis=0)
        if not np.all(np.isfinite(t)):
            raise ValueError("non-finite V11 trajectory")
        return t

    def _z(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.scales is None:
            raise RuntimeError("fit_scales() must be called before using V11 memory")
        return np.abs((a - b) / self.scales[None, :])

    def _rms(self, a: np.ndarray, b: np.ndarray) -> float:
        z = self._z(a, b)
        return float(np.sqrt(np.mean(z * z)))

    def _local_density(self, query: np.ndarray) -> int:
        return sum(self._rms(r.trajectory, query) <= self.coarse_rms for r in self._records)

    def _resolution_factor(self, density: int, confirmations: int = 1) -> float:
        maturity = np.log1p(max(0, density)) + 0.35 * np.log1p(max(1, confirmations))
        factor = 1.0 / (1.0 + self.density_alpha * maturity)
        return float(max(self.resolution_floor, factor))

    def _compatible(self, reference: FullBodyPrototype, query: np.ndarray) -> tuple[float, float, int, float, bool]:
        density = self._local_density(query)
        factor = self._resolution_factor(density, reference.confirmations)
        z = self._z(reference.trajectory, query)
        rms = float(np.sqrt(np.mean(z * z)))
        mx = float(np.max(z))
        gate = self.base_gate * factor
        ok = bool(np.all(z <= gate))
        return rms, mx, density, factor, ok

    def observe(self, history: Iterable[np.ndarray], target_state: np.ndarray, recovery_gain: float) -> bool:
        traj = self.trajectory(history)
        target = np.asarray(target_state, dtype=float)
        if target.ndim != 1 or target.shape[0] != traj.shape[1]:
            return False
        best = None
        for i, r in enumerate(self._records):
            rms, _mx, _d, _f, ok = self._compatible(r, traj)
            if ok and rms <= self.direct_rms:
                if best is None or rms < best[0]:
                    best = (rms, i)
        self.admitted += 1
        if best is None:
            self._records.append(FullBodyPrototype(traj.copy(), target.copy(), 1, float(recovery_gain)))
        else:
            r = self._records[best[1]]
            n = r.confirmations + 1
            r.trajectory += (traj - r.trajectory) / float(n)
            r.target_state += (target - r.target_state) / float(n)
            r.confirmations = n
            r.gain_sum += float(recovery_gain)
            self.merged += 1
        if len(self._records) > self.max_records:
            self._records.sort(key=lambda r: (r.confirmations, r.mean_gain))
            del self._records[: len(self._records) - self.max_records]
        return True

    def recall(self, history: Iterable[np.ndarray], *, k: int = 5, min_confidence: float = 0.40) -> Optional[FullBodyRecall]:
        query = self.trajectory(history)
        current = query[-1]
        ranked = []
        for r in self._records:
            if r.confirmations < self.min_confirmations:
                continue
            rms, mx, density, factor, ok = self._compatible(r, query)
            if ok and rms <= self.interpolation_rms:
                ranked.append((rms, mx, density, factor, r))
        if not ranked:
            return None
        ranked.sort(key=lambda x: x[0])
        rms, mx, density, factor, nearest = ranked[0]
        if rms <= self.direct_rms * factor / max(self.resolution_floor, 0.5):
            cterm = min(1.0, np.log1p(nearest.confirmations) / np.log(21.0))
            conf = float(np.exp(-rms) * (0.72 + 0.28 * cterm))
            if conf >= min_confidence:
                target = nearest.target_state.copy()
                return FullBodyRecall(target, target-current, conf, 1.0, 1, True, rms, mx, density, factor)
        chosen = ranked[:max(2, k)]
        corrections = np.asarray([r.target_state-current for *_x, r in chosen])
        normed = corrections / self.scales[None, :]
        norms = np.linalg.norm(normed, axis=1)
        valid = norms > 1e-9
        if np.count_nonzero(valid) < 2:
            return None
        unit = normed[valid] / norms[valid, None]
        coherence = float(np.linalg.norm(np.mean(unit, axis=0)))
        if coherence < self.min_coherence:
            return None
        weights = np.asarray([
            (1.0 + max(0.0, r.mean_gain)) * np.log1p(r.confirmations) * np.exp(-d / 0.35)
            for d, _mx, _den, _fac, r in chosen
        ])
        weights /= np.sum(weights)
        target = np.sum(np.asarray([r.target_state for *_x, r in chosen]) * weights[:, None], axis=0)
        avg = float(np.average([x[0] for x in chosen], weights=weights))
        mx = float(max(x[1] for x in chosen))
        den = int(round(np.average([x[2] for x in chosen], weights=weights)))
        fac = float(np.average([x[3] for x in chosen], weights=weights))
        conf = float(np.exp(-avg) * coherence)
        if conf < min_confidence:
            return None
        return FullBodyRecall(target, target-current, conf, coherence, len(chosen), False, avg, mx, den, fac)

    def stats(self) -> dict[str, float | int]:
        conf = [r.confirmations for r in self._records]
        return {
            "records": len(conf),
            "confirmed_records": sum(x >= self.min_confirmations for x in conf),
            "mean_confirmations": float(np.mean(conf)) if conf else 0.0,
            "max_confirmations": max(conf) if conf else 0,
            "admitted": self.admitted,
            "merged": self.merged,
        }
