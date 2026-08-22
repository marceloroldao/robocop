from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from memory.transition_memory import BalanceState, stability_score


@dataclass
class FullTrajectoryPrototype:
    trajectory: np.ndarray  # [time, sensor]
    target_state: np.ndarray
    recovery_gain: float
    confirmations: int = 1
    gain_sum: float = 0.0

    def __post_init__(self) -> None:
        if self.gain_sum == 0.0:
            self.gain_sum = float(self.recovery_gain)

    @property
    def mean_gain(self) -> float:
        return float(self.gain_sum / max(1, self.confirmations))


@dataclass(frozen=True)
class FullTrajectoryRecall:
    target_state: np.ndarray
    correction_vector: np.ndarray
    confidence: float
    coherence: float
    neighbors: int
    direct: bool
    rms_distance: float
    max_channel_error: float
    local_density: float
    resolution_factor: float


class FullSensorTrajectoryMemory:
    """V10.4: compare full sensor histories point-by-point.

    A neighbor is valid only when every normalized sensor at every time step is
    inside a locally adaptive hard gate. Local memory density tightens the gate.
    Aggregate RMS is used only to rank already-compatible trajectories.
    """

    def __init__(
        self,
        *,
        target_height: float = 1.0,
        min_recovery_gain: float = 0.03,
        min_degradation: float = 0.01,
        min_confirmations: int = 3,
        base_gate: float = 2.0,
        min_gate_factor: float = 0.32,
        density_alpha: float = 0.22,
        broad_density_gate: float = 3.0,
        merge_rms: float = 0.55,
        direct_rms: float = 0.70,
        interpolation_rms: float = 1.0,
        min_coherence: float = 0.75,
        max_records: int = 5000,
    ) -> None:
        self.target_height = float(target_height)
        self.min_recovery_gain = float(min_recovery_gain)
        self.min_degradation = float(min_degradation)
        self.min_confirmations = int(min_confirmations)
        self.base_gate = float(base_gate)
        self.min_gate_factor = float(min_gate_factor)
        self.density_alpha = float(density_alpha)
        self.broad_density_gate = float(broad_density_gate)
        self.merge_rms = float(merge_rms)
        self.direct_rms = float(direct_rms)
        self.interpolation_rms = float(interpolation_rms)
        self.min_coherence = float(min_coherence)
        self.max_records = int(max_records)
        self.sensor_scale = np.asarray([0.10, 0.25, 0.25, 0.60, 0.35, 0.10], dtype=float)
        self._records: list[FullTrajectoryPrototype] = []
        self.candidates = self.admitted = self.merged = 0

    def trajectory(self, history: Iterable[BalanceState]) -> np.ndarray:
        h = list(history)
        if len(h) < 3:
            raise ValueError("full trajectory address requires at least three states")
        return np.asarray([s.vector() for s in h], dtype=float)

    def _normalized_error(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.shape != b.shape:
            return np.full((1, 1), np.inf)
        return np.abs((a - b) / self.sensor_scale[None, :])

    def _local_density(self, query: np.ndarray) -> float:
        density = 0.0
        for r in self._records:
            if r.trajectory.shape != query.shape:
                continue
            z = self._normalized_error(r.trajectory, query)
            if np.all(z <= self.broad_density_gate):
                density += float(r.confirmations)
        return density

    def _resolution_factor(self, density: float) -> float:
        factor = 1.0 / (1.0 + self.density_alpha * np.log1p(max(0.0, density)))
        return float(max(self.min_gate_factor, factor))

    def _distance(self, a: np.ndarray, b: np.ndarray, gate: float) -> tuple[float, float, bool]:
        z = self._normalized_error(a, b)
        if not np.all(np.isfinite(z)):
            return float("inf"), float("inf"), False
        rms = float(np.sqrt(np.mean(z * z)))
        mx = float(np.max(z))
        return rms, mx, bool(np.all(z <= gate))

    def observe_window(self, history: Iterable[BalanceState], future_states: Iterable[BalanceState]) -> bool:
        hist, future = list(history), list(future_states)
        if len(hist) < 3 or not future:
            return False
        old = stability_score(hist[0], self.target_height)
        now = stability_score(hist[-1], self.target_height)
        if old - now < self.min_degradation:
            return False
        fs = np.asarray([stability_score(s, self.target_height) for s in future], dtype=float)
        j = int(np.argmax(fs)); gain = float(fs[j] - now)
        self.candidates += 1
        if not np.isfinite(gain) or gain < self.min_recovery_gain:
            return False
        traj = self.trajectory(hist); target = future[j].vector().astype(float)
        density = self._local_density(traj); factor = self._resolution_factor(density)
        gate = self.base_gate * factor
        best: Optional[tuple[float, int]] = None
        for i, r in enumerate(self._records):
            rms, _, ok = self._distance(r.trajectory, traj, gate)
            if ok and rms <= self.merge_rms * factor:
                td = float(np.sqrt(np.mean(((r.target_state - target) / self.sensor_scale) ** 2)))
                obj = rms + 0.30 * td
                if best is None or obj < best[0]:
                    best = (obj, i)
        self.admitted += 1
        if best is None:
            self._records.append(FullTrajectoryPrototype(traj, target, gain))
        else:
            r = self._records[best[1]]; n = r.confirmations + 1
            r.trajectory += (traj - r.trajectory) / n
            r.target_state += (target - r.target_state) / n
            r.confirmations = n; r.gain_sum += gain; r.recovery_gain = max(r.recovery_gain, gain)
            self.merged += 1
        if len(self._records) > self.max_records:
            self._records.sort(key=lambda r: (r.confirmations, r.mean_gain))
            del self._records[: len(self._records) - self.max_records]
        return True

    def recall(self, history: Iterable[BalanceState], *, k: int = 5, min_confidence: float = 0.45) -> Optional[FullTrajectoryRecall]:
        traj = self.trajectory(history); current = traj[-1]
        density = self._local_density(traj); factor = self._resolution_factor(density)
        gate = self.base_gate * factor
        ranked = []
        for r in self._records:
            if r.confirmations < self.min_confirmations or r.trajectory.shape != traj.shape:
                continue
            rms, mx, ok = self._distance(r.trajectory, traj, gate)
            if ok and rms <= self.interpolation_rms * factor:
                ranked.append((rms, mx, r))
        if not ranked:
            return None
        ranked.sort(key=lambda x: x[0])
        rms, mx, nearest = ranked[0]
        if rms <= self.direct_rms * factor:
            conf = float(np.exp(-rms) * (0.75 + 0.25 * min(1.0, np.log1p(nearest.confirmations) / np.log(21.0))))
            if conf >= min_confidence:
                target = nearest.target_state.copy()
                return FullTrajectoryRecall(target, target-current, conf, 1.0, 1, True, rms, mx, density, factor)
        chosen = ranked[: max(2, k)]
        corrections = np.asarray([r.target_state-current for _,_,r in chosen])
        normed = corrections / self.sensor_scale
        norms = np.linalg.norm(normed, axis=1); valid = norms > 1e-9
        if np.count_nonzero(valid) < 2:
            return None
        unit = normed[valid] / norms[valid, None]
        coherence = float(np.linalg.norm(np.mean(unit, axis=0)))
        if coherence < self.min_coherence:
            return None
        weights = np.asarray([(1+r.mean_gain)*np.log1p(r.confirmations)*np.exp(-d/0.35) for d,_,r in chosen], dtype=float)
        weights /= np.sum(weights)
        target = np.sum(np.asarray([r.target_state for _,_,r in chosen]) * weights[:,None], axis=0)
        avg = float(np.average([d for d,_,_ in chosen], weights=weights)); mx = float(max(x[1] for x in chosen))
        conf = float(np.exp(-avg) * coherence)
        if conf < min_confidence:
            return None
        return FullTrajectoryRecall(target, target-current, conf, coherence, len(chosen), False, avg, mx, density, factor)

    def stats(self) -> dict[str, float|int]:
        c = [r.confirmations for r in self._records]
        return {"records":len(c),"confirmed_records":sum(x>=self.min_confirmations for x in c),"mean_confirmations":float(np.mean(c)) if c else 0.0,"max_confirmations":max(c) if c else 0,"candidates":self.candidates,"admitted":self.admitted,"merged":self.merged}
