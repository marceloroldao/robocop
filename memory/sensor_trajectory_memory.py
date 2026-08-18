from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from memory.transition_memory import BalanceState, stability_score


@dataclass
class SensorTrajectoryPrototype:
    address: np.ndarray  # [S, dS, ddS]
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
class SensorTrajectoryRecall:
    target_state: np.ndarray
    correction_vector: np.ndarray
    confidence: float
    coherence: float
    neighbors: int
    direct: bool
    rms_distance: float
    max_sensor_error: float


class SensorTrajectoryMemory:
    """V10.2 resolutive address = sensor state + first/second differences.

    Similarity is conjunctive: low aggregate distance is insufficient when any
    normalized sensor (or derivative channel) violates its hard gate.
    """

    def __init__(
        self,
        *,
        target_height: float = 1.0,
        min_recovery_gain: float = 0.03,
        min_degradation: float = 0.01,
        min_confirmations: int = 3,
        merge_rms: float = 0.55,
        direct_rms: float = 0.70,
        interpolation_rms: float = 1.05,
        hard_sensor_gate: float = 2.25,
        hard_derivative_gate: float = 2.75,
        min_coherence: float = 0.75,
        max_records: int = 5000,
    ) -> None:
        self.target_height = float(target_height)
        self.min_recovery_gain = float(min_recovery_gain)
        self.min_degradation = float(min_degradation)
        self.min_confirmations = int(min_confirmations)
        self.merge_rms = float(merge_rms)
        self.direct_rms = float(direct_rms)
        self.interpolation_rms = float(interpolation_rms)
        self.hard_sensor_gate = float(hard_sensor_gate)
        self.hard_derivative_gate = float(hard_derivative_gate)
        self.min_coherence = float(min_coherence)
        self.max_records = int(max_records)
        self._records: list[SensorTrajectoryPrototype] = []
        self.candidates = self.admitted = self.merged = 0
        # Fixed engineering scales; evaluation data never tunes them.
        self.sensor_scale = np.asarray([0.10, 0.25, 0.25, 0.60, 0.35, 0.10], dtype=float)
        self.d1_scale = np.asarray([0.05, 0.10, 0.10, 0.30, 0.18, 0.07], dtype=float)
        self.d2_scale = np.asarray([0.04, 0.08, 0.08, 0.25, 0.15, 0.06], dtype=float)
        self.address_scale = np.concatenate([self.sensor_scale, self.d1_scale, self.d2_scale])

    def address(self, history: Iterable[BalanceState]) -> np.ndarray:
        h = list(history)
        if len(h) < 3:
            raise ValueError("V10.2 address requires at least three sensor states")
        s0, s1, s2 = h[-3].vector(), h[-2].vector(), h[-1].vector()
        d1_prev, d1 = s1 - s0, s2 - s1
        d2 = d1 - d1_prev
        return np.concatenate([s2, d1, d2]).astype(float)

    def _distance(self, a: np.ndarray, b: np.ndarray) -> tuple[float, float, bool]:
        z = np.abs((a - b) / self.address_scale)
        rms = float(np.sqrt(np.mean(z * z)))
        max_sensor = float(np.max(z[:6]))
        sensor_ok = bool(np.all(z[:6] <= self.hard_sensor_gate))
        derivative_ok = bool(np.all(z[6:] <= self.hard_derivative_gate))
        return rms, max_sensor, sensor_ok and derivative_ok

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
        addr = self.address(hist); target = future[j].vector().astype(float)
        best: Optional[tuple[float, int]] = None
        for i, r in enumerate(self._records):
            rms, _, gate = self._distance(r.address, addr)
            if gate and rms <= self.merge_rms:
                td = float(np.sqrt(np.mean(((r.target_state-target)/self.sensor_scale)**2)))
                obj = rms + 0.35 * td
                if best is None or obj < best[0]: best = (obj, i)
        self.admitted += 1
        if best is None:
            self._records.append(SensorTrajectoryPrototype(addr, target, gain))
        else:
            r = self._records[best[1]]; n = r.confirmations + 1
            r.address += (addr-r.address)/n; r.target_state += (target-r.target_state)/n
            r.confirmations = n; r.gain_sum += gain; r.recovery_gain = max(r.recovery_gain, gain)
            self.merged += 1
        if len(self._records) > self.max_records:
            self._records.sort(key=lambda r:(r.confirmations,r.mean_gain)); del self._records[:len(self._records)-self.max_records]
        return True

    def recall(self, history: Iterable[BalanceState], *, k: int = 5, min_confidence: float = 0.45) -> Optional[SensorTrajectoryRecall]:
        addr = self.address(history); current = addr[:6]
        ranked = []
        for r in self._records:
            if r.confirmations < self.min_confirmations: continue
            rms, max_sensor, gate = self._distance(r.address, addr)
            if gate and rms <= self.interpolation_rms:
                ranked.append((rms, max_sensor, r))
        if not ranked: return None
        ranked.sort(key=lambda x:x[0])
        rms, mx, nearest = ranked[0]
        if rms <= self.direct_rms:
            conf = float(np.exp(-rms)*(0.75+0.25*min(1.0,np.log1p(nearest.confirmations)/np.log(21.0))))
            if conf >= min_confidence:
                target=nearest.target_state.copy()
                return SensorTrajectoryRecall(target,target-current,conf,1.0,1,True,rms,mx)
        chosen = ranked[:max(2,k)]
        corrections=np.asarray([r.target_state-current for _,_,r in chosen])
        normed=corrections/self.sensor_scale
        norms=np.linalg.norm(normed,axis=1); valid=norms>1e-9
        if np.count_nonzero(valid)<2: return None
        unit=normed[valid]/norms[valid,None]; coherence=float(np.linalg.norm(np.mean(unit,axis=0)))
        if coherence < self.min_coherence: return None
        weights=np.asarray([(1+r.mean_gain)*np.log1p(r.confirmations)*np.exp(-d/0.40) for d,_,r in chosen],dtype=float)
        weights/=np.sum(weights)
        target=np.sum(np.asarray([r.target_state for _,_,r in chosen])*weights[:,None],axis=0)
        avg=float(np.average([d for d,_,_ in chosen],weights=weights)); mx=float(max(x[1] for x in chosen))
        conf=float(np.exp(-avg)*coherence)
        if conf < min_confidence: return None
        return SensorTrajectoryRecall(target,target-current,conf,coherence,len(chosen),False,avg,mx)

    def stats(self) -> dict[str,float|int]:
        c=[r.confirmations for r in self._records]
        return {"records":len(c),"confirmed_records":sum(x>=self.min_confirmations for x in c),"mean_confirmations":float(np.mean(c)) if c else 0.0,"max_confirmations":max(c) if c else 0,"candidates":self.candidates,"admitted":self.admitted,"merged":self.merged}
