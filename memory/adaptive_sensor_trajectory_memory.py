from __future__ import annotations

from typing import Iterable, Optional
import numpy as np

from memory.sensor_trajectory_memory import SensorTrajectoryMemory, SensorTrajectoryRecall, SensorTrajectoryPrototype
from memory.transition_memory import BalanceState, stability_score


class AdaptiveSensorTrajectoryMemory(SensorTrajectoryMemory):
    """V10.3: sensorwise address whose resolution tightens with local memory density.

    Broad engineering gates define the maximum neighborhood. Local density then
    contracts all S/dS/ddS gates and RMS radii. RMS only ranks candidates that
    already passed the per-channel adaptive hard gate.
    """

    def __init__(self, *, density_radius: float = 1.35, density_alpha: float = 0.34,
                 min_resolution_factor: float = 0.38, confirmation_alpha: float = 0.08,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.density_radius = float(density_radius)
        self.density_alpha = float(density_alpha)
        self.min_resolution_factor = float(min_resolution_factor)
        self.confirmation_alpha = float(confirmation_alpha)
        self.last_local_density = 0
        self.last_resolution_factor = 1.0

    def _raw_normalized(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.abs((a - b) / self.address_scale)

    def _broad_compatible(self, z: np.ndarray) -> bool:
        return bool(np.all(z[:6] <= self.hard_sensor_gate) and np.all(z[6:] <= self.hard_derivative_gate))

    def local_density(self, address: np.ndarray) -> int:
        n = 0
        for r in self._records:
            z = self._raw_normalized(r.address, address)
            if not self._broad_compatible(z):
                continue
            rms = float(np.sqrt(np.mean(z*z)))
            if rms <= self.density_radius:
                n += max(1, min(r.confirmations, 6))
        return n

    def resolution_factor(self, density: int, confirmations: int = 1) -> float:
        f = 1.0 / (1.0 + self.density_alpha * np.log1p(max(0, density)))
        f /= (1.0 + self.confirmation_alpha * np.log1p(max(1, confirmations)))
        return float(max(self.min_resolution_factor, min(1.0, f)))

    def _adaptive_distance(self, a: np.ndarray, b: np.ndarray, density: int,
                           confirmations: int = 1) -> tuple[float, float, bool, float]:
        z = self._raw_normalized(a, b)
        rms = float(np.sqrt(np.mean(z*z)))
        max_sensor = float(np.max(z[:6]))
        factor = self.resolution_factor(density, confirmations)
        sensor_gate = self.hard_sensor_gate * factor
        derivative_gate = self.hard_derivative_gate * factor
        gate = bool(np.all(z[:6] <= sensor_gate) and np.all(z[6:] <= derivative_gate))
        return rms, max_sensor, gate, factor

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
        density = self.local_density(addr)
        best: Optional[tuple[float, int]] = None
        for i, r in enumerate(self._records):
            rms, _, gate, factor = self._adaptive_distance(r.address, addr, density, r.confirmations)
            if gate and rms <= self.merge_rms * factor:
                td = float(np.sqrt(np.mean(((r.target_state-target)/self.sensor_scale)**2)))
                obj = rms + 0.35*td
                if best is None or obj < best[0]: best=(obj,i)
        self.admitted += 1
        if best is None:
            self._records.append(SensorTrajectoryPrototype(addr,target,gain))
        else:
            r=self._records[best[1]]; n=r.confirmations+1
            r.address += (addr-r.address)/n; r.target_state += (target-r.target_state)/n
            r.confirmations=n; r.gain_sum += gain; r.recovery_gain=max(r.recovery_gain,gain); self.merged += 1
        if len(self._records)>self.max_records:
            self._records.sort(key=lambda r:(r.confirmations,r.mean_gain)); del self._records[:len(self._records)-self.max_records]
        return True

    def recall(self, history: Iterable[BalanceState], *, k: int = 5,
               min_confidence: float = 0.45) -> Optional[SensorTrajectoryRecall]:
        addr=self.address(history); current=addr[:6]
        density=self.local_density(addr); self.last_local_density=density
        ranked=[]
        factors=[]
        for r in self._records:
            if r.confirmations < self.min_confirmations: continue
            rms,mx,gate,factor=self._adaptive_distance(r.address,addr,density,r.confirmations)
            if gate and rms <= self.interpolation_rms*factor:
                ranked.append((rms,mx,r,factor)); factors.append(factor)
        self.last_resolution_factor=float(np.mean(factors)) if factors else self.resolution_factor(density)
        if not ranked:return None
        ranked.sort(key=lambda x:x[0]); rms,mx,nearest,factor=ranked[0]
        if rms <= self.direct_rms*factor:
            conf=float(np.exp(-rms)*(0.75+0.25*min(1.0,np.log1p(nearest.confirmations)/np.log(21.0))))
            if conf>=min_confidence:
                target=nearest.target_state.copy()
                return SensorTrajectoryRecall(target,target-current,conf,1.0,1,True,rms,mx)
        chosen=ranked[:max(2,k)]
        corrections=np.asarray([r.target_state-current for _,_,r,_ in chosen]); normed=corrections/self.sensor_scale
        norms=np.linalg.norm(normed,axis=1); valid=norms>1e-9
        if np.count_nonzero(valid)<2:return None
        unit=normed[valid]/norms[valid,None]; coherence=float(np.linalg.norm(np.mean(unit,axis=0)))
        if coherence<self.min_coherence:return None
        weights=np.asarray([(1+r.mean_gain)*np.log1p(r.confirmations)*np.exp(-d/0.40) for d,_,r,_ in chosen],dtype=float); weights/=np.sum(weights)
        target=np.sum(np.asarray([r.target_state for _,_,r,_ in chosen])*weights[:,None],axis=0)
        avg=float(np.average([d for d,_,_,_ in chosen],weights=weights)); mx=float(max(x[1] for x in chosen)); conf=float(np.exp(-avg)*coherence)
        if conf<min_confidence:return None
        return SensorTrajectoryRecall(target,target-current,conf,coherence,len(chosen),False,avg,mx)

    def stats(self) -> dict[str,float|int]:
        out=super().stats(); out.update({"density_alpha":self.density_alpha,"min_resolution_factor":self.min_resolution_factor})
        return out
