from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from memory.transition_memory import BalanceState, stability_score


@dataclass
class RecoveryTargetPrototype:
    context: np.ndarray
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
class RecoveryTargetRecall:
    target_state: np.ndarray
    correction_vector: np.ndarray
    confidence: float
    coherence: float
    neighbors: int
    direct: bool


class RecoveryTargetMemory:
    """V10.1: remember/interpolate recovered body-state targets, not motor commands."""

    def __init__(self, *, target_height: float = 1.0, min_recovery_gain: float = 0.03,
                 min_degradation: float = 0.01, merge_context_distance: float = 0.65,
                 merge_target_distance: float = 0.55, min_confirmations: int = 3,
                 recall_max_distance: float = 0.85, interpolation_max_distance: float = 1.25,
                 min_coherence: float = 0.72, max_records: int = 5000) -> None:
        self.target_height = float(target_height)
        self.min_recovery_gain = float(min_recovery_gain)
        self.min_degradation = float(min_degradation)
        self.merge_context_distance = float(merge_context_distance)
        self.merge_target_distance = float(merge_target_distance)
        self.min_confirmations = int(min_confirmations)
        self.recall_max_distance = float(recall_max_distance)
        self.interpolation_max_distance = float(interpolation_max_distance)
        self.min_coherence = float(min_coherence)
        self.max_records = int(max_records)
        self._records: list[RecoveryTargetPrototype] = []
        self.candidates = self.admitted = self.merged = 0
        self._context_scale = np.asarray([0.10,0.25,0.25,0.60,0.35,0.10,0.06,0.12,0.12,0.35,0.20,0.08], dtype=float)
        self._state_scale = np.asarray([0.10,0.25,0.25,0.60,0.35,0.10], dtype=float)

    def context(self, history: Iterable[BalanceState]) -> np.ndarray:
        states = list(history)
        current = states[-1].vector()
        return np.concatenate([current, current - states[0].vector()]).astype(float)

    def _cd(self, a: np.ndarray, b: np.ndarray) -> float:
        d = (a-b)/self._context_scale
        return float(np.sqrt(np.mean(d*d)))

    def _sd(self, a: np.ndarray, b: np.ndarray) -> float:
        d = (a-b)/self._state_scale
        return float(np.sqrt(np.mean(d*d)))

    def observe_window(self, history: Iterable[BalanceState], future_states: Iterable[BalanceState]) -> bool:
        hist, future = list(history), list(future_states)
        if len(hist) < 2 or not future:
            return False
        old = stability_score(hist[0], self.target_height)
        now = stability_score(hist[-1], self.target_height)
        if old - now < self.min_degradation:
            return False
        scores = np.asarray([stability_score(s, self.target_height) for s in future], dtype=float)
        j = int(np.argmax(scores))
        gain = float(scores[j] - now)
        self.candidates += 1
        if gain < self.min_recovery_gain:
            return False
        ctx = self.context(hist)
        target = future[j].vector().astype(float)
        best = None
        for i, r in enumerate(self._records):
            cd, td = self._cd(r.context, ctx), self._sd(r.target_state, target)
            if cd <= self.merge_context_distance and td <= self.merge_target_distance:
                obj = cd + td
                if best is None or obj < best[0]: best = (obj, i)
        self.admitted += 1
        if best is not None:
            r = self._records[best[1]]; n = r.confirmations + 1
            r.context += (ctx-r.context)/n; r.target_state += (target-r.target_state)/n
            r.confirmations = n; r.gain_sum += gain; r.recovery_gain = max(r.recovery_gain, gain)
            self.merged += 1
        else:
            self._records.append(RecoveryTargetPrototype(ctx, target, gain))
        return True

    def recall(self, history: Iterable[BalanceState], *, k: int = 5, min_confidence: float = 0.45) -> Optional[RecoveryTargetRecall]:
        ctx = self.context(history); current = ctx[:6]
        ranked = []
        for r in self._records:
            if r.confirmations < self.min_confirmations: continue
            d = self._cd(r.context, ctx)
            if d <= self.interpolation_max_distance:
                ranked.append((d, r))
        if not ranked: return None
        ranked.sort(key=lambda x: x[0])
        nearest_d, nearest = ranked[0]
        if nearest_d <= self.recall_max_distance:
            conf = float(np.exp(-nearest_d) * (0.75 + 0.25*min(1.0, np.log1p(nearest.confirmations)/np.log(21.0))))
            if conf >= min_confidence:
                target = nearest.target_state.copy()
                return RecoveryTargetRecall(target, target-current, conf, 1.0, 1, True)
        chosen = ranked[:max(2, k)]
        corrections = np.asarray([r.target_state-current for _,r in chosen])
        norms = np.linalg.norm(corrections/self._state_scale, axis=1)
        valid = norms > 1e-9
        if np.count_nonzero(valid) < 2: return None
        unit = corrections[valid]/self._state_scale
        unit /= np.linalg.norm(unit, axis=1, keepdims=True)
        mean_dir = np.mean(unit, axis=0)
        coherence = float(np.linalg.norm(mean_dir))
        if coherence < self.min_coherence: return None
        weights = np.asarray([(1+r.mean_gain)*np.log1p(r.confirmations)*np.exp(-d/0.45) for d,r in chosen], dtype=float)
        weights /= np.sum(weights)
        target = np.sum(np.asarray([r.target_state for _,r in chosen])*weights[:,None], axis=0)
        conf = float(np.exp(-np.average([d for d,_ in chosen], weights=weights))*coherence)
        if conf < min_confidence: return None
        return RecoveryTargetRecall(target, target-current, conf, coherence, len(chosen), False)

    def stats(self) -> dict[str, float|int]:
        c=[r.confirmations for r in self._records]
        return {"records":len(c),"confirmed_records":sum(x>=self.min_confirmations for x in c),"mean_confirmations":float(np.mean(c)) if c else 0.0,"max_confirmations":max(c) if c else 0,"candidates":self.candidates,"admitted":self.admitted,"merged":self.merged}
