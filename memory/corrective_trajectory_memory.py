from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from memory.transition_memory import BalanceState, stability_score


@dataclass
class CorrectivePrototype:
    context: np.ndarray
    action_sequence: np.ndarray
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
class CorrectiveRecall:
    action_sequence: np.ndarray
    confidence: float
    distance: float
    confirmations: int
    mean_gain: float


class CorrectiveTrajectoryMemory:
    """V9 memory of short corrective trajectories rather than isolated actions.

    A context encodes both the current body state and the recent direction of
    motion. A prototype is admitted only when the incoming context is degrading
    and the following action sequence produces a measurable future recovery.
    """

    def __init__(
        self,
        *,
        target_height: float = 1.0,
        min_recovery_gain: float = 0.03,
        min_degradation: float = 0.01,
        merge_context_distance: float = 0.65,
        merge_action_distance: float = 0.25,
        min_confirmations: int = 3,
        recall_max_distance: float = 0.85,
        max_records: int = 5000,
    ) -> None:
        self.target_height = float(target_height)
        self.min_recovery_gain = float(min_recovery_gain)
        self.min_degradation = float(min_degradation)
        self.merge_context_distance = float(merge_context_distance)
        self.merge_action_distance = float(merge_action_distance)
        self.min_confirmations = int(min_confirmations)
        self.recall_max_distance = float(recall_max_distance)
        self.max_records = int(max_records)
        self._records: list[CorrectivePrototype] = []
        self.candidates = 0
        self.admitted = 0
        self.merged = 0

        # Current state + recent state delta. Scales roughly normalize the 12-D
        # context without fitting statistics from the evaluation set.
        self._context_scale = np.asarray(
            [0.10, 0.25, 0.25, 0.60, 0.35, 0.10,
             0.06, 0.12, 0.12, 0.35, 0.20, 0.08],
            dtype=np.float64,
        )

    @property
    def size(self) -> int:
        return len(self._records)

    def context(self, history: Iterable[BalanceState]) -> np.ndarray:
        states = list(history)
        if len(states) < 2:
            raise ValueError("trajectory context requires at least two states")
        current = states[-1].vector()
        # Use a multi-cycle derivative to suppress single-frame sensor noise.
        delta = current - states[0].vector()
        return np.concatenate([current, delta]).astype(np.float64)

    def _context_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        d = (a - b) / self._context_scale
        return float(np.sqrt(np.mean(d * d)))

    @staticmethod
    def _action_distance(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return float("inf")
        scale = max(1.0, float(np.sqrt(np.mean(a * a))), float(np.sqrt(np.mean(b * b))))
        return float(np.sqrt(np.mean((a - b) ** 2)) / scale)

    def observe_window(
        self,
        history: Iterable[BalanceState],
        action_sequence: np.ndarray,
        future_states: Iterable[BalanceState],
    ) -> bool:
        hist = list(history)
        future = list(future_states)
        if len(hist) < 2 or not future:
            return False

        actions = np.asarray(action_sequence, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[0] == 0 or not np.all(np.isfinite(actions)):
            return False

        score_old = stability_score(hist[0], self.target_height)
        score_now = stability_score(hist[-1], self.target_height)
        degradation = score_old - score_now
        if degradation < self.min_degradation:
            return False

        future_scores = np.asarray([stability_score(s, self.target_height) for s in future], dtype=float)
        recovery_gain = float(np.max(future_scores) - score_now)
        self.candidates += 1
        if not np.isfinite(recovery_gain) or recovery_gain < self.min_recovery_gain:
            return False

        ctx = self.context(hist)
        best: Optional[tuple[float, int]] = None
        for i, record in enumerate(self._records):
            cd = self._context_distance(record.context, ctx)
            if cd > self.merge_context_distance:
                continue
            ad = self._action_distance(record.action_sequence, actions)
            if ad > self.merge_action_distance:
                continue
            objective = cd + ad
            if best is None or objective < best[0]:
                best = (objective, i)

        self.admitted += 1
        if best is not None:
            record = self._records[best[1]]
            n = record.confirmations + 1
            record.context = record.context + (ctx - record.context) / float(n)
            record.action_sequence = record.action_sequence + (actions - record.action_sequence) / float(n)
            record.confirmations = n
            record.gain_sum += recovery_gain
            record.recovery_gain = max(record.recovery_gain, recovery_gain)
            self.merged += 1
            return True

        self._records.append(
            CorrectivePrototype(
                context=ctx,
                action_sequence=actions.copy(),
                recovery_gain=recovery_gain,
            )
        )
        if len(self._records) > self.max_records:
            self._records.sort(key=lambda r: (r.confirmations, r.mean_gain))
            del self._records[: len(self._records) - self.max_records]
        return True

    def recall(self, history: Iterable[BalanceState], *, min_confidence: float = 0.0) -> Optional[CorrectiveRecall]:
        if not self._records:
            return None
        ctx = self.context(history)
        ranked = []
        for record in self._records:
            if record.confirmations < self.min_confirmations:
                continue
            distance = self._context_distance(record.context, ctx)
            quality = 1.0 + min(1.0, record.mean_gain) + 0.10 * np.log1p(record.confirmations)
            ranked.append((distance / quality, distance, record))
        if not ranked:
            return None
        _, distance, record = min(ranked, key=lambda x: x[0])
        if distance > self.recall_max_distance:
            return None
        confirmation_term = min(1.0, np.log1p(record.confirmations) / np.log(21.0))
        confidence = float(np.exp(-distance) * (0.72 + 0.28 * confirmation_term))
        if confidence < min_confidence:
            return None
        return CorrectiveRecall(
            action_sequence=record.action_sequence.copy(),
            confidence=confidence,
            distance=distance,
            confirmations=record.confirmations,
            mean_gain=record.mean_gain,
        )

    def stats(self) -> dict[str, float | int]:
        conf = [r.confirmations for r in self._records]
        return {
            "records": len(self._records),
            "confirmed_records": sum(c >= self.min_confirmations for c in conf),
            "mean_confirmations": float(np.mean(conf)) if conf else 0.0,
            "max_confirmations": max(conf) if conf else 0,
            "candidates": self.candidates,
            "admitted": self.admitted,
            "merged": self.merged,
        }
