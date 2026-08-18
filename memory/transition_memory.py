from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class BalanceState:
    """Minimal invariant body-state representation for balance memory."""

    height: float
    roll: float
    pitch: float
    angular_speed: float
    vertical_speed: float
    support_margin: float = 0.0

    def vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.height,
                self.roll,
                self.pitch,
                self.angular_speed,
                self.vertical_speed,
                self.support_margin,
            ],
            dtype=np.float64,
        )


@dataclass
class TransitionPrototype:
    before: BalanceState
    action: np.ndarray
    after: BalanceState
    gain: float
    z1: tuple[int, ...]
    z2: tuple[int, ...]
    confirmations: int = 1
    gain_sum: float = 0.0

    def __post_init__(self) -> None:
        if self.gain_sum == 0.0:
            self.gain_sum = float(self.gain)

    @property
    def mean_gain(self) -> float:
        return float(self.gain_sum / max(self.confirmations, 1))


@dataclass(frozen=True)
class Recall:
    action: np.ndarray
    confidence: float
    gain: float
    distance: float
    z1_match: bool
    z2_match: bool
    layer: str
    confirmations: int


def stability_score(state: BalanceState, target_height: float = 1.0) -> float:
    height_error = abs(state.height - target_height)
    tilt = float(np.hypot(state.roll, state.pitch))
    return float(
        1.0
        - 0.90 * height_error
        - 0.55 * tilt
        - 0.20 * abs(state.angular_speed)
        - 0.15 * abs(state.vertical_speed)
        + 0.10 * state.support_margin
    )


class ResolutiveTransitionMemory:
    """Hierarchical memory of stabilizing transition prototypes.

    Z1 = familiar body region.
    Z2 = familiar body region plus matching transition direction.
    Z3 = strongly confirmed recovery prototype for a rarer/critical state.
    MISS = no sufficiently trustworthy experience; baseline should retain control.
    """

    def __init__(
        self,
        *,
        min_gain: float = 0.01,
        target_height: float = 1.0,
        z1_quantization: Iterable[float] = (0.05, 0.08, 0.08, 0.15, 0.10, 0.05),
        distance_scale: Iterable[float] = (0.08, 0.15, 0.15, 0.30, 0.20, 0.10),
        max_records: int = 20_000,
        merge_state_distance: float = 0.35,
        merge_action_distance: float = 0.18,
        min_confirmations: int = 3,
        z1_max_distance: float = 0.40,
        z2_max_distance: float = 0.65,
        z3_max_distance: float = 0.95,
    ) -> None:
        self.min_gain = float(min_gain)
        self.target_height = float(target_height)
        if not np.isfinite(self.target_height):
            raise ValueError("target_height must be finite")
        self.z1_quantization = np.asarray(tuple(z1_quantization), dtype=np.float64)
        self.distance_scale = np.asarray(tuple(distance_scale), dtype=np.float64)
        if self.z1_quantization.shape != (6,) or np.any(self.z1_quantization <= 0):
            raise ValueError("z1_quantization must contain six positive values")
        if self.distance_scale.shape != (6,) or np.any(self.distance_scale <= 0):
            raise ValueError("distance_scale must contain six positive values")
        self.max_records = int(max_records)
        if self.max_records <= 0:
            raise ValueError("max_records must be positive")
        self.merge_state_distance = float(merge_state_distance)
        self.merge_action_distance = float(merge_action_distance)
        self.min_confirmations = max(1, int(min_confirmations))
        self.z1_max_distance = float(z1_max_distance)
        self.z2_max_distance = float(z2_max_distance)
        self.z3_max_distance = float(z3_max_distance)
        self._records: list[TransitionPrototype] = []
        self._observations_admitted = 0
        self._observations_merged = 0

    @property
    def size(self) -> int:
        return len(self._records)

    def _z1(self, state: BalanceState) -> tuple[int, ...]:
        q = np.rint(state.vector() / self.z1_quantization).astype(np.int64)
        return tuple(int(x) for x in q)

    @staticmethod
    def _z2(before: BalanceState, after: BalanceState, eps: float = 1e-6) -> tuple[int, ...]:
        delta = after.vector() - before.vector()
        signed = np.where(delta > eps, 1, np.where(delta < -eps, -1, 0))
        return tuple(int(x) for x in signed)

    def _state_distance(self, a: BalanceState, b: BalanceState) -> float:
        diff = (a.vector() - b.vector()) / self.distance_scale
        return float(np.sqrt(np.mean(diff * diff)))

    @staticmethod
    def _action_distance(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return float("inf")
        scale = max(1.0, float(np.sqrt(np.mean(a * a))), float(np.sqrt(np.mean(b * b))))
        return float(np.sqrt(np.mean((a - b) ** 2)) / scale)

    @staticmethod
    def _blend_state(a: BalanceState, b: BalanceState, n: int) -> BalanceState:
        av = a.vector()
        bv = b.vector()
        v = av + (bv - av) / float(n)
        return BalanceState(*[float(x) for x in v])

    def observe(
        self,
        before: BalanceState,
        action: Iterable[float] | np.ndarray,
        after: BalanceState,
        *,
        terminal: bool = False,
    ) -> bool:
        if terminal:
            return False

        action_array = np.asarray(action, dtype=np.float64).reshape(-1).copy()
        if action_array.size == 0 or not np.all(np.isfinite(action_array)):
            raise ValueError("action must be a finite non-empty vector")

        gain = stability_score(after, self.target_height) - stability_score(before, self.target_height)
        if not np.isfinite(gain) or gain < self.min_gain:
            return False

        z1 = self._z1(before)
        z2 = self._z2(before, after)
        self._observations_admitted += 1

        best: Optional[tuple[float, int]] = None
        for i, record in enumerate(self._records):
            if record.z1 != z1 or record.z2 != z2:
                continue
            sd = self._state_distance(record.before, before)
            if sd > self.merge_state_distance:
                continue
            ad = self._action_distance(record.action, action_array)
            if ad > self.merge_action_distance:
                continue
            objective = sd + ad
            if best is None or objective < best[0]:
                best = (objective, i)

        if best is not None:
            record = self._records[best[1]]
            n = record.confirmations + 1
            record.before = self._blend_state(record.before, before, n)
            record.after = self._blend_state(record.after, after, n)
            record.action = record.action + (action_array - record.action) / float(n)
            record.confirmations = n
            record.gain_sum += float(gain)
            record.gain = max(record.gain, float(gain))
            self._observations_merged += 1
            return True

        self._records.append(
            TransitionPrototype(
                before=before,
                action=action_array,
                after=after,
                gain=float(gain),
                z1=z1,
                z2=z2,
            )
        )
        if len(self._records) > self.max_records:
            self._records.sort(key=lambda r: (r.confirmations, r.mean_gain))
            del self._records[: len(self._records) - self.max_records]
        return True

    def recall(
        self,
        state: BalanceState,
        *,
        recent_state: Optional[BalanceState] = None,
        min_confidence: float = 0.0,
    ) -> Optional[Recall]:
        if not self._records:
            return None

        query_z1 = self._z1(state)
        query_z2 = self._z2(recent_state, state) if recent_state is not None else None

        candidates: list[tuple[float, TransitionPrototype, bool, bool]] = []
        for record in self._records:
            distance = self._state_distance(record.before, state)
            z1_match = record.z1 == query_z1
            z2_match = query_z2 is not None and record.z2 == query_z2
            quality = 1.0 + min(record.mean_gain, 1.0) + 0.12 * np.log1p(record.confirmations)
            objective = distance / quality
            if z1_match:
                objective *= 0.80
            if z2_match:
                objective *= 0.62
            candidates.append((float(objective), record, z1_match, z2_match))

        _, record, z1_match, z2_match = min(candidates, key=lambda x: x[0])
        raw_distance = self._state_distance(record.before, state)

        if record.confirmations < self.min_confirmations:
            return None

        if z1_match and raw_distance <= self.z1_max_distance:
            layer = "Z1"
        elif z2_match and raw_distance <= self.z2_max_distance:
            layer = "Z2"
        elif record.confirmations >= max(5, self.min_confirmations) and raw_distance <= self.z3_max_distance:
            layer = "Z3"
        else:
            return None

        confirmation_term = min(1.0, np.log1p(record.confirmations) / np.log(21.0))
        confidence = float(np.exp(-raw_distance) * (0.70 + 0.30 * confirmation_term))
        if z1_match:
            confidence = min(1.0, confidence + 0.05)
        if z2_match:
            confidence = min(1.0, confidence + 0.08)

        if confidence < min_confidence:
            return None

        return Recall(
            action=record.action.copy(),
            confidence=confidence,
            gain=record.mean_gain,
            distance=raw_distance,
            z1_match=z1_match,
            z2_match=z2_match,
            layer=layer,
            confirmations=record.confirmations,
        )

    def stats(self) -> dict[str, float | int]:
        if not self._records:
            return {
                "records": 0,
                "mean_gain": 0.0,
                "z1_regions": 0,
                "z2_patterns": 0,
                "confirmed_records": 0,
                "mean_confirmations": 0.0,
                "observations_admitted": self._observations_admitted,
                "observations_merged": self._observations_merged,
                "target_height": self.target_height,
            }
        gains = np.asarray([r.mean_gain for r in self._records], dtype=np.float64)
        confirmations = np.asarray([r.confirmations for r in self._records], dtype=np.float64)
        return {
            "records": len(self._records),
            "mean_gain": float(gains.mean()),
            "z1_regions": len({r.z1 for r in self._records}),
            "z2_patterns": len({r.z2 for r in self._records}),
            "confirmed_records": int(np.sum(confirmations >= self.min_confirmations)),
            "mean_confirmations": float(confirmations.mean()),
            "observations_admitted": self._observations_admitted,
            "observations_merged": self._observations_merged,
            "target_height": self.target_height,
        }
