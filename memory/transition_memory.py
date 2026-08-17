from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class BalanceState:
    """Minimal body-state representation used by the resolutive memory.

    The adapter that talks to a simulator may expose hundreds of raw sensors;
    this class keeps only invariant balance features needed by the memory core.
    """

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


@dataclass(frozen=True)
class Transition:
    before: BalanceState
    action: np.ndarray
    after: BalanceState
    gain: float
    z1: tuple[int, ...]
    z2: tuple[int, ...]


@dataclass(frozen=True)
class Recall:
    action: np.ndarray
    confidence: float
    gain: float
    distance: float
    z1_match: bool
    z2_match: bool


def stability_score(state: BalanceState, target_height: float = 1.0) -> float:
    """Dimensionless balance score; larger is more stable.

    It intentionally depends on generic body invariants rather than on any
    BahiaRT implementation detail. Coefficients are conservative defaults and
    can later be calibrated against the chosen MuJoCo robot.
    """

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
    """Hierarchical memory of actions that moved the body toward stability.

    Z1: coarse body region (where the robot is).
    Z2: signed transition pattern (how the body is moving).
    Z3: stored action/reflex prototype (what previously improved stability).

    The memory does not learn from a state alone. A record is admitted only
    when an observed state-action-next_state transition produces a measurable
    stability gain and is not terminal.
    """

    def __init__(
        self,
        *,
        min_gain: float = 0.01,
        z1_quantization: Iterable[float] = (0.05, 0.08, 0.08, 0.15, 0.10, 0.05),
        distance_scale: Iterable[float] = (0.08, 0.15, 0.15, 0.30, 0.20, 0.10),
        max_records: int = 20_000,
    ) -> None:
        self.min_gain = float(min_gain)
        self.z1_quantization = np.asarray(tuple(z1_quantization), dtype=np.float64)
        self.distance_scale = np.asarray(tuple(distance_scale), dtype=np.float64)
        if self.z1_quantization.shape != (6,) or np.any(self.z1_quantization <= 0):
            raise ValueError("z1_quantization must contain six positive values")
        if self.distance_scale.shape != (6,) or np.any(self.distance_scale <= 0):
            raise ValueError("distance_scale must contain six positive values")
        self.max_records = int(max_records)
        if self.max_records <= 0:
            raise ValueError("max_records must be positive")
        self._records: list[Transition] = []

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

    def observe(
        self,
        before: BalanceState,
        action: Iterable[float] | np.ndarray,
        after: BalanceState,
        *,
        terminal: bool = False,
    ) -> bool:
        """Observe a transition. Returns True only when it enters memory."""

        if terminal:
            return False

        action_array = np.asarray(action, dtype=np.float64).reshape(-1).copy()
        if action_array.size == 0 or not np.all(np.isfinite(action_array)):
            raise ValueError("action must be a finite non-empty vector")

        gain = stability_score(after) - stability_score(before)
        if not np.isfinite(gain) or gain < self.min_gain:
            return False

        record = Transition(
            before=before,
            action=action_array,
            after=after,
            gain=float(gain),
            z1=self._z1(before),
            z2=self._z2(before, after),
        )
        self._records.append(record)
        if len(self._records) > self.max_records:
            del self._records[0 : len(self._records) - self.max_records]
        return True

    def recall(
        self,
        state: BalanceState,
        *,
        recent_state: Optional[BalanceState] = None,
        min_confidence: float = 0.0,
    ) -> Optional[Recall]:
        """Recall the closest useful reflex for the current balance condition.

        If ``recent_state`` is supplied, the query also carries a Z2 trend. A
        matching Z2 transition receives a distance bonus, so the same posture
        can lead to different reflexes depending on the direction of motion.
        """

        if not self._records:
            return None

        query = state.vector()
        query_z1 = self._z1(state)
        query_z2 = self._z2(recent_state, state) if recent_state is not None else None

        best: Optional[tuple[float, Transition, bool, bool]] = None
        for record in self._records:
            diff = (record.before.vector() - query) / self.distance_scale
            distance = float(np.sqrt(np.mean(diff * diff)))
            z1_match = record.z1 == query_z1
            z2_match = query_z2 is not None and record.z2 == query_z2

            if z1_match:
                distance *= 0.75
            if z2_match:
                distance *= 0.55

            # Prefer both geometric proximity and historically larger recovery.
            objective = distance / (1.0 + max(record.gain, 0.0))
            if best is None or objective < best[0]:
                best = (objective, record, z1_match, z2_match)

        assert best is not None
        _, record, z1_match, z2_match = best
        raw_distance = float(
            np.sqrt(
                np.mean(
                    ((record.before.vector() - query) / self.distance_scale) ** 2
                )
            )
        )
        confidence = float(np.exp(-raw_distance))
        if z1_match:
            confidence = min(1.0, confidence + 0.10)
        if z2_match:
            confidence = min(1.0, confidence + 0.15)

        if confidence < min_confidence:
            return None

        return Recall(
            action=record.action.copy(),
            confidence=confidence,
            gain=record.gain,
            distance=raw_distance,
            z1_match=z1_match,
            z2_match=z2_match,
        )

    def stats(self) -> dict[str, float | int]:
        if not self._records:
            return {"records": 0, "mean_gain": 0.0, "z1_regions": 0, "z2_patterns": 0}
        gains = np.asarray([r.gain for r in self._records], dtype=np.float64)
        return {
            "records": len(self._records),
            "mean_gain": float(gains.mean()),
            "z1_regions": len({r.z1 for r in self._records}),
            "z2_patterns": len({r.z2 for r in self._records}),
        }
