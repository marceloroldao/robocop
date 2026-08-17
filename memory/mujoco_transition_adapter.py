from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .transition_memory import BalanceState, Recall, ResolutiveTransitionMemory


def quaternion_roll_pitch(quaternion: Iterable[float]) -> tuple[float, float]:
    """Return roll and pitch (radians) from a MuJoCo w,x,y,z quaternion."""

    q = np.asarray(tuple(quaternion), dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("quaternion must have four components in w,x,y,z order")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return 0.0, 0.0
    w, x, y, z = q / norm

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = float(np.arcsin(sinp))
    return roll, pitch


def extract_balance_state(
    env,
    *,
    support_margin: float = 0.0,
) -> BalanceState:
    """Extract simulator-independent balance invariants from a MuJoCo env.

    This adapter intentionally depends only on MuJoCo's conventional floating
    base layout: qpos[0:3] position, qpos[3:7] quaternion and qvel[0:3]
    translational / qvel[3:6] angular velocity. It does not depend on BahiaRT
    source code or robot-specific policy internals.
    """

    data = env.unwrapped.data
    qpos = np.asarray(data.qpos, dtype=np.float64)
    qvel = np.asarray(data.qvel, dtype=np.float64)
    if qpos.size < 7 or qvel.size < 6:
        raise ValueError("MuJoCo state must expose at least 7 qpos and 6 qvel values")

    roll, pitch = quaternion_roll_pitch(qpos[3:7])
    angular_speed = float(np.linalg.norm(qvel[3:6]))
    return BalanceState(
        height=float(qpos[2]),
        roll=roll,
        pitch=pitch,
        angular_speed=angular_speed,
        vertical_speed=float(qvel[2]),
        support_margin=float(support_margin),
    )


@dataclass(frozen=True)
class PassiveStepResult:
    admitted: bool
    before: BalanceState
    after: BalanceState
    recall: Optional[Recall]


class PassiveTransitionObserver:
    """Observe a reference controller without changing any of its actions.

    Usage::

        observer.reset(env)
        action = baseline_policy(...)
        obs, reward, terminated, truncated, info = env.step(action)
        result = observer.after_step(action, env, terminal=terminated or truncated)

    The baseline remains fully in control. The observer only extracts the
    transition, stores stabilizing examples, and reports what the memory would
    have recalled at the pre-action state. This makes the first A/B phase
    behaviorally non-invasive.
    """

    def __init__(
        self,
        memory: ResolutiveTransitionMemory,
        *,
        recall_confidence: float = 0.60,
    ) -> None:
        self.memory = memory
        self.recall_confidence = float(recall_confidence)
        if not 0.0 <= self.recall_confidence <= 1.0:
            raise ValueError("recall_confidence must be between 0 and 1")
        self._previous: Optional[BalanceState] = None
        self._trend_origin: Optional[BalanceState] = None

    def reset(self, env, *, support_margin: float = 0.0) -> BalanceState:
        state = extract_balance_state(env, support_margin=support_margin)
        self._previous = state
        self._trend_origin = None
        return state

    def after_step(
        self,
        action: Iterable[float] | np.ndarray,
        env,
        *,
        terminal: bool = False,
        support_margin: float = 0.0,
    ) -> PassiveStepResult:
        if self._previous is None:
            raise RuntimeError("reset(env) must be called before after_step")

        before = self._previous
        after = extract_balance_state(env, support_margin=support_margin)
        recalled = self.memory.recall(
            before,
            recent_state=self._trend_origin,
            min_confidence=self.recall_confidence,
        )
        admitted = self.memory.observe(before, action, after, terminal=terminal)

        self._trend_origin = before
        self._previous = after
        return PassiveStepResult(
            admitted=admitted,
            before=before,
            after=after,
            recall=recalled,
        )

    @property
    def current_state(self) -> Optional[BalanceState]:
        return self._previous
