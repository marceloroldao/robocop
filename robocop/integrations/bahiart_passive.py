from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from memory.transition_memory import BalanceState, Recall, ResolutiveTransitionMemory


@dataclass(frozen=True)
class BahiaRTProbeStats:
    cycles: int
    completed_transitions: int
    admitted_transitions: int
    recalls: int


class BahiaRTPassiveBridge:
    """Non-invasive bridge from BahiaRT public state to resolutive memory.

    The bridge deliberately uses duck typing and only public runtime attributes
    exposed by the external agent (world position, orientation, gyro and motor
    targets). It does not import or copy BahiaRT source into RoboCOP.

    Call ``before_decision`` immediately after BahiaRT has received/parsed the
    new world state and ``after_decision`` immediately after its normal decision
    maker has selected motor targets. The bridge never changes those targets.
    """

    def __init__(
        self,
        memory: ResolutiveTransitionMemory,
        *,
        recall_confidence: float = 0.65,
    ) -> None:
        self.memory = memory
        self.recall_confidence = float(recall_confidence)
        self._pending_state: Optional[BalanceState] = None
        self._pending_action: Optional[np.ndarray] = None
        self._previous_state: Optional[BalanceState] = None
        self._last_time: Optional[float] = None
        self._last_height: Optional[float] = None
        self._current_state: Optional[BalanceState] = None
        self._current_recall: Optional[Recall] = None
        self._cycles = 0
        self._completed = 0
        self._admitted = 0
        self._recalls = 0

    @staticmethod
    def _safe_time(agent) -> Optional[float]:
        value = getattr(agent.world, "server_time", None)
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    def _state(self, agent) -> BalanceState:
        world = agent.world
        robot = agent.robot

        position = np.asarray(world.global_position, dtype=np.float64)
        euler_deg = np.asarray(robot.global_orientation_euler, dtype=np.float64)
        gyro_deg = np.asarray(robot.gyroscope, dtype=np.float64)
        if position.size < 3 or euler_deg.size < 2 or gyro_deg.size < 3:
            raise ValueError("BahiaRT runtime state does not expose required balance sensors")

        height = float(position[2])
        now = self._safe_time(agent)
        vertical_speed = 0.0
        if (
            now is not None
            and self._last_time is not None
            and self._last_height is not None
            and now > self._last_time
        ):
            vertical_speed = (height - self._last_height) / (now - self._last_time)

        self._last_time = now
        self._last_height = height

        return BalanceState(
            height=height,
            roll=float(np.deg2rad(euler_deg[0])),
            pitch=float(np.deg2rad(euler_deg[1])),
            angular_speed=float(np.linalg.norm(np.deg2rad(gyro_deg[:3]))),
            vertical_speed=float(vertical_speed),
            support_margin=0.0,
        )

    @staticmethod
    def action_vector(agent) -> np.ndarray:
        """Read, but never modify, the currently selected motor targets."""

        robot = agent.robot
        names = tuple(robot.ROBOT_MOTORS)
        values = [float(robot.motor_targets[name]["target_position"]) for name in names]
        return np.asarray(values, dtype=np.float64)

    def before_decision(self, agent) -> Optional[bool]:
        """Complete the previous state/action transition using the new state."""

        current = self._state(agent)
        self._current_state = current
        self._cycles += 1

        admitted: Optional[bool] = None
        if self._pending_state is not None and self._pending_action is not None:
            fallen = bool(agent.world.is_fallen()) if hasattr(agent.world, "is_fallen") else False
            admitted = self.memory.observe(
                self._pending_state,
                self._pending_action,
                current,
                terminal=fallen,
            )
            self._completed += 1
            if admitted:
                self._admitted += 1

        self._current_recall = self.memory.recall(
            current,
            recent_state=self._previous_state,
            min_confidence=self.recall_confidence,
        )
        if self._current_recall is not None:
            self._recalls += 1
        return admitted

    def after_decision(self, agent) -> np.ndarray:
        """Snapshot BahiaRT's selected action for evaluation on the next cycle."""

        if self._current_state is None:
            raise RuntimeError("before_decision(agent) must be called first")
        selected = self.action_vector(agent)
        self._previous_state = self._pending_state
        self._pending_state = self._current_state
        self._pending_action = selected.copy()
        return selected.copy()

    @property
    def current_recall(self) -> Optional[Recall]:
        return self._current_recall

    def stats(self) -> BahiaRTProbeStats:
        return BahiaRTProbeStats(
            cycles=self._cycles,
            completed_transitions=self._completed,
            admitted_transitions=self._admitted,
            recalls=self._recalls,
        )
