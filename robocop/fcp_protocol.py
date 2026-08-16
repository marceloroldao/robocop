from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Iterable, Literal, Sequence


BridgeMode = Literal["pass_through", "reflex_blend", "behavior_guard"]


def _finite_tuple(values: Iterable[float], name: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must contain only finite values")
    return out


@dataclass(frozen=True)
class FCPSensorFrame:
    """License-neutral wire representation of one FC Portugal control frame.

    This module does not import or copy FCPCodebase. A GPL-side adapter may
    serialize FC Portugal state into this structure over IPC/JSON.
    """

    timestamp_ms: int
    behavior: str
    torso_rpy: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    linear_velocity: tuple[float, float, float]
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    foot_contact: tuple[float, ...] = ()
    walk_target: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ms", int(self.timestamp_ms))
        object.__setattr__(self, "behavior", str(self.behavior))
        object.__setattr__(self, "torso_rpy", _finite_tuple(self.torso_rpy, "torso_rpy"))
        object.__setattr__(self, "angular_velocity", _finite_tuple(self.angular_velocity, "angular_velocity"))
        object.__setattr__(self, "linear_velocity", _finite_tuple(self.linear_velocity, "linear_velocity"))
        object.__setattr__(self, "joint_position", _finite_tuple(self.joint_position, "joint_position"))
        object.__setattr__(self, "joint_velocity", _finite_tuple(self.joint_velocity, "joint_velocity"))
        object.__setattr__(self, "foot_contact", _finite_tuple(self.foot_contact, "foot_contact"))
        object.__setattr__(self, "walk_target", _finite_tuple(self.walk_target, "walk_target"))
        if len(self.torso_rpy) != 3 or len(self.angular_velocity) != 3 or len(self.linear_velocity) != 3:
            raise ValueError("torso_rpy, angular_velocity and linear_velocity must have length 3")
        if len(self.walk_target) != 3:
            raise ValueError("walk_target must have length 3")
        if len(self.joint_position) != len(self.joint_velocity):
            raise ValueError("joint_position and joint_velocity must have equal length")

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "FCPSensorFrame":
        data = json.loads(payload)
        return cls(**data)


@dataclass(frozen=True)
class ResolutiveDirective:
    """Decision returned by the RoboCOP side of the bridge."""

    mode: BridgeMode = "pass_through"
    confidence: float = 0.0
    blend: float = 0.0
    action_delta: tuple[float, ...] = ()
    behavior_guard: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.mode not in {"pass_through", "reflex_blend", "behavior_guard"}:
            raise ValueError(f"unsupported bridge mode: {self.mode}")
        confidence = float(self.confidence)
        blend = float(self.blend)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be finite and in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "blend", blend)
        object.__setattr__(self, "action_delta", _finite_tuple(self.action_delta, "action_delta"))
        if self.behavior_guard is not None:
            object.__setattr__(self, "behavior_guard", str(self.behavior_guard))
        object.__setattr__(self, "reason", str(self.reason))

    def apply(self, baseline_action: Sequence[float]) -> tuple[float, ...]:
        base = _finite_tuple(baseline_action, "baseline_action")
        if self.mode != "reflex_blend" or not self.action_delta:
            return base
        if len(base) != len(self.action_delta):
            raise ValueError("baseline_action and action_delta must have equal length")
        return tuple(b + self.blend * d for b, d in zip(base, self.action_delta))

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "ResolutiveDirective":
        return cls(**json.loads(payload))
