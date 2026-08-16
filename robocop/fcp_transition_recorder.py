from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from .fcp_protocol import FCPSensorFrame, ResolutiveDirective


def _finite_tuple(values: Iterable[float], name: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must contain only finite values")
    return out


@dataclass(frozen=True)
class FCPTransition:
    """One observed FC Portugal transition with the original action preserved.

    The recorder is intentionally observational. It stores sensor_before,
    baseline_action, sensor_after and optional outcome metadata, but never
    changes the baseline action.
    """

    before: FCPSensorFrame
    baseline_action: tuple[float, ...]
    after: FCPSensorFrame
    reward: float = 0.0
    energy_proxy: float = 0.0
    terminal: bool = False
    source: str = "fcportugal"

    def __post_init__(self) -> None:
        action = _finite_tuple(self.baseline_action, "baseline_action")
        reward = float(self.reward)
        energy = float(self.energy_proxy)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        if not math.isfinite(energy) or energy < 0.0:
            raise ValueError("energy_proxy must be finite and >= 0")
        if self.after.timestamp_ms < self.before.timestamp_ms:
            raise ValueError("transition timestamps must be monotonic")
        object.__setattr__(self, "baseline_action", action)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "energy_proxy", energy)
        object.__setattr__(self, "terminal", bool(self.terminal))
        object.__setattr__(self, "source", str(self.source))

    @property
    def dt_ms(self) -> int:
        return self.after.timestamp_ms - self.before.timestamp_ms

    @property
    def delta_rpy(self) -> tuple[float, float, float]:
        return tuple(b - a for a, b in zip(self.before.torso_rpy, self.after.torso_rpy))

    @property
    def delta_omega(self) -> tuple[float, float, float]:
        return tuple(b - a for a, b in zip(self.before.angular_velocity, self.after.angular_velocity))

    def to_dict(self) -> dict:
        return {
            "before": asdict(self.before),
            "baseline_action": list(self.baseline_action),
            "after": asdict(self.after),
            "reward": self.reward,
            "energy_proxy": self.energy_proxy,
            "terminal": self.terminal,
            "source": self.source,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "FCPTransition":
        data = json.loads(payload)
        data["before"] = FCPSensorFrame(**data["before"])
        data["after"] = FCPSensorFrame(**data["after"])
        return cls(**data)


class FCPPassThroughRecorder:
    """Observes FC Portugal actions and records state transitions.

    Contract: `directive()` always returns pass-through and `observe_action()`
    returns the exact baseline values. This gives us a clean competitive
    baseline before any resolutive intervention is enabled.
    """

    def __init__(self, output_path: str | Path | None = None) -> None:
        self.output_path = Path(output_path) if output_path is not None else None
        self.transitions: list[FCPTransition] = []
        self.frames_seen = 0
        self.actions_seen = 0

    def directive(self) -> ResolutiveDirective:
        return ResolutiveDirective(
            mode="pass_through",
            confidence=1.0,
            blend=0.0,
            reason="observational baseline: FC Portugal action unchanged",
        )

    def observe_action(self, baseline_action: Sequence[float]) -> tuple[float, ...]:
        action = _finite_tuple(baseline_action, "baseline_action")
        self.actions_seen += 1
        return action

    def record(
        self,
        before: FCPSensorFrame,
        baseline_action: Sequence[float],
        after: FCPSensorFrame,
        *,
        reward: float = 0.0,
        energy_proxy: float | None = None,
        terminal: bool = False,
    ) -> FCPTransition:
        action = self.observe_action(baseline_action)
        if energy_proxy is None:
            energy_proxy = sum(v * v for v in action) / max(len(action), 1)
        transition = FCPTransition(
            before=before,
            baseline_action=action,
            after=after,
            reward=reward,
            energy_proxy=energy_proxy,
            terminal=terminal,
        )
        self.transitions.append(transition)
        self.frames_seen += 2
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(transition.to_json() + "\n")
        return transition

    def stats(self) -> dict[str, float | int]:
        n = len(self.transitions)
        if not n:
            return {
                "transitions": 0,
                "actions_seen": self.actions_seen,
                "mean_dt_ms": 0.0,
                "mean_energy_proxy": 0.0,
                "terminal_fraction": 0.0,
            }
        return {
            "transitions": n,
            "actions_seen": self.actions_seen,
            "mean_dt_ms": sum(t.dt_ms for t in self.transitions) / n,
            "mean_energy_proxy": sum(t.energy_proxy for t in self.transitions) / n,
            "terminal_fraction": sum(1 for t in self.transitions if t.terminal) / n,
        }


def load_jsonl(path: str | Path) -> list[FCPTransition]:
    rows: list[FCPTransition] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(FCPTransition.from_json(line))
    return rows
