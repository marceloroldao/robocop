from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable


def _finite_tuple(values: Iterable[float], name: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must contain only finite values")
    return out


@dataclass(frozen=True)
class FCPWalkTrace:
    """License-neutral trace of FC Portugal Walk policy I/O.

    The public Walk environment exposes a 63-value observation vector and the
    policy emits a 16-value action vector. RoboCOP stores those values without
    changing them, plus the next observation for transition learning.
    """

    timestamp_ms: int
    obs_before: tuple[float, ...]
    action: tuple[float, ...]
    obs_after: tuple[float, ...]
    terminal: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ms", int(self.timestamp_ms))
        object.__setattr__(self, "obs_before", _finite_tuple(self.obs_before, "obs_before"))
        object.__setattr__(self, "action", _finite_tuple(self.action, "action"))
        object.__setattr__(self, "obs_after", _finite_tuple(self.obs_after, "obs_after"))
        object.__setattr__(self, "terminal", bool(self.terminal))
        if len(self.obs_before) != 63 or len(self.obs_after) != 63:
            raise ValueError("FC Portugal Walk observations must have length 63")
        if len(self.action) != 16:
            raise ValueError("FC Portugal Walk actions must have length 16")

    @property
    def height_before(self) -> float:
        return self.obs_before[1] / 3.0

    @property
    def height_after(self) -> float:
        return self.obs_after[1] / 3.0

    @property
    def vel_z_before(self) -> float:
        return self.obs_before[2] * 2.0

    @property
    def vel_z_after(self) -> float:
        return self.obs_after[2] * 2.0

    @property
    def roll_before_deg(self) -> float:
        return self.obs_before[3] * 15.0

    @property
    def roll_after_deg(self) -> float:
        return self.obs_after[3] * 15.0

    @property
    def pitch_before_deg(self) -> float:
        return self.obs_before[4] * 15.0

    @property
    def pitch_after_deg(self) -> float:
        return self.obs_after[4] * 15.0

    @property
    def gyro_before_deg_s(self) -> tuple[float, float, float]:
        return tuple(v * 100.0 for v in self.obs_before[5:8])

    @property
    def gyro_after_deg_s(self) -> tuple[float, float, float]:
        return tuple(v * 100.0 for v in self.obs_after[5:8])

    @property
    def energy_proxy(self) -> float:
        return sum(v * v for v in self.action) / len(self.action)

    def to_json(self) -> str:
        return json.dumps({
            "timestamp_ms": self.timestamp_ms,
            "obs_before": list(self.obs_before),
            "action": list(self.action),
            "obs_after": list(self.obs_after),
            "terminal": self.terminal,
        }, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> "FCPWalkTrace":
        return cls(**json.loads(payload))


class FCPWalkTraceRecorder:
    def __init__(self, output_path: str | Path | None = None) -> None:
        self.output_path = Path(output_path) if output_path is not None else None
        self.traces: list[FCPWalkTrace] = []

    def record(self, timestamp_ms: int, obs_before, action, obs_after, terminal: bool = False) -> FCPWalkTrace:
        trace = FCPWalkTrace(timestamp_ms, obs_before, action, obs_after, terminal)
        self.traces.append(trace)
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(trace.to_json() + "\n")
        return trace


def load_walk_traces(path: str | Path) -> list[FCPWalkTrace]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(FCPWalkTrace.from_json(line))
    return rows
