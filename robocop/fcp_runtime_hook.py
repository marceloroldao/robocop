from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .fcp_walk_trace import FCPWalkTraceRecorder


@dataclass
class RuntimeHookStats:
    observations: int = 0
    actions: int = 0
    transitions: int = 0


class RuntimeWalkCollector:
    """Collect obs(63) -> action(16) -> next obs(63) without changing policy I/O."""

    def __init__(self, output_path: str | Path) -> None:
        self.recorder = FCPWalkTraceRecorder(output_path)
        self.pending_obs: np.ndarray | None = None
        self.pending_action: np.ndarray | None = None
        self.pending_timestamp_ms: int | None = None
        self.stats = RuntimeHookStats()

    def on_observation(self, obs: Any, timestamp_ms: int) -> Any:
        arr = np.asarray(obs, dtype=float).reshape(-1)
        if arr.size != 63:
            raise ValueError(f"FC Portugal Walk observation must have 63 values, got {arr.size}")

        if self.pending_obs is not None and self.pending_action is not None:
            self.recorder.record(
                int(self.pending_timestamp_ms),
                self.pending_obs,
                self.pending_action,
                arr,
                terminal=False,
            )
            self.stats.transitions += 1

        self.pending_obs = arr.copy()
        self.pending_action = None
        self.pending_timestamp_ms = int(timestamp_ms)
        self.stats.observations += 1
        return obs

    def on_action(self, action: Any) -> Any:
        arr = np.asarray(action, dtype=float).reshape(-1)
        if arr.size != 16:
            raise ValueError(f"FC Portugal Walk action must have 16 values, got {arr.size}")
        if self.pending_obs is None:
            raise RuntimeError("action observed before Walk observation")
        self.pending_action = arr.copy()
        self.stats.actions += 1
        return action


def install_hook(output_path: str | Path, *, verbose: bool = True) -> RuntimeWalkCollector:
    """Patch external FC Portugal modules in memory only; no external source is edited."""
    import behaviors.custom.Walk.Env as env_module
    import behaviors.custom.Walk.Walk as walk_module

    collector = RuntimeWalkCollector(output_path)
    original_observe = env_module.Env.observe
    original_run_mlp: Callable[..., Any] = walk_module.run_mlp

    def observe_wrapped(self, *args, **kwargs):
        obs = original_observe(self, *args, **kwargs)
        timestamp_ms = int(getattr(self.world, "time_local_ms", 0))
        return collector.on_observation(obs, timestamp_ms)

    def run_mlp_wrapped(obs, model, *args, **kwargs):
        action = original_run_mlp(obs, model, *args, **kwargs)
        return collector.on_action(action)

    env_module.Env.observe = observe_wrapped
    walk_module.run_mlp = run_mlp_wrapped

    if verbose:
        print(f"[RoboCOP] FC Portugal Walk tracing enabled -> {output_path}", file=sys.stderr)
    return collector


def install_from_env() -> RuntimeWalkCollector | None:
    output = os.environ.get("ROBOCOP_FCP_TRACE")
    if not output:
        return None
    return install_hook(output, verbose=os.environ.get("ROBOCOP_FCP_TRACE_QUIET") != "1")
