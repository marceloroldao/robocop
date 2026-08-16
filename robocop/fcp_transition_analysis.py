from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable

from .fcp_transition_recorder import FCPTransition, load_jsonl


@dataclass(frozen=True)
class TransitionAssessment:
    stability_before: float
    stability_after: float
    gain: float
    angular_speed_before: float
    angular_speed_after: float
    stabilizing: bool


def _norm3(v: tuple[float, float, float]) -> float:
    return sqrt(sum(x * x for x in v))


def stability_score(torso_rpy: tuple[float, float, float], angular_velocity: tuple[float, float, float]) -> float:
    """Simple dimensionless balance score in [0, 1]. Higher is better.

    This is intentionally conservative and model-independent: upright torso and
    low angular speed are rewarded. It is a diagnostic score, not a physics claim.
    """
    roll, pitch, _ = torso_rpy
    tilt = sqrt(roll * roll + pitch * pitch)
    omega = _norm3(angular_velocity)
    return 1.0 / (1.0 + 1.5 * tilt + 0.35 * omega)


def assess_transition(t: FCPTransition, min_gain: float = 0.01) -> TransitionAssessment:
    sb = stability_score(t.before.torso_rpy, t.before.angular_velocity)
    sa = stability_score(t.after.torso_rpy, t.after.angular_velocity)
    gain = sa - sb
    wb = _norm3(t.before.angular_velocity)
    wa = _norm3(t.after.angular_velocity)
    stabilizing = (not t.terminal) and gain >= min_gain and wa <= wb * 1.10
    return TransitionAssessment(sb, sa, gain, wb, wa, stabilizing)


def analyze(transitions: Iterable[FCPTransition], min_gain: float = 0.01) -> dict:
    rows = list(transitions)
    assessed = [assess_transition(t, min_gain=min_gain) for t in rows]
    good = [a for a in assessed if a.stabilizing]
    terminal = [t for t in rows if t.terminal]
    return {
        "transitions": len(rows),
        "stabilizing": len(good),
        "stabilizing_fraction": (len(good) / len(rows)) if rows else 0.0,
        "terminal": len(terminal),
        "mean_gain_all": (sum(a.gain for a in assessed) / len(assessed)) if assessed else 0.0,
        "mean_gain_stabilizing": (sum(a.gain for a in good) / len(good)) if good else 0.0,
        "mean_energy_stabilizing": (
            sum(t.energy_proxy for t, a in zip(rows, assessed) if a.stabilizing) / len(good)
        ) if good else 0.0,
    }


def analyze_jsonl(path: str | Path, min_gain: float = 0.01) -> dict:
    return analyze(load_jsonl(path), min_gain=min_gain)
