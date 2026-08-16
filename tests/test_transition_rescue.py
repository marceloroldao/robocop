import numpy as np

from robocop.transition_memory import SensorSnapshot, TransitionRecorder, TransitionSample
from robocop.transition_rescue import TransitionRescueMemory


def snap(height, vertical, omega, vel_z, q0=0.0):
    return SensorSnapshot(
        height=float(height),
        vertical=float(vertical),
        omega=float(omega),
        vel_z=float(vel_z),
        q=np.full(17, q0, dtype=float),
        qd=np.zeros(17, dtype=float),
    )


def sample(before, after, action, reward=5.0, energy=0.006, terminated=False, step=0):
    return TransitionSample(
        before=before,
        action=np.full(17, action, dtype=float),
        after=after,
        reward=reward,
        energy=energy,
        terminated=terminated,
        step=step,
    )


def test_memory_prefers_transition_that_improves_balance():
    rec = TransitionRecorder()
    before = snap(1.20, 0.82, 0.80, -0.20)
    rec.add(sample(before, snap(1.28, 0.90, 0.45, -0.05), 0.10, energy=0.005))
    rec.add(sample(before, snap(1.16, 0.72, 1.10, -0.35), -0.10, energy=0.005))
    # Add a terminal transition so the episode has a fall; with prefall_window=1
    # only this final transition is excluded.
    rec.add(sample(snap(1.05, 0.60, 1.5, -0.5), snap(0.98, 0.50, 2.0, -0.7), 0.0, terminated=True, step=2))
    mem = TransitionRescueMemory.fit([rec], prefall_window=1, min_improvement=0.0, min_after_stability=0.0)
    action, gain, distance = mem.lookup(before, k=2)
    assert action is not None
    assert np.mean(action) > 0.0
    assert gain > 0.0
    assert distance < 1e-8


def test_prefall_transition_is_not_stored_as_rescue():
    rec = TransitionRecorder()
    stable_before = snap(1.25, 0.88, 0.4, -0.05)
    rec.add(sample(stable_before, snap(1.30, 0.92, 0.3, 0.0), 0.05, step=0))
    risky_before = snap(1.05, 0.60, 1.4, -0.4)
    rec.add(sample(risky_before, snap(0.98, 0.50, 1.9, -0.7), -0.20, terminated=True, step=1))
    mem = TransitionRescueMemory.fit([rec], prefall_window=1, min_improvement=0.0, min_after_stability=0.0)
    assert mem.stats()["prototypes"] == 1
    action, _, _ = mem.lookup(stable_before)
    assert action is not None
    assert np.mean(action) > 0.0


def test_empty_memory_returns_no_action():
    mem = TransitionRescueMemory([], np.zeros(38), np.ones(38))
    action, gain, distance = mem.lookup(snap(1.2, 0.8, 0.5, 0.0))
    assert action is None
    assert gain == 0.0
    assert np.isinf(distance)
