import numpy as np

from robocop.balance_reflex import BalanceReflexMemory
from robocop.transition_memory import SensorSnapshot, TransitionSample, TransitionRecorder


def snap(height, vertical, omega, vel_z, qv=0.0, qdv=0.0):
    q = np.full(17, qv, dtype=float)
    qd = np.full(17, qdv, dtype=float)
    return SensorSnapshot(height, vertical, omega, vel_z, q, qd)


def build_recorder(direction):
    r = TransitionRecorder()
    s0 = snap(1.40, 0.99, 0.10 * direction, -0.05 * direction, qdv=0.1 * direction)
    s1 = snap(1.39, 0.97, 0.40 * direction, -0.20 * direction, qdv=0.4 * direction)
    s2 = snap(1.41, 0.995, 0.08 * direction, 0.02 * direction, qdv=0.05 * direction)
    action = np.full(17, -0.12 * direction)
    r.add(TransitionSample(s0, np.zeros(17), s1, 4.5, 0.006, False, 0))
    r.add(TransitionSample(s1, action, s2, 5.0, 0.004, False, 1))
    return r


def test_opposite_motion_retrieves_opposite_reflex():
    plus = build_recorder(+1.0)
    minus = build_recorder(-1.0)
    memory = BalanceReflexMemory.fit([plus, minus], prefall_window=0, min_improvement=0.001)
    a_plus, gain_plus, _ = memory.lookup(plus.samples[0].after, plus.samples[1].before)
    a_minus, gain_minus, _ = memory.lookup(minus.samples[0].after, minus.samples[1].before)
    assert a_plus is not None and a_minus is not None
    assert np.mean(a_plus) < 0
    assert np.mean(a_minus) > 0
    assert gain_plus > 0 and gain_minus > 0


def test_reflex_memory_excludes_terminal_samples():
    r = build_recorder(+1.0)
    bad = TransitionSample(r.samples[-1].before, np.full(17, 0.3),
                           snap(0.90, 0.3, 2.0, -1.0), 0.0, 0.09, True, 2)
    r.add(bad)
    memory = BalanceReflexMemory.fit([r], prefall_window=1, min_improvement=0.001)
    assert all(np.mean(p.action) < 0.0 for p in memory.prototypes)


def test_reflex_stats_are_finite():
    memory = BalanceReflexMemory.fit([build_recorder(+1.0)], prefall_window=0, min_improvement=0.001)
    stats = memory.stats()
    assert stats['prototypes'] >= 1
    assert np.isfinite(stats['mean_gain'])
    assert np.isfinite(stats['mean_energy'])
