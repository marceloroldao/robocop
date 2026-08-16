import numpy as np

from robocop.field import FieldState
from robocop.intensity_memory import GainOutcome, IntensityPrototype, IntensityTrajectoryMemory


def test_lower_energy_gain_wins_when_reward_and_survival_match():
    p = IntensityPrototype(np.array([1.0, 0.0]))
    for _ in range(3):
        p.record_gain(0.05, 0.004, 5.0, 1.0)
        p.record_gain(0.20, 0.020, 5.0, 1.0)
    assert p.best_gain((0.05, 0.20)) == 0.05


def test_exploration_rotates_to_least_tested_gain():
    p = IntensityPrototype(np.array([1.0, 0.0]))
    p.record_gain(0.05, 0.004, 5.0, 1.0)
    p.record_gain(0.10, 0.005, 5.0, 1.0)
    assert p.next_exploration_gain((0.05, 0.10, 0.15, 0.20)) == 0.15


def test_memory_returns_learned_gain_after_repeated_observations():
    m = IntensityTrajectoryMemory(gain_candidates=(0.05, 0.20))
    s = FieldState(height=1.4, vertical=1.0, omega=0.2, vel_z=0.0)
    g = np.array([1.0, 0.0, 0.0])
    for _ in range(10):
        m.learn(s, g, 0.05, 0.004, 5.0, 1.0)
        m.learn(s, g, 0.20, 0.020, 5.0, 1.0)
    grad, gain, level, proto, ambiguity = m.lookup(s)
    assert grad is not None
    assert gain == 0.05
    assert level in (1, 2, 3)
