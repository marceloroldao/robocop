import numpy as np

from robocop.field import FieldState
from robocop.trajectory_memory import TrajectoryMemory


def state(omega=0.2, vel_z=0.0):
    return FieldState(height=1.40, vertical=0.98, omega=omega, vel_z=vel_z)


def test_separates_opposed_trajectories_in_same_z1_region():
    mem = TrajectoryMemory(merge_cosine=0.82)
    s_a = state(omega=0.2)
    s_b = state(omega=1.1)
    g_a = np.array([1.0, 0.0, 0.0])
    g_b = np.array([-1.0, 0.0, 0.0])
    for _ in range(8):
        mem.learn(s_a, g_a, energy=0.005, reward=1.0, survival=1.0)
        mem.learn(s_b, g_b, energy=0.010, reward=0.8, survival=0.8)

    z1_key = mem.keys(s_a)[0]
    assert len(mem.z1[z1_key].prototypes) == 2
    assert mem.z1[z1_key].ambiguity() > mem.z1_max_ambiguity

    grad_a, level_a, _, _ = mem.lookup(s_a)
    grad_b, level_b, _, _ = mem.lookup(s_b)
    assert level_a >= 2
    assert level_b >= 2
    assert np.dot(grad_a, g_a) > 0
    assert np.dot(grad_b, g_b) > 0


def test_low_energy_prototype_wins_when_branches_share_refined_region():
    mem = TrajectoryMemory(merge_cosine=0.70, z1_max_ambiguity=1.5, z2_max_ambiguity=1.5)
    s = state(omega=0.2)
    efficient = np.array([1.0, 0.0, 0.0])
    costly = np.array([0.0, 1.0, 0.0])

    for _ in range(10):
        mem.learn(s, efficient, energy=0.004, reward=1.0, survival=1.0)
        mem.learn(s, costly, energy=0.020, reward=1.0, survival=1.0)

    grad, level, prototype, _ = mem.lookup(s)
    assert level == 1
    assert prototype.mean_energy < 0.01
    assert np.dot(grad, efficient) > np.dot(grad, costly)


def test_freeze_preserves_memory():
    mem = TrajectoryMemory()
    s = state()
    g = np.array([1.0, 0.0])
    for _ in range(6):
        mem.learn(s, g, energy=0.005, reward=1.0, survival=1.0)
    before = mem.stats().copy()
    mem.freeze()
    for _ in range(10):
        mem.learn(s, -g, energy=0.5, reward=0.0, survival=0.0)
    assert mem.stats() == before
