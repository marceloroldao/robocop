import numpy as np

from robocop.mujoco_env import quaternion_vertical


def test_identity_quaternion_is_vertical():
    assert quaternion_vertical(np.array([1.0, 0.0, 0.0, 0.0])) == 1.0


def test_quaternion_is_normalized_internally():
    assert quaternion_vertical(np.array([2.0, 0.0, 0.0, 0.0])) == 1.0


def test_ninety_degree_tilt_reduces_verticality():
    q = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
    assert abs(quaternion_vertical(q)) < 1e-12
