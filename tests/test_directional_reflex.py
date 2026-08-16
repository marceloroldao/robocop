import numpy as np

from robocop.directional_reflex import (
    DirectionalSnapshot,
    DirectionalTransition,
    DirectionalReflexMemory,
    quaternion_roll_pitch,
)


def snap(roll, pitch, ox, oy, oz=0.0, h=1.40, vz=0.0, qd_sign=0.0):
    return DirectionalSnapshot(
        height=h, vertical=0.98, roll=roll, pitch=pitch,
        omega_x=ox, omega_y=oy, omega_z=oz, vel_z=vz,
        q=np.zeros(17), qd=np.full(17, qd_sign),
    )


def episode(direction):
    s0 = snap(0.02*direction, 0.03*direction, 0.05*direction, 0.08*direction, qd_sign=0.02*direction)
    s1 = snap(0.10*direction, 0.14*direction, 0.30*direction, 0.42*direction, h=1.37, vz=-0.15, qd_sign=0.20*direction)
    s2 = snap(0.01*direction, 0.02*direction, 0.03*direction, 0.05*direction, h=1.40, vz=0.01, qd_sign=0.01*direction)
    correction = np.full(17, -0.12*direction)
    return [
        DirectionalTransition(s0, np.zeros(17), s1, 4.5, 0.006, False),
        DirectionalTransition(s1, correction, s2, 5.0, 0.004, False),
    ]


def test_quaternion_roll_pitch_identity():
    roll, pitch = quaternion_roll_pitch(np.array([1.0, 0.0, 0.0, 0.0]))
    assert abs(roll) < 1e-12
    assert abs(pitch) < 1e-12


def test_opposite_fall_directions_retrieve_opposite_reflexes():
    plus = episode(+1.0)
    minus = episode(-1.0)
    memory = DirectionalReflexMemory.fit([plus, minus], prefall_window=0, min_improvement=0.001)
    a_plus, g_plus, _ = memory.lookup(plus[0].after, plus[1].before)
    a_minus, g_minus, _ = memory.lookup(minus[0].after, minus[1].before)
    assert np.mean(a_plus) < 0
    assert np.mean(a_minus) > 0
    assert g_plus > 0 and g_minus > 0


def test_directional_memory_preserves_signed_angular_information():
    plus = episode(+1.0)
    minus = episode(-1.0)
    assert plus[1].before.omega_y > 0
    assert minus[1].before.omega_y < 0
    assert plus[1].before.pitch > 0
    assert minus[1].before.pitch < 0
