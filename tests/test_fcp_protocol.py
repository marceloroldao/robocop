import math
import pytest

from robocop.fcp_protocol import FCPSensorFrame, ResolutiveDirective


def make_frame(roll=0.2, omega_x=-0.4):
    return FCPSensorFrame(
        timestamp_ms=20,
        behavior="Walk",
        torso_rpy=(roll, -0.1, 0.05),
        angular_velocity=(omega_x, 0.2, -0.3),
        linear_velocity=(0.4, -0.2, 0.01),
        joint_position=(0.1, -0.2, 0.3),
        joint_velocity=(-0.4, 0.5, -0.6),
        foot_contact=(0.8, 0.2),
        walk_target=(1.0, -0.5, 20.0),
    )


def test_sensor_roundtrip_preserves_signed_direction():
    a = make_frame(roll=0.25, omega_x=-0.7)
    b = FCPSensorFrame.from_json(a.to_json())
    assert b == a
    assert b.torso_rpy[0] > 0
    assert b.angular_velocity[0] < 0


def test_opposite_fall_directions_remain_distinct():
    left = make_frame(roll=0.3, omega_x=0.8)
    right = make_frame(roll=-0.3, omega_x=-0.8)
    assert left.torso_rpy[0] == -right.torso_rpy[0]
    assert left.angular_velocity[0] == -right.angular_velocity[0]
    assert left.to_json() != right.to_json()


def test_reflex_directive_blends_delta_without_replacing_baseline():
    d = ResolutiveDirective(
        mode="reflex_blend",
        confidence=0.9,
        blend=0.25,
        action_delta=(0.4, -0.2),
        reason="known stabilizing transition",
    )
    out = d.apply((0.1, 0.2))
    assert out == pytest.approx((0.2, 0.15))


def test_protocol_rejects_nonfinite_sensor_values():
    with pytest.raises(ValueError):
        make_frame(roll=math.nan)
