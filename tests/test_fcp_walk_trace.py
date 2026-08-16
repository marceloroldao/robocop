from pathlib import Path
import pytest

from robocop.fcp_walk_trace import FCPWalkTrace, FCPWalkTraceRecorder, load_walk_traces


def obs(height=0.9, vel_z=0.0, roll=0.0, pitch=0.0, gyro=(0.0, 0.0, 0.0)):
    x = [0.0] * 63
    x[1] = height * 3.0
    x[2] = vel_z / 2.0
    x[3] = roll / 15.0
    x[4] = pitch / 15.0
    x[5:8] = [g / 100.0 for g in gyro]
    return x


def test_trace_preserves_exact_policy_io():
    before = obs(0.9, -0.1, 8.0, -4.0, (20.0, -30.0, 5.0))
    action = [0.1 * i for i in range(16)]
    after = obs(0.91, 0.0, 4.0, -2.0, (10.0, -12.0, 2.0))
    t = FCPWalkTrace(20, before, action, after)
    assert t.obs_before == tuple(before)
    assert t.action == tuple(action)
    assert t.obs_after == tuple(after)


def test_semantic_decode_matches_public_walk_scaling():
    t = FCPWalkTrace(20, obs(0.92, -0.14, 7.5, -3.0, (25.0, -40.0, 5.0)), [0.0]*16, obs())
    assert t.height_before == pytest.approx(0.92)
    assert t.vel_z_before == pytest.approx(-0.14)
    assert t.roll_before_deg == pytest.approx(7.5)
    assert t.pitch_before_deg == pytest.approx(-3.0)
    assert t.gyro_before_deg_s == pytest.approx((25.0, -40.0, 5.0))


def test_energy_proxy_is_mean_square_policy_action():
    t = FCPWalkTrace(20, obs(), [1.0, -1.0] + [0.0]*14, obs())
    assert t.energy_proxy == pytest.approx(2.0/16.0)


def test_jsonl_roundtrip(tmp_path: Path):
    path = tmp_path / "walk.jsonl"
    r = FCPWalkTraceRecorder(path)
    original = r.record(20, obs(), [0.0]*16, obs(0.91))
    assert load_walk_traces(path) == [original]


def test_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        FCPWalkTrace(20, [0.0]*62, [0.0]*16, [0.0]*63)
    with pytest.raises(ValueError):
        FCPWalkTrace(20, [0.0]*63, [0.0]*15, [0.0]*63)
