from pathlib import Path

import pytest

from robocop.fcp_protocol import FCPSensorFrame
from robocop.fcp_transition_recorder import FCPPassThroughRecorder, FCPTransition, load_jsonl


def frame(t, roll=0.0, pitch=0.0, omega=(0.0, 0.0, 0.0)):
    return FCPSensorFrame(
        timestamp_ms=t,
        behavior="Walk",
        torso_rpy=(roll, pitch, 0.0),
        angular_velocity=omega,
        linear_velocity=(0.2, 0.0, 0.0),
        joint_position=(0.1, -0.2, 0.3),
        joint_velocity=(0.4, -0.5, 0.6),
        foot_contact=(1.0, 1.0),
        walk_target=(1.0, 0.0, 0.0),
    )


def test_pass_through_returns_exact_baseline_action():
    recorder = FCPPassThroughRecorder()
    baseline = (0.13, -0.27, 0.91)
    observed = recorder.observe_action(baseline)
    directive = recorder.directive()
    assert observed == baseline
    assert directive.mode == "pass_through"
    assert directive.apply(baseline) == baseline


def test_transition_preserves_signed_sensor_deltas_and_action():
    before = frame(100, roll=0.20, pitch=-0.10, omega=(0.6, -0.2, 0.1))
    after = frame(120, roll=-0.05, pitch=0.04, omega=(-0.3, 0.5, 0.0))
    action = (0.2, -0.4, 0.1)
    t = FCPPassThroughRecorder().record(before, action, after, reward=1.2)
    assert t.baseline_action == action
    assert t.dt_ms == 20
    assert t.delta_rpy == pytest.approx((-0.25, 0.14, 0.0))
    assert t.delta_omega == pytest.approx((-0.9, 0.7, -0.1))


def test_json_roundtrip_and_jsonl_persistence(tmp_path: Path):
    path = tmp_path / "transitions.jsonl"
    recorder = FCPPassThroughRecorder(path)
    original = recorder.record(frame(0), (0.1, -0.2), frame(20, roll=0.1), terminal=False)
    parsed = FCPTransition.from_json(original.to_json())
    assert parsed == original
    rows = load_jsonl(path)
    assert rows == [original]


def test_default_energy_proxy_is_mean_square_action():
    t = FCPPassThroughRecorder().record(frame(0), (1.0, -1.0, 0.0, 0.0), frame(20))
    assert t.energy_proxy == pytest.approx(0.5)


def test_recorder_rejects_time_reversal():
    with pytest.raises(ValueError):
        FCPPassThroughRecorder().record(frame(40), (0.0,), frame(20))


def test_stats_summarize_recorded_baseline():
    r = FCPPassThroughRecorder()
    r.record(frame(0), (1.0, 0.0), frame(20))
    r.record(frame(20), (0.0, 0.0), frame(60), terminal=True)
    s = r.stats()
    assert s["transitions"] == 2
    assert s["mean_dt_ms"] == pytest.approx(30.0)
    assert s["terminal_fraction"] == pytest.approx(0.5)
