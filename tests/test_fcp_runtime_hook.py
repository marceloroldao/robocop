import numpy as np
import pytest

from robocop.fcp_runtime_hook import RuntimeWalkCollector
from robocop.fcp_walk_trace import load_walk_traces


def test_runtime_collector_closes_transition_on_next_observation(tmp_path):
    path = tmp_path / "trace.jsonl"
    c = RuntimeWalkCollector(path)
    obs0 = np.zeros(63)
    obs1 = np.ones(63)
    action = np.arange(16, dtype=float) / 10.0

    returned_obs = c.on_observation(obs0, 100)
    returned_action = c.on_action(action)
    c.on_observation(obs1, 120)

    assert returned_obs is obs0
    assert returned_action is action
    assert c.stats.transitions == 1
    traces = load_walk_traces(path)
    assert len(traces) == 1
    assert traces[0].timestamp_ms == 100
    assert traces[0].obs_before == tuple(obs0)
    assert traces[0].action == tuple(action)
    assert traces[0].obs_after == tuple(obs1)


def test_direction_signs_are_preserved(tmp_path):
    c = RuntimeWalkCollector(tmp_path / "trace.jsonl")
    obs0 = np.zeros(63)
    obs1 = np.zeros(63)
    obs0[3] = +0.4
    obs0[4] = -0.2
    obs0[5:8] = (+0.3, -0.4, +0.5)
    obs1[3] = -0.4
    obs1[4] = +0.2
    obs1[5:8] = (-0.3, +0.4, -0.5)
    c.on_observation(obs0, 0)
    c.on_action(np.zeros(16))
    c.on_observation(obs1, 20)
    t = c.recorder.traces[0]
    assert t.roll_before_deg > 0 and t.roll_after_deg < 0
    assert t.pitch_before_deg < 0 and t.pitch_after_deg > 0
    assert t.gyro_before_deg_s == pytest.approx((30, -40, 50))
    assert t.gyro_after_deg_s == pytest.approx((-30, 40, -50))


def test_invalid_shapes_fail(tmp_path):
    c = RuntimeWalkCollector(tmp_path / "trace.jsonl")
    with pytest.raises(ValueError):
        c.on_observation(np.zeros(62), 0)
    c.on_observation(np.zeros(63), 0)
    with pytest.raises(ValueError):
        c.on_action(np.zeros(15))


def test_action_before_observation_fails(tmp_path):
    c = RuntimeWalkCollector(tmp_path / "trace.jsonl")
    with pytest.raises(RuntimeError):
        c.on_action(np.zeros(16))
