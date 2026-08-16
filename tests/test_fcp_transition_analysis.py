import pytest

from robocop.fcp_protocol import FCPSensorFrame
from robocop.fcp_transition_analysis import assess_transition, analyze, stability_score
from robocop.fcp_transition_recorder import FCPTransition


def frame(t, roll, pitch, omega):
    return FCPSensorFrame(
        timestamp_ms=t,
        behavior="Walk",
        torso_rpy=(roll, pitch, 0.0),
        angular_velocity=omega,
        linear_velocity=(0.0, 0.0, 0.0),
        joint_position=(0.0, 0.0),
        joint_velocity=(0.0, 0.0),
        foot_contact=(1.0, 1.0),
        walk_target=(0.5, 0.0, 0.0),
    )


def transition(before, after, terminal=False, energy=0.01):
    return FCPTransition(before, (0.1, -0.1), after, energy_proxy=energy, terminal=terminal)


def test_upright_low_omega_scores_better():
    good = stability_score((0.02, -0.02, 0.0), (0.05, 0.0, 0.0))
    bad = stability_score((0.4, 0.3, 0.0), (1.5, 0.8, 0.0))
    assert good > bad


def test_known_recovery_is_stabilizing():
    t = transition(
        frame(0, 0.35, 0.20, (1.0, 0.2, 0.0)),
        frame(20, 0.12, 0.06, (0.35, 0.1, 0.0)),
    )
    a = assess_transition(t)
    assert a.gain > 0.01
    assert a.stabilizing


def test_terminal_transition_is_not_saved_as_stabilizing():
    t = transition(
        frame(0, 0.35, 0.20, (1.0, 0.2, 0.0)),
        frame(20, 0.01, 0.01, (0.01, 0.0, 0.0)),
        terminal=True,
    )
    assert not assess_transition(t).stabilizing


def test_analysis_reports_fraction_and_energy():
    good = transition(frame(0, 0.4, 0.2, (1.0, 0.0, 0.0)), frame(20, 0.1, 0.05, (0.2, 0.0, 0.0)), energy=0.004)
    bad = transition(frame(20, 0.1, 0.05, (0.2, 0.0, 0.0)), frame(40, 0.3, 0.2, (0.9, 0.0, 0.0)), energy=0.020)
    s = analyze([good, bad])
    assert s["transitions"] == 2
    assert s["stabilizing"] == 1
    assert s["stabilizing_fraction"] == pytest.approx(0.5)
    assert s["mean_energy_stabilizing"] == pytest.approx(0.004)
