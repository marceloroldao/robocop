import numpy as np

from robocop.prefall_model import PrefallRiskModel, make_transition
from robocop.transition_memory import SensorSnapshot, TransitionRecorder


def snap(height, vertical, omega, vel_z):
    return SensorSnapshot(
        height=float(height),
        vertical=float(vertical),
        omega=float(omega),
        vel_z=float(vel_z),
        q=np.zeros(17),
        qd=np.zeros(17),
    )


def build_recorder(stable=True):
    rec = TransitionRecorder()
    for i in range(20):
        if stable:
            before = snap(1.38, 0.98, 0.2, -0.01)
            after = snap(1.38, 0.98, 0.2, -0.01)
            terminated = False
            reward = 5.2
        else:
            before = snap(1.15 - 0.005 * i, 0.80, 1.0, -0.25)
            after = snap(1.12 - 0.005 * i, 0.76, 1.2, -0.35)
            terminated = i == 19
            reward = 4.0
        rec.add(make_transition(before, np.ones(17) * 0.05, after, reward, terminated, i))
    return rec


def test_prefall_model_assigns_higher_risk_to_bad_transition():
    good = build_recorder(True)
    bad = build_recorder(False)
    model = PrefallRiskModel.fit([good, bad], prefall_window=20)

    good_risk = model.risk(good.samples[5])
    bad_risk = model.risk(bad.samples[10])

    assert bad_risk > good_risk
    assert bad_risk > 0.5
    assert good_risk < 0.5


def test_prefall_model_requires_both_classes():
    good = build_recorder(True)
    try:
        PrefallRiskModel.fit([good], prefall_window=12)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
