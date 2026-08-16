import numpy as np

from robocop.controllers import FieldModulatedController, PDController
from robocop.field import FieldState, ResolutiveField
from robocop.memory import DescriptiveMemory


def test_field_prefers_upright_target_height():
    field = ResolutiveField()
    good = field.score(FieldState(1.40, 1.0, 0.0, 0.0))
    bad = field.score(FieldState(1.00, 0.4, 2.0, -1.0))
    assert good > bad


def test_memory_learns_and_freezes():
    mem = DescriptiveMemory()
    state = FieldState(1.40, 1.0, 0.1, 0.0)
    grad = np.ones(17)
    for _ in range(12):
        mem.learn(state, grad, 0.005, 5.3)
    found, level, node = mem.lookup(state)
    assert found is not None
    assert level == 1
    before = node.visits
    mem.freeze()
    mem.learn(state, grad * 0.5, 0.01, 4.0)
    _, _, node_after = mem.lookup(state)
    assert node_after.visits == before


def test_field_modulation_is_bounded():
    pd = PDController(action_limit=0.4)
    ctrl = FieldModulatedController(pd=pd, field_gain=0.2)
    q = np.zeros(17)
    qd = np.zeros(17)
    target = np.ones(17) * 10.0
    grad = np.ones(17)
    action = ctrl.action(q, qd, target, grad, confidence=1.0)
    assert action.shape == (17,)
    assert np.all(action <= 0.4)
    assert np.all(action >= -0.4)
