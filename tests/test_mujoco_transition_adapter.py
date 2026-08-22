from types import SimpleNamespace

import numpy as np

from memory.mujoco_transition_adapter import (
    PassiveTransitionObserver,
    extract_balance_state,
    quaternion_roll_pitch,
)
from memory.transition_memory import ResolutiveTransitionMemory


class FakeEnv:
    def __init__(self, qpos, qvel):
        self.unwrapped = self
        self.data = SimpleNamespace(
            qpos=np.asarray(qpos, dtype=float),
            qvel=np.asarray(qvel, dtype=float),
        )


def test_quaternion_roll_pitch_identity():
    roll, pitch = quaternion_roll_pitch([1.0, 0.0, 0.0, 0.0])
    assert abs(roll) < 1e-12
    assert abs(pitch) < 1e-12


def test_extract_balance_state_reads_floating_base():
    env = FakeEnv(
        qpos=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.2],
        qvel=[0.0, 0.0, -0.12, 0.3, 0.4, 0.0, 0.1],
    )
    state = extract_balance_state(env, support_margin=0.07)
    assert state.height == 1.0
    assert state.vertical_speed == -0.12
    assert abs(state.angular_speed - 0.5) < 1e-12
    assert state.support_margin == 0.07


def test_passive_observer_records_recovery_without_modifying_action():
    memory = ResolutiveTransitionMemory(min_gain=0.001)
    env = FakeEnv(
        qpos=[0.0, 0.0, 0.93, 0.99219767, 0.12467473, 0.0, 0.0],
        qvel=[0.0, 0.0, -0.20, 0.70, 0.0, 0.0],
    )
    observer = PassiveTransitionObserver(memory, recall_confidence=0.0)
    before = observer.reset(env)

    action = np.asarray([0.4, -0.2, 0.1])
    action_copy = action.copy()

    env.data.qpos = np.asarray([0.0, 0.0, 0.99, 0.99968752, 0.02499740, 0.0, 0.0])
    env.data.qvel = np.asarray([0.0, 0.0, -0.02, 0.15, 0.0, 0.0])

    result = observer.after_step(action, env)

    assert result.before == before
    assert result.admitted
    assert memory.size == 1
    np.testing.assert_array_equal(action, action_copy)


def test_passive_observer_rejects_terminal_transition():
    memory = ResolutiveTransitionMemory(min_gain=0.001)
    env = FakeEnv(
        qpos=[0.0, 0.0, 0.94, 1.0, 0.0, 0.0, 0.0],
        qvel=[0.0, 0.0, -0.2, 0.5, 0.0, 0.0],
    )
    observer = PassiveTransitionObserver(memory, recall_confidence=0.0)
    observer.reset(env)

    env.data.qpos[2] = 0.99
    env.data.qvel[2] = -0.01
    env.data.qvel[3] = 0.1

    result = observer.after_step([0.2, -0.1], env, terminal=True)
    assert not result.admitted
    assert memory.size == 0


def test_passive_observer_can_recall_previous_recovery():
    memory = ResolutiveTransitionMemory(min_gain=0.001)
    observer = PassiveTransitionObserver(memory, recall_confidence=0.4)

    env = FakeEnv(
        qpos=[0.0, 0.0, 0.94, 0.99500417, 0.09983342, 0.0, 0.0],
        qvel=[0.0, 0.0, -0.15, 0.55, 0.0, 0.0],
    )
    observer.reset(env)
    learned_action = np.asarray([-0.35, 0.20])
    env.data.qpos = np.asarray([0.0, 0.0, 0.99, 0.99980001, 0.01999867, 0.0, 0.0])
    env.data.qvel = np.asarray([0.0, 0.0, -0.02, 0.12, 0.0, 0.0])
    assert observer.after_step(learned_action, env).admitted

    env.data.qpos = np.asarray([0.0, 0.0, 0.941, 0.9951035, 0.0988384, 0.0, 0.0])
    env.data.qvel = np.asarray([0.0, 0.0, -0.14, 0.54, 0.0, 0.0])
    observer.reset(env)
    env.data.qpos = np.asarray([0.0, 0.0, 0.96, 0.99718882, 0.07492971, 0.0, 0.0])
    env.data.qvel = np.asarray([0.0, 0.0, -0.08, 0.35, 0.0, 0.0])

    result = observer.after_step([0.0, 0.0], env)
    assert result.recall is not None
    np.testing.assert_allclose(result.recall.action, learned_action)
