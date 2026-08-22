from types import SimpleNamespace

import numpy as np

from memory.transition_memory import ResolutiveTransitionMemory
from robocop.integrations.bahiart_passive import BahiaRTPassiveBridge


class FakeWorld:
    def __init__(self):
        self.global_position = np.asarray([0.0, 0.0, 1.0], dtype=float)
        self.server_time = 0.0

    def is_fallen(self):
        return bool(self.global_position[2] < 0.3)


class FakeRobot:
    ROBOT_MOTORS = ("hip", "knee", "ankle")

    def __init__(self):
        self.global_orientation_euler = np.asarray([0.0, 0.0, 0.0], dtype=float)
        self.gyroscope = np.asarray([0.0, 0.0, 0.0], dtype=float)
        self.motor_targets = {
            name: {"target_position": 0.0, "kp": 10.0, "kd": 0.1}
            for name in self.ROBOT_MOTORS
        }


class FakeAgent:
    def __init__(self):
        self.world = FakeWorld()
        self.robot = FakeRobot()


def test_bridge_reads_public_bahiart_runtime_state_in_si_units():
    memory = ResolutiveTransitionMemory(min_gain=0.001)
    bridge = BahiaRTPassiveBridge(memory, recall_confidence=0.0)
    agent = FakeAgent()
    agent.world.global_position[2] = 0.92
    agent.robot.global_orientation_euler[:] = [10.0, -5.0, 30.0]
    agent.robot.gyroscope[:] = [30.0, 40.0, 0.0]

    bridge.before_decision(agent)
    state = bridge._current_state
    assert state is not None
    assert abs(state.roll - np.deg2rad(10.0)) < 1e-12
    assert abs(state.pitch - np.deg2rad(-5.0)) < 1e-12
    assert abs(state.angular_speed - np.deg2rad(50.0)) < 1e-12


def test_bridge_estimates_vertical_speed_from_consecutive_server_times():
    memory = ResolutiveTransitionMemory(min_gain=0.001)
    bridge = BahiaRTPassiveBridge(memory, recall_confidence=0.0)
    agent = FakeAgent()

    agent.world.global_position[2] = 0.90
    agent.world.server_time = 1.0
    bridge.before_decision(agent)
    bridge.after_decision(agent)

    agent.world.global_position[2] = 0.94
    agent.world.server_time = 1.2
    bridge.before_decision(agent)
    assert bridge._current_state is not None
    assert abs(bridge._current_state.vertical_speed - 0.20) < 1e-12


def test_bridge_records_baseline_motor_targets_without_modifying_them():
    memory = ResolutiveTransitionMemory(min_gain=0.001, target_height=1.0)
    bridge = BahiaRTPassiveBridge(memory, recall_confidence=0.0)
    agent = FakeAgent()

    agent.world.global_position[2] = 0.90
    agent.world.server_time = 1.0
    agent.robot.global_orientation_euler[:] = [12.0, 0.0, 0.0]
    agent.robot.gyroscope[:] = [40.0, 0.0, 0.0]
    bridge.before_decision(agent)

    expected = np.asarray([0.2, -0.4, 0.1])
    for name, value in zip(agent.robot.ROBOT_MOTORS, expected):
        agent.robot.motor_targets[name]["target_position"] = float(value)
    captured = bridge.after_decision(agent)
    np.testing.assert_allclose(captured, expected)

    agent.world.global_position[2] = 0.99
    agent.world.server_time = 1.2
    agent.robot.global_orientation_euler[:] = [2.0, 0.0, 0.0]
    agent.robot.gyroscope[:] = [5.0, 0.0, 0.0]
    admitted = bridge.before_decision(agent)

    assert admitted is True
    assert memory.size == 1
    actual = np.asarray(
        [agent.robot.motor_targets[n]["target_position"] for n in agent.robot.ROBOT_MOTORS]
    )
    np.testing.assert_allclose(actual, expected)


def test_bridge_never_admits_transition_that_ends_fallen():
    memory = ResolutiveTransitionMemory(min_gain=0.001, target_height=1.0)
    bridge = BahiaRTPassiveBridge(memory, recall_confidence=0.0)
    agent = FakeAgent()

    agent.world.global_position[2] = 0.80
    agent.world.server_time = 1.0
    bridge.before_decision(agent)
    bridge.after_decision(agent)

    agent.world.global_position[2] = 0.20
    agent.world.server_time = 1.2
    admitted = bridge.before_decision(agent)
    assert admitted is False
    assert memory.size == 0


def test_bridge_stats_count_completed_transitions():
    memory = ResolutiveTransitionMemory(min_gain=99.0)
    bridge = BahiaRTPassiveBridge(memory, recall_confidence=1.0)
    agent = FakeAgent()

    bridge.before_decision(agent)
    bridge.after_decision(agent)
    agent.world.server_time = 0.1
    bridge.before_decision(agent)

    stats = bridge.stats()
    assert stats.cycles == 2
    assert stats.completed_transitions == 1
    assert stats.admitted_transitions == 0
