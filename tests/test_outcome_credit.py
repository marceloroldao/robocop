import numpy as np

from robocop.field import FieldState
from robocop.outcome_credit import OutcomeCredit
from robocop.trajectory_memory import TrajectoryPrototype


def test_credit_uses_executed_action_energy():
    credit = OutcomeCredit()
    state = FieldState(height=1.4, vertical=1.0, omega=0.0, vel_z=0.0)
    gradient = np.array([1.0, 0.0])
    action = np.array([0.2, -0.4])
    credit.arm(state, gradient, action)
    outcome = credit.resolve(reward=5.0, terminated=False)
    assert outcome is not None
    assert np.isclose(outcome.energy, np.mean(action ** 2))
    assert outcome.reward == 5.0
    assert outcome.survival == 1.0


def test_terminal_outcome_gets_zero_survival():
    credit = OutcomeCredit()
    state = FieldState(height=1.1, vertical=0.7, omega=1.0, vel_z=-0.2)
    credit.arm(state, np.array([0.0, 1.0]), np.array([0.1, 0.1]))
    outcome = credit.resolve(reward=2.0, terminated=True)
    assert outcome is not None
    assert outcome.survival == 0.0
    assert credit.resolve(reward=1.0, terminated=False) is None


def test_quality_prefers_lower_realized_energy_for_equal_outcomes():
    low = TrajectoryPrototype(
        gradient=np.array([1.0, 0.0]),
        visits=8,
        confidence=0.9,
        mean_energy=0.004,
        mean_reward=5.0,
        mean_survival=1.0,
    )
    high = TrajectoryPrototype(
        gradient=np.array([0.0, 1.0]),
        visits=8,
        confidence=0.9,
        mean_energy=0.020,
        mean_reward=5.0,
        mean_survival=1.0,
    )
    assert low.quality() > high.quality()
