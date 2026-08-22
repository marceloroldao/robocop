import numpy as np

from memory.transition_memory import (
    BalanceState,
    ResolutiveTransitionMemory,
    stability_score,
)


def test_stability_score_rewards_recovery():
    unstable = BalanceState(0.92, 0.25, -0.18, 0.8, -0.3, -0.05)
    recovered = BalanceState(0.99, 0.05, -0.03, 0.2, -0.03, 0.08)
    assert stability_score(recovered) > stability_score(unstable)


def test_memory_admits_only_stabilizing_nonterminal_transitions():
    memory = ResolutiveTransitionMemory(min_gain=0.01, min_confirmations=1)
    before = BalanceState(0.94, 0.20, 0.0, 0.7, -0.2, -0.02)
    after_good = BalanceState(0.99, 0.05, 0.0, 0.2, -0.02, 0.06)
    after_bad = BalanceState(0.90, 0.28, 0.0, 1.0, -0.4, -0.08)
    assert memory.observe(before, [0.4, -0.2], after_good)
    assert not memory.observe(before, [0.8, 0.8], after_bad)
    assert not memory.observe(before, [0.4, -0.2], after_good, terminal=True)
    assert memory.size == 1


def test_repeated_experience_consolidates_instead_of_growing():
    memory = ResolutiveTransitionMemory(min_gain=0.001, min_confirmations=3)
    before = BalanceState(0.95, 0.18, -0.02, 0.6, -0.15, 0.0)
    after = BalanceState(0.99, 0.04, -0.01, 0.18, -0.02, 0.07)
    action = np.asarray([0.25, -0.5, 0.15])
    for _ in range(5):
        assert memory.observe(before, action, after)
    stats = memory.stats()
    assert memory.size == 1
    assert stats["observations_admitted"] == 5
    assert stats["observations_merged"] == 4
    assert stats["confirmed_records"] == 1
    assert stats["mean_confirmations"] == 5.0


def test_unconfirmed_experience_returns_miss():
    memory = ResolutiveTransitionMemory(min_gain=0.001, min_confirmations=3)
    before = BalanceState(0.95, 0.18, -0.02, 0.6, -0.15, 0.0)
    after = BalanceState(0.99, 0.04, -0.01, 0.18, -0.02, 0.07)
    memory.observe(before, [0.25, -0.5], after)
    assert memory.recall(before) is None


def test_confirmed_recall_returns_layer_and_action():
    memory = ResolutiveTransitionMemory(min_gain=0.001, min_confirmations=3)
    before = BalanceState(0.95, 0.18, -0.02, 0.6, -0.15, 0.0)
    after = BalanceState(0.99, 0.04, -0.01, 0.18, -0.02, 0.07)
    action = np.asarray([0.25, -0.5, 0.15])
    for _ in range(3):
        memory.observe(before, action, after)
    query = BalanceState(0.951, 0.176, -0.018, 0.61, -0.14, 0.0)
    recall = memory.recall(query, min_confidence=0.5)
    assert recall is not None
    np.testing.assert_allclose(recall.action, action)
    assert recall.layer in {"Z1", "Z2", "Z3"}
    assert recall.confirmations >= 3


def test_far_state_is_a_miss_even_with_confirmed_memory():
    memory = ResolutiveTransitionMemory(min_gain=0.001, min_confirmations=3)
    before = BalanceState(0.95, 0.18, 0.0, 0.6, -0.15, 0.0)
    after = BalanceState(0.99, 0.03, 0.0, 0.15, -0.01, 0.07)
    for _ in range(6):
        memory.observe(before, [-0.3, 0.2], after)
    far = BalanceState(0.55, -1.2, 0.8, 4.0, -2.0, -0.4)
    assert memory.recall(far) is None


def test_z2_direction_disambiguates_same_posture():
    memory = ResolutiveTransitionMemory(min_gain=0.001, min_confirmations=3)
    falling_right = BalanceState(0.96, 0.16, 0.0, 0.55, -0.08, 0.01)
    recovered_right = BalanceState(0.99, 0.03, 0.0, 0.12, -0.01, 0.07)
    action_right = np.asarray([-0.6, 0.2])
    falling_left = BalanceState(0.96, -0.16, 0.0, 0.55, -0.08, 0.01)
    recovered_left = BalanceState(0.99, -0.03, 0.0, 0.12, -0.01, 0.07)
    action_left = np.asarray([0.6, -0.2])
    for _ in range(3):
        memory.observe(falling_right, action_right, recovered_right)
        memory.observe(falling_left, action_left, recovered_left)
    previous = BalanceState(0.965, 0.10, 0.0, 0.45, -0.04, 0.02)
    current = BalanceState(0.958, 0.155, 0.0, 0.57, -0.09, 0.01)
    recall = memory.recall(current, recent_state=previous)
    assert recall is not None
    np.testing.assert_allclose(recall.action, action_right)


def test_stats_report_hierarchy():
    memory = ResolutiveTransitionMemory(min_gain=0.001, min_confirmations=1)
    before = BalanceState(0.94, 0.2, 0.0, 0.7, -0.2, 0.0)
    after = BalanceState(0.99, 0.03, 0.0, 0.1, -0.01, 0.08)
    assert memory.observe(before, [0.1, 0.2], after)
    stats = memory.stats()
    assert stats["records"] == 1
    assert stats["mean_gain"] > 0
    assert stats["z1_regions"] == 1
    assert stats["z2_patterns"] == 1
