from __future__ import annotations

import argparse

import numpy as np

from benchmarks.humanoid_balance import (
    BaseAgent,
    ConfidenceV3Agent,
    FullFieldAgent,
    MemoryV5Agent,
    TrajectoryV6Agent,
    run_episode,
    print_summary,
    train_memory,
)
from robocop.controllers import FieldModulatedController, PDController
from robocop.field import ResolutiveField
from robocop.intensity_memory import IntensityTrajectoryMemory
from robocop.memory import DescriptiveMemory
from robocop.mujoco_env import extract_humanoid_state
from robocop.mujoco_field import finite_difference_gradient
from robocop.trajectory_memory import TrajectoryMemory


class IntensityV62Agent(BaseAgent):
    name = "TRAJETORIAS V6.2"

    def __init__(self, memory: IntensityTrajectoryMemory, learn: bool):
        super().__init__()
        self.field = ResolutiveField()
        self.mod = FieldModulatedController(self.pd, field_gain=0.20)
        self.memory = memory
        self.learn_enabled = learn
        self.pending = None
        self.explore_counter = 0

    def reset(self, env):
        super().reset(env)
        self.pending = None
        self.explore_counter = 0

    def act(self, env):
        obs = extract_humanoid_state(env, self.q_target)
        base = self.pd.action(obs.q, obs.qd, self.q_target)
        self.lookups += 1
        grad, learned_gain, level, proto, _ambiguity = self.memory.lookup(obs.field_state)

        if grad is None:
            grad, sims = finite_difference_gradient(env, self.field, base)
            self.sims += sims
            gain = self.memory.gain_candidates[self.explore_counter % len(self.memory.gain_candidates)] if self.learn_enabled else 0.20
        else:
            self.hits += 1
            self.level_hits[level] += 1
            gain = learned_gain
            if self.learn_enabled and self.explore_counter % 5 == 0:
                gain = proto.next_exploration_gain(self.memory.gain_candidates)

        self.explore_counter += 1
        confidence = float(np.clip(gain / 0.20, 0.0, 1.0))
        action = self.mod.action(obs.q, obs.qd, self.q_target, grad, confidence)
        if self.learn_enabled and not self.memory.frozen:
            self.pending = (
                obs.field_state,
                np.array(grad, copy=True),
                float(gain),
                float(np.mean(action ** 2)),
            )
        return action

    def observe(self, reward: float, terminated: bool, truncated: bool) -> None:
        if not self.learn_enabled or self.memory.frozen or self.pending is None:
            return
        state, grad, gain, energy = self.pending
        survival = 0.0 if (terminated or truncated) else 1.0
        self.memory.learn(state, grad, gain, energy, float(reward), survival)
        self.pending = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--train-seeds", type=int, default=30)
    args = parser.parse_args()

    train_range = range(900, 900 + args.train_seeds)
    memory_v5 = DescriptiveMemory()
    memory_v61 = TrajectoryMemory()
    memory_v62 = IntensityTrajectoryMemory()

    train_memory(lambda: MemoryV5Agent(memory_v5, True), train_range, args.max_steps, "V5")
    memory_v5.freeze()
    print("Memoria V5:", memory_v5.stats())

    train_memory(lambda: TrajectoryV6Agent(memory_v61, True), train_range, args.max_steps, "V6.1")
    memory_v61.freeze()
    print("Memoria V6.1:", memory_v61.stats())

    train_memory(lambda: IntensityV62Agent(memory_v62, True), train_range, args.max_steps, "V6.2")
    memory_v62.freeze()
    print("Memoria V6.2:", memory_v62.stats())

    results = []
    for seed in range(1000, 1000 + args.seeds):
        agents = (
            BaseAgent(),
            FullFieldAgent(),
            ConfidenceV3Agent(),
            MemoryV5Agent(memory_v5, False),
            TrajectoryV6Agent(memory_v61, False),
            IntensityV62Agent(memory_v62, False),
        )
        for agent in agents:
            r = run_episode(agent, seed, args.max_steps)
            results.append(r)
            print(
                f"{r.controller:18s} seed={seed} passos={r.steps} E={r.energy:.6f} "
                f"CPU={r.cpu_ms:.3f} sims={r.sims_per_step:.2f} hit={r.hit_rate:.2f} "
                f"Z1={r.hit_z1:.2f} Z2={r.hit_z2:.2f} Z3={r.hit_z3:.2f}"
            )
    print_summary(results)


if __name__ == "__main__":
    main()
