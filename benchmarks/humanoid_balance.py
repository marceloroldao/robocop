from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from robocop.controllers import PDController, FieldModulatedController
from robocop.field import ResolutiveField
from robocop.memory import DescriptiveMemory
from robocop.trajectory_memory import TrajectoryMemory
from robocop.mujoco_env import make_humanoid_env, extract_humanoid_state
from robocop.mujoco_field import finite_difference_gradient


@dataclass
class EpisodeResult:
    controller: str
    seed: int
    steps: int
    reward: float
    reward_per_step: float
    torque: float
    energy: float
    cpu_ms: float
    sims_per_step: float
    hit_rate: float = 0.0
    hit_z1: float = 0.0
    hit_z2: float = 0.0
    hit_z3: float = 0.0


class BaseAgent:
    name = "PD"

    def __init__(self):
        self.pd = PDController()
        self.q_target = None
        self.sims = 0
        self.hits = 0
        self.lookups = 0
        self.level_hits = {1: 0, 2: 0, 3: 0}

    def reset(self, env):
        obs = extract_humanoid_state(env)
        self.q_target = obs.q.copy()
        self.sims = self.hits = self.lookups = 0
        self.level_hits = {1: 0, 2: 0, 3: 0}

    def act(self, env):
        obs = extract_humanoid_state(env, self.q_target)
        return self.pd.action(obs.q, obs.qd, self.q_target)


class FullFieldAgent(BaseAgent):
    name = "CAMPO COMPLETO"

    def __init__(self):
        super().__init__()
        self.field = ResolutiveField()
        self.mod = FieldModulatedController(self.pd, field_gain=0.20)

    def act(self, env):
        obs = extract_humanoid_state(env, self.q_target)
        base = self.pd.action(obs.q, obs.qd, self.q_target)
        grad, sims = finite_difference_gradient(env, self.field, base)
        self.sims += sims
        return self.mod.action(obs.q, obs.qd, self.q_target, grad, 1.0)


class ConfidenceV3Agent(BaseAgent):
    name = "HIERARQUICO V3"

    def __init__(self, top_k=5):
        super().__init__()
        self.field = ResolutiveField()
        self.mod = FieldModulatedController(self.pd, field_gain=0.20)
        self.top_k = top_k
        self.gradient = None
        self.importance = None
        self.confidence = 1.0
        self.step_index = 0

    def reset(self, env):
        super().reset(env)
        n = env.action_space.shape[0]
        self.gradient = np.zeros(n)
        self.importance = np.ones(n) * 1e-9
        self.confidence = 1.0
        self.step_index = 0

    def act(self, env):
        obs = extract_humanoid_state(env, self.q_target)
        base = self.pd.action(obs.q, obs.qd, self.q_target)
        if self.step_index < 3:
            grad, sims = finite_difference_gradient(env, self.field, base)
            self.gradient = grad
            self.importance = 0.92 * self.importance + 0.08 * np.abs(grad)
            self.sims += sims
            self.confidence = 1.0
        elif self.step_index % 2 == 0:
            idx = np.argsort(self.importance)[::-1][: self.top_k]
            partial, sims = finite_difference_gradient(env, self.field, base, indices=idx)
            self.sims += sims
            old = self.gradient[idx]
            n1, n2 = np.linalg.norm(old), np.linalg.norm(partial[idx])
            cos = 0.0 if n1 < 1e-12 or n2 < 1e-12 else float(np.dot(old, partial[idx]) / (n1 * n2))
            if cos < 0.35 and self.step_index % 4 == 0:
                grad, sims = finite_difference_gradient(env, self.field, base)
                self.gradient = grad
                self.sims += sims
                self.confidence = 1.0
            else:
                self.gradient = 0.75 * self.gradient + 0.25 * partial
                self.confidence = float(np.clip(0.75 + 0.25 * max(cos, 0.0), 0.25, 1.0))
            self.importance = 0.92 * self.importance + 0.08 * np.abs(self.gradient)
        else:
            self.confidence *= 0.98
        self.step_index += 1
        return self.mod.action(obs.q, obs.qd, self.q_target, self.gradient, self.confidence)


class MemoryV5Agent(BaseAgent):
    name = "MEMORIA V5"

    def __init__(self, memory: DescriptiveMemory, learn: bool):
        super().__init__()
        self.field = ResolutiveField()
        self.mod = FieldModulatedController(self.pd, field_gain=0.20)
        self.memory = memory
        self.learn_enabled = learn

    def act(self, env):
        obs = extract_humanoid_state(env, self.q_target)
        base = self.pd.action(obs.q, obs.qd, self.q_target)
        self.lookups += 1
        grad, level, node = self.memory.lookup(obs.field_state)
        if grad is None:
            grad, sims = finite_difference_gradient(env, self.field, base)
            self.sims += sims
            if self.learn_enabled and not self.memory.frozen:
                self.memory.learn(obs.field_state, grad, float(np.mean(base ** 2)), self.field.score(obs.field_state))
            confidence = 1.0
        else:
            self.hits += 1
            self.level_hits[level] += 1
            confidence = float(np.clip(node.confidence, 0.0, 1.0))
        return self.mod.action(obs.q, obs.qd, self.q_target, grad, confidence)


class TrajectoryV6Agent(BaseAgent):
    name = "TRAJETORIAS V6"

    def __init__(self, memory: TrajectoryMemory, learn: bool):
        super().__init__()
        self.field = ResolutiveField()
        self.mod = FieldModulatedController(self.pd, field_gain=0.20)
        self.memory = memory
        self.learn_enabled = learn

    def act(self, env):
        obs = extract_humanoid_state(env, self.q_target)
        base = self.pd.action(obs.q, obs.qd, self.q_target)
        self.lookups += 1
        grad, level, prototype, _ambiguity = self.memory.lookup(obs.field_state)
        if grad is None:
            grad, sims = finite_difference_gradient(env, self.field, base)
            self.sims += sims
            if self.learn_enabled and not self.memory.frozen:
                energy = float(np.mean(base ** 2))
                reward_proxy = self.field.score(obs.field_state)
                survival_proxy = float(np.clip(reward_proxy / 3.5, 0.0, 1.0))
                self.memory.learn(obs.field_state, grad, energy, reward_proxy, survival_proxy)
            confidence = 1.0
        else:
            self.hits += 1
            self.level_hits[level] += 1
            confidence = float(np.clip(prototype.confidence, 0.0, 1.0))
            energy_factor = float(np.clip(0.010 / max(prototype.mean_energy, 1e-6), 0.35, 1.0))
            confidence *= energy_factor
        return self.mod.action(obs.q, obs.qd, self.q_target, grad, confidence)


def run_episode(agent, seed: int, max_steps: int):
    env = make_humanoid_env()
    env.reset(seed=seed)
    agent.reset(env)
    reward_total = torque_total = energy_total = cpu_total = 0.0
    steps = 0
    try:
        for _ in range(max_steps):
            t0 = time.perf_counter()
            action = agent.act(env)
            cpu_total += time.perf_counter() - t0
            torque_total += float(np.mean(np.abs(action)))
            energy_total += float(np.mean(action ** 2))
            _, reward, terminated, truncated, _ = env.step(action)
            reward_total += float(reward)
            steps += 1
            if terminated or truncated:
                break
    finally:
        env.close()
    denom = max(steps, 1)
    lookup_denom = max(agent.lookups, 1)
    return EpisodeResult(
        controller=agent.name,
        seed=seed,
        steps=steps,
        reward=reward_total,
        reward_per_step=reward_total / denom,
        torque=torque_total / denom,
        energy=energy_total / denom,
        cpu_ms=cpu_total / denom * 1000.0,
        sims_per_step=agent.sims / denom,
        hit_rate=agent.hits / lookup_denom,
        hit_z1=agent.level_hits[1] / lookup_denom,
        hit_z2=agent.level_hits[2] / lookup_denom,
        hit_z3=agent.level_hits[3] / lookup_denom,
    )


def print_summary(results):
    names = sorted(set(r.controller for r in results))
    print("\n" + "=" * 78)
    print("RESULTADO FINAL")
    print("=" * 78)
    for name in names:
        rows = [r for r in results if r.controller == name]
        def avg(attr): return float(np.mean([getattr(r, attr) for r in rows]))
        print(f"\n{name}")
        print(f"Passos: {avg('steps'):.3f}")
        print(f"Recompensa: {avg('reward'):.3f}")
        print(f"R/passo: {avg('reward_per_step'):.5f}")
        print(f"Torque: {avg('torque'):.6f}")
        print(f"Energia: {avg('energy'):.6f}")
        print(f"CPU: {avg('cpu_ms'):.4f} ms/passo")
        print(f"Sims/passo: {avg('sims_per_step'):.3f}")
        print(f"Hit total: {avg('hit_rate')*100:.2f}%")
        print(f"Hit Z1/Z2/Z3: {avg('hit_z1')*100:.2f}% / {avg('hit_z2')*100:.2f}% / {avg('hit_z3')*100:.2f}%")


def train_memory(agent_factory, seeds, max_steps, label):
    print(f"FASE A - treino {label}")
    for seed in seeds:
        r = run_episode(agent_factory(), seed, max_steps)
        print(f"seed={seed} passos={r.steps} hit={r.hit_rate:.2f} sims={r.sims_per_step:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--train-seeds", type=int, default=30)
    args = parser.parse_args()

    train_range = range(900, 900 + args.train_seeds)
    memory_v5 = DescriptiveMemory()
    memory_v6 = TrajectoryMemory()

    train_memory(lambda: MemoryV5Agent(memory_v5, True), train_range, args.max_steps, "V5")
    memory_v5.freeze()
    print("Memoria V5:", memory_v5.stats())

    train_memory(lambda: TrajectoryV6Agent(memory_v6, True), train_range, args.max_steps, "V6")
    memory_v6.freeze()
    print("Memoria V6:", memory_v6.stats())

    results = []
    for seed in range(1000, 1000 + args.seeds):
        agents = (
            BaseAgent(),
            FullFieldAgent(),
            ConfidenceV3Agent(),
            MemoryV5Agent(memory_v5, False),
            TrajectoryV6Agent(memory_v6, False),
        )
        for agent in agents:
            r = run_episode(agent, seed, args.max_steps)
            results.append(r)
            print(
                f"{r.controller:16s} seed={seed} passos={r.steps} E={r.energy:.6f} "
                f"CPU={r.cpu_ms:.3f} sims={r.sims_per_step:.2f} hit={r.hit_rate:.2f} "
                f"Z1={r.hit_z1:.2f} Z2={r.hit_z2:.2f} Z3={r.hit_z3:.2f}"
            )
    print_summary(results)


if __name__ == "__main__":
    main()
