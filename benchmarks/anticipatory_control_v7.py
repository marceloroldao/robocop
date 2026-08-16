from __future__ import annotations

import argparse
import time

import numpy as np

from benchmarks.humanoid_balance import EpisodeResult, TrajectoryV6Agent, run_episode
from benchmarks.transition_diagnostics import record_episode, train_v61
from robocop.mujoco_env import extract_humanoid_state, make_humanoid_env
from robocop.mujoco_field import finite_difference_gradient
from robocop.prefall_model import PrefallRiskModel, make_transition
from robocop.trajectory_memory import TrajectoryMemory
from robocop.controllers import FieldModulatedController
from robocop.field import ResolutiveField


class AnticipatoryV7Agent(TrajectoryV6Agent):
    name = "TRANSICOES V7"

    def __init__(self, memory: TrajectoryMemory, risk_model: PrefallRiskModel, threshold=0.60, rescue_steps=3):
        super().__init__(memory, False)
        self.risk_model = risk_model
        self.threshold = float(threshold)
        self.rescue_steps = int(rescue_steps)
        self.rescue_left = 0
        self.last_snapshot = None
        self.last_action = None
        self.last_reward = 0.0
        self.last_risk = 0.0
        self.risk_events = 0
        self.rescue_actions = 0
        self.field = ResolutiveField()
        self.mod = FieldModulatedController(self.pd, field_gain=0.20)

    def reset(self, env):
        super().reset(env)
        obs = extract_humanoid_state(env, self.q_target)
        from robocop.transition_memory import SensorSnapshot
        self.last_snapshot = SensorSnapshot.from_parts(obs.field_state, obs.q, obs.qd)
        self.last_action = np.zeros(env.action_space.shape[0], dtype=float)
        self.last_reward = 0.0
        self.last_risk = 0.0
        self.rescue_left = 0
        self.risk_events = 0
        self.rescue_actions = 0

    def observe(self, reward: float, terminated: bool, truncated: bool) -> None:
        self.last_reward = float(reward)

    def act(self, env):
        from robocop.transition_memory import SensorSnapshot

        obs = extract_humanoid_state(env, self.q_target)
        current = SensorSnapshot.from_parts(obs.field_state, obs.q, obs.qd)

        if self.last_snapshot is not None and self.last_action is not None:
            transition = make_transition(
                self.last_snapshot,
                self.last_action,
                current,
                self.last_reward,
                False,
                0,
            )
            self.last_risk = self.risk_model.risk(transition)
            if self.last_risk >= self.threshold and self.rescue_left <= 0:
                self.rescue_left = self.rescue_steps
                self.risk_events += 1

        base = self.pd.action(obs.q, obs.qd, self.q_target)

        if self.rescue_left > 0:
            grad, sims = finite_difference_gradient(env, self.field, base)
            self.sims += sims
            action = self.mod.action(obs.q, obs.qd, self.q_target, grad, 1.0)
            self.rescue_left -= 1
            self.rescue_actions += 1
        else:
            self.lookups += 1
            grad, level, prototype, _ambiguity = self.memory.lookup(obs.field_state)
            if grad is None:
                grad, sims = finite_difference_gradient(env, self.field, base)
                self.sims += sims
                confidence = 1.0
            else:
                self.hits += 1
                self.level_hits[level] += 1
                confidence = float(np.clip(prototype.confidence, 0.0, 1.0))
                energy_factor = float(np.clip(0.010 / max(prototype.mean_energy, 1e-6), 0.35, 1.0))
                confidence *= energy_factor
            action = self.mod.action(obs.q, obs.qd, self.q_target, grad, confidence)

        self.last_snapshot = current
        self.last_action = np.asarray(action, dtype=float).copy()
        return action


def run_v7_episode(agent: AnticipatoryV7Agent, seed: int, max_steps: int):
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
            agent.observe(float(reward), bool(terminated), bool(truncated))
            reward_total += float(reward)
            steps += 1
            if terminated or truncated:
                break
    finally:
        env.close()
    denom = max(steps, 1)
    lookup_denom = max(agent.lookups, 1)
    result = EpisodeResult(
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
    return result, agent.risk_events, agent.rescue_actions


def summarize(label, rows):
    steps = np.asarray([r.steps for r in rows], dtype=float)
    rewards = np.asarray([r.reward for r in rows], dtype=float)
    energy = np.asarray([r.energy for r in rows], dtype=float)
    cpu = np.asarray([r.cpu_ms for r in rows], dtype=float)
    sims = np.asarray([r.sims_per_step for r in rows], dtype=float)
    print(f"\n{label}")
    print(f"Passos: {steps.mean():.3f}")
    print(f"Recompensa: {rewards.mean():.3f}")
    print(f"Energia: {energy.mean():.6f}")
    print(f"CPU: {cpu.mean():.4f} ms/passo")
    print(f"Sims/passo: {sims.mean():.3f}")
    print(f">=100 passos: {100*np.mean(steps >= 100):.2f}%")
    print(f">=250 passos: {100*np.mean(steps >= 250):.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--train-seeds", type=int, default=30)
    parser.add_argument("--risk-seeds", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--rescue-steps", type=int, default=3)
    parser.add_argument("--prefall-window", type=int, default=12)
    args = parser.parse_args()

    memory = TrajectoryMemory()
    train_v61(memory, args.train_seeds, 500)

    print("\nFASE B - aprendendo transicoes pre-queda")
    recorders = []
    for seed in range(1200, 1200 + args.risk_seeds):
        rec = record_episode(memory, seed, 500)
        recorders.append(rec)
        print(f"seed={seed} transicoes={len(rec.samples)}")
    risk_model = PrefallRiskModel.fit(recorders, prefall_window=args.prefall_window)
    centroid_distance = float(np.linalg.norm(risk_model.prefall_centroid - risk_model.stable_centroid))
    print(f"Distancia treino stable/prefall: {centroid_distance:.4f}")

    baseline_rows = []
    v7_rows = []
    total_events = total_rescues = 0
    print("\nFASE C - teste cego V6.1 vs V7")
    for seed in range(1400, 1400 + args.seeds):
        base = run_episode(TrajectoryV6Agent(memory, False), seed, args.max_steps)
        v7, events, rescues = run_v7_episode(
            AnticipatoryV7Agent(memory, risk_model, args.threshold, args.rescue_steps),
            seed,
            args.max_steps,
        )
        baseline_rows.append(base)
        v7_rows.append(v7)
        total_events += events
        total_rescues += rescues
        print(
            f"seed={seed} | V6.1={base.steps:3d} E={base.energy:.5f} sims={base.sims_per_step:.2f} "
            f"| V7={v7.steps:3d} E={v7.energy:.5f} sims={v7.sims_per_step:.2f} "
            f"eventos={events} resgates={rescues}"
        )

    print("\n" + "=" * 84)
    print("RESULTADO V7 - CONTROLE ANTECIPATORIO")
    print("=" * 84)
    summarize("V6.1 BASELINE", baseline_rows)
    summarize("V7 TRANSICOES", v7_rows)
    b_steps = np.mean([r.steps for r in baseline_rows])
    v_steps = np.mean([r.steps for r in v7_rows])
    b_energy = np.mean([r.energy for r in baseline_rows])
    v_energy = np.mean([r.energy for r in v7_rows])
    b_sims = np.mean([r.sims_per_step for r in baseline_rows])
    v_sims = np.mean([r.sims_per_step for r in v7_rows])
    print("\nV7 vs V6.1")
    print(f"Sobrevivencia: {(v_steps / b_steps - 1) * 100:+.2f}%")
    print(f"Energia: {(v_energy / b_energy - 1) * 100:+.2f}%")
    print(f"Sims/passo: {(v_sims / b_sims - 1) * 100:+.2f}%")
    print(f"Eventos de risco: {total_events}")
    print(f"Acoes de resgate: {total_rescues}")


if __name__ == "__main__":
    main()
