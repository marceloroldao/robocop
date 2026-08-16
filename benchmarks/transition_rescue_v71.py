from __future__ import annotations

import argparse
import time

import numpy as np

from benchmarks.humanoid_balance import EpisodeResult, TrajectoryV6Agent, run_episode
from benchmarks.transition_diagnostics import record_episode, train_v61
from benchmarks.anticipatory_control_v7 import AnticipatoryV7Agent, run_v7_episode, summarize
from robocop.mujoco_env import extract_humanoid_state, make_humanoid_env
from robocop.prefall_model import PrefallRiskModel, make_transition
from robocop.trajectory_memory import TrajectoryMemory
from robocop.transition_memory import SensorSnapshot
from robocop.transition_rescue import TransitionRescueMemory


class TransitionRescueV71Agent(TrajectoryV6Agent):
    name = "TRANSICOES V7.1"

    def __init__(
        self,
        memory: TrajectoryMemory,
        risk_model: PrefallRiskModel,
        rescue_memory: TransitionRescueMemory,
        threshold: float = 0.60,
        blend: float = 0.35,
        max_distance: float = 1.25,
        rescue_steps: int = 1,
    ):
        super().__init__(memory, False)
        self.risk_model = risk_model
        self.rescue_memory = rescue_memory
        self.threshold = float(threshold)
        self.blend = float(blend)
        self.max_distance = float(max_distance)
        self.rescue_steps = int(rescue_steps)
        self.last_snapshot = None
        self.last_action = None
        self.last_reward = 0.0
        self.last_risk = 0.0
        self.rescue_left = 0
        self.pending_rescue_action = None
        self.risk_events = 0
        self.rescue_hits = 0
        self.rescue_actions = 0
        self.rescue_misses = 0
        self.mean_rescue_distance = []
        self.mean_expected_gain = []

    def reset(self, env):
        super().reset(env)
        obs = extract_humanoid_state(env, self.q_target)
        self.last_snapshot = SensorSnapshot.from_parts(obs.field_state, obs.q, obs.qd)
        self.last_action = np.zeros(env.action_space.shape[0], dtype=float)
        self.last_reward = 0.0
        self.last_risk = 0.0
        self.rescue_left = 0
        self.pending_rescue_action = None
        self.risk_events = 0
        self.rescue_hits = 0
        self.rescue_actions = 0
        self.rescue_misses = 0
        self.mean_rescue_distance = []
        self.mean_expected_gain = []

    def observe(self, reward: float, terminated: bool, truncated: bool) -> None:
        self.last_reward = float(reward)

    def _normal_action(self, env, obs):
        base = self.pd.action(obs.q, obs.qd, self.q_target)
        self.lookups += 1
        grad, level, prototype, _ambiguity = self.memory.lookup(obs.field_state)
        if grad is None:
            from robocop.mujoco_field import finite_difference_gradient
            grad, sims = finite_difference_gradient(env, self.field, base)
            self.sims += sims
            confidence = 1.0
        else:
            self.hits += 1
            self.level_hits[level] += 1
            confidence = float(np.clip(prototype.confidence, 0.0, 1.0))
            energy_factor = float(np.clip(0.010 / max(prototype.mean_energy, 1e-6), 0.35, 1.0))
            confidence *= energy_factor
        return self.mod.action(obs.q, obs.qd, self.q_target, grad, confidence)

    def act(self, env):
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
                self.risk_events += 1
                rescue_action, expected_gain, distance = self.rescue_memory.lookup(current)
                if (
                    rescue_action is not None
                    and expected_gain > 0.0
                    and distance <= self.max_distance
                ):
                    self.pending_rescue_action = np.asarray(rescue_action, dtype=float)
                    self.rescue_left = self.rescue_steps
                    self.rescue_hits += 1
                    self.mean_rescue_distance.append(float(distance))
                    self.mean_expected_gain.append(float(expected_gain))
                else:
                    self.rescue_misses += 1

        normal = self._normal_action(env, obs)
        if self.rescue_left > 0 and self.pending_rescue_action is not None:
            action = (1.0 - self.blend) * normal + self.blend * self.pending_rescue_action
            action = np.clip(action, env.action_space.low, env.action_space.high)
            self.rescue_left -= 1
            self.rescue_actions += 1
            if self.rescue_left <= 0:
                self.pending_rescue_action = None
        else:
            action = normal

        self.last_snapshot = current
        self.last_action = np.asarray(action, dtype=float).copy()
        return action


def run_v71_episode(agent: TransitionRescueV71Agent, seed: int, max_steps: int):
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
    diagnostics = {
        "risk_events": agent.risk_events,
        "rescue_hits": agent.rescue_hits,
        "rescue_actions": agent.rescue_actions,
        "rescue_misses": agent.rescue_misses,
        "distance": float(np.mean(agent.mean_rescue_distance)) if agent.mean_rescue_distance else float("nan"),
        "expected_gain": float(np.mean(agent.mean_expected_gain)) if agent.mean_expected_gain else 0.0,
    }
    return result, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--train-seeds", type=int, default=30)
    parser.add_argument("--risk-seeds", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--blend", type=float, default=0.35)
    parser.add_argument("--max-distance", type=float, default=1.25)
    parser.add_argument("--rescue-steps", type=int, default=1)
    parser.add_argument("--prefall-window", type=int, default=12)
    args = parser.parse_args()

    memory = TrajectoryMemory()
    train_v61(memory, args.train_seeds, 500)

    print("\nFASE B - aprendendo transicoes de risco e transicoes de resgate")
    recorders = []
    for seed in range(1200, 1200 + args.risk_seeds):
        rec = record_episode(memory, seed, 500)
        recorders.append(rec)
        print(f"seed={seed} transicoes={len(rec.samples)}")

    risk_model = PrefallRiskModel.fit(recorders, prefall_window=args.prefall_window)
    rescue_memory = TransitionRescueMemory.fit(recorders, prefall_window=args.prefall_window)
    print("Memoria de resgate:", rescue_memory.stats())

    baseline_rows = []
    v7_rows = []
    v71_rows = []
    totals = {"risk_events": 0, "rescue_hits": 0, "rescue_actions": 0, "rescue_misses": 0}
    distances = []
    gains = []

    print("\nFASE C - teste cego V6.1 vs V7 vs V7.1")
    for seed in range(1500, 1500 + args.seeds):
        base = run_episode(TrajectoryV6Agent(memory, False), seed, args.max_steps)
        v7, _, _ = run_v7_episode(
            AnticipatoryV7Agent(memory, risk_model, args.threshold, 3), seed, args.max_steps
        )
        v71, diag = run_v71_episode(
            TransitionRescueV71Agent(
                memory,
                risk_model,
                rescue_memory,
                threshold=args.threshold,
                blend=args.blend,
                max_distance=args.max_distance,
                rescue_steps=args.rescue_steps,
            ),
            seed,
            args.max_steps,
        )
        baseline_rows.append(base)
        v7_rows.append(v7)
        v71_rows.append(v71)
        for key in totals:
            totals[key] += int(diag[key])
        if np.isfinite(diag["distance"]):
            distances.append(diag["distance"])
        gains.append(diag["expected_gain"])
        print(
            f"seed={seed} | V6.1={base.steps:3d} E={base.energy:.5f} sims={base.sims_per_step:.2f} "
            f"| V7={v7.steps:3d} E={v7.energy:.5f} sims={v7.sims_per_step:.2f} "
            f"| V7.1={v71.steps:3d} E={v71.energy:.5f} sims={v71.sims_per_step:.2f} "
            f"risk={diag['risk_events']} hit={diag['rescue_hits']} miss={diag['rescue_misses']}"
        )

    print("\n" + "=" * 88)
    print("RESULTADO V7.1 - MEMORIA DE TRANSICOES DE RESGATE")
    print("=" * 88)
    summarize("V6.1 BASELINE", baseline_rows)
    summarize("V7 CAMPO COMPLETO COMO RESGATE", v7_rows)
    summarize("V7.1 TRANSICAO BOA COMO RESGATE", v71_rows)

    b_steps = np.mean([r.steps for r in baseline_rows])
    x_steps = np.mean([r.steps for r in v71_rows])
    b_energy = np.mean([r.energy for r in baseline_rows])
    x_energy = np.mean([r.energy for r in v71_rows])
    b_sims = np.mean([r.sims_per_step for r in baseline_rows])
    x_sims = np.mean([r.sims_per_step for r in v71_rows])
    print("\nV7.1 vs V6.1")
    print(f"Sobrevivencia: {(x_steps / b_steps - 1) * 100:+.2f}%")
    print(f"Energia: {(x_energy / b_energy - 1) * 100:+.2f}%")
    print(f"Sims/passo: {(x_sims / b_sims - 1) * 100:+.2f}%")
    print(f"Eventos de risco: {totals['risk_events']}")
    print(f"Resgates encontrados: {totals['rescue_hits']}")
    print(f"Resgates sem correspondencia: {totals['rescue_misses']}")
    print(f"Acoes de resgate aplicadas: {totals['rescue_actions']}")
    print(f"Distancia media de resgate: {np.mean(distances) if distances else float('nan'):.4f}")
    print(f"Ganho de estabilidade esperado: {np.mean(gains):.5f}")


if __name__ == "__main__":
    main()
