from __future__ import annotations

import argparse
import time

import numpy as np

from benchmarks.humanoid_balance import EpisodeResult, TrajectoryV6Agent, run_episode
from benchmarks.transition_diagnostics import record_episode, train_v61
from robocop.balance_reflex import BalanceReflexMemory
from robocop.mujoco_env import extract_humanoid_state, make_humanoid_env
from robocop.prefall_model import PrefallRiskModel, make_transition
from robocop.trajectory_memory import TrajectoryMemory
from robocop.transition_memory import SensorSnapshot


class BalanceReflexV72Agent(TrajectoryV6Agent):
    name = "REFLEXOS V7.2"

    def __init__(self, memory, risk_model, reflex_memory, threshold=0.60, blend=0.35, max_distance=1.25):
        super().__init__(memory, False)
        self.risk_model = risk_model
        self.reflex_memory = reflex_memory
        self.threshold = float(threshold)
        self.blend = float(blend)
        self.max_distance = float(max_distance)
        self.prev_snapshot = None
        self.last_snapshot = None
        self.last_action = None
        self.last_reward = 0.0
        self.risk_events = 0
        self.reflex_hits = 0
        self.reflex_misses = 0

    def reset(self, env):
        super().reset(env)
        obs = extract_humanoid_state(env, self.q_target)
        snap = SensorSnapshot.from_parts(obs.field_state, obs.q, obs.qd)
        self.prev_snapshot = snap
        self.last_snapshot = snap
        self.last_action = np.zeros(env.action_space.shape[0], dtype=float)
        self.last_reward = 0.0
        self.risk_events = 0
        self.reflex_hits = 0
        self.reflex_misses = 0

    def observe(self, reward: float, terminated: bool, truncated: bool) -> None:
        self.last_reward = float(reward)

    def act(self, env):
        obs = extract_humanoid_state(env, self.q_target)
        current = SensorSnapshot.from_parts(obs.field_state, obs.q, obs.qd)

        risk = 0.0
        if self.last_snapshot is not None and self.last_action is not None:
            transition = make_transition(self.last_snapshot, self.last_action, current, self.last_reward, False, 0)
            risk = self.risk_model.risk(transition)

        normal = super().act(env)
        action = normal

        if risk >= self.threshold and self.prev_snapshot is not None:
            self.risk_events += 1
            reflex, gain, distance = self.reflex_memory.lookup(self.prev_snapshot, current)
            if reflex is not None and gain > 0.0 and distance <= self.max_distance:
                action = np.clip((1.0 - self.blend) * normal + self.blend * reflex,
                                 env.action_space.low, env.action_space.high)
                self.reflex_hits += 1
            else:
                self.reflex_misses += 1

        self.prev_snapshot = self.last_snapshot
        self.last_snapshot = current
        self.last_action = np.asarray(action, dtype=float).copy()
        return action


def run_v72_episode(agent, seed, max_steps):
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
    d = max(steps, 1)
    ld = max(agent.lookups, 1)
    return EpisodeResult(
        controller=agent.name, seed=seed, steps=steps, reward=reward_total,
        reward_per_step=reward_total/d, torque=torque_total/d, energy=energy_total/d,
        cpu_ms=cpu_total/d*1000.0, sims_per_step=agent.sims/d,
        hit_rate=agent.hits/ld, hit_z1=agent.level_hits[1]/ld,
        hit_z2=agent.level_hits[2]/ld, hit_z3=agent.level_hits[3]/ld,
    ), agent.risk_events, agent.reflex_hits, agent.reflex_misses


def summarize(label, rows):
    s = np.asarray([r.steps for r in rows], float)
    e = np.asarray([r.energy for r in rows], float)
    sims = np.asarray([r.sims_per_step for r in rows], float)
    print(f"\n{label}")
    print(f"Passos: {s.mean():.3f}")
    print(f"Energia: {e.mean():.6f}")
    print(f"Sims/passo: {sims.mean():.3f}")
    print(f">=100 passos: {100*np.mean(s>=100):.2f}%")
    print(f">=250 passos: {100*np.mean(s>=250):.2f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=30)
    p.add_argument('--train-seeds', type=int, default=30)
    p.add_argument('--risk-seeds', type=int, default=40)
    p.add_argument('--max-steps', type=int, default=500)
    p.add_argument('--threshold', type=float, default=0.60)
    p.add_argument('--blend', type=float, default=0.35)
    p.add_argument('--max-distance', type=float, default=1.25)
    p.add_argument('--prefall-window', type=int, default=12)
    args = p.parse_args()

    memory = TrajectoryMemory()
    train_v61(memory, args.train_seeds, 500)

    print('\nFASE B - aprendendo risco e reflexos')
    recorders = []
    for seed in range(1200, 1200 + args.risk_seeds):
        rec = record_episode(memory, seed, 500)
        recorders.append(rec)
        print(f"seed={seed} transicoes={len(rec.samples)}")

    risk_model = PrefallRiskModel.fit(recorders, prefall_window=args.prefall_window)
    reflex_memory = BalanceReflexMemory.fit(recorders, prefall_window=args.prefall_window)
    print('Memoria de reflexos:', reflex_memory.stats())

    base_rows, reflex_rows = [], []
    total_risk = total_hits = total_misses = 0
    print('\nFASE C - teste cego V6.1 vs V7.2')
    for seed in range(1600, 1600 + args.seeds):
        base = run_episode(TrajectoryV6Agent(memory, False), seed, args.max_steps)
        reflex, risk, hits, misses = run_v72_episode(
            BalanceReflexV72Agent(memory, risk_model, reflex_memory,
                                  args.threshold, args.blend, args.max_distance),
            seed, args.max_steps)
        base_rows.append(base)
        reflex_rows.append(reflex)
        total_risk += risk; total_hits += hits; total_misses += misses
        print(f"seed={seed} | V6.1={base.steps:3d} E={base.energy:.5f} sims={base.sims_per_step:.2f} "
              f"| V7.2={reflex.steps:3d} E={reflex.energy:.5f} sims={reflex.sims_per_step:.2f} "
              f"risk={risk} reflex={hits} miss={misses}")

    print('\n' + '='*84)
    print('RESULTADO V7.2 - MEMORIA DE REFLEXOS DE EQUILIBRIO')
    print('='*84)
    summarize('V6.1 BASELINE', base_rows)
    summarize('V7.2 REFLEXOS', reflex_rows)
    b = np.mean([r.steps for r in base_rows]); x = np.mean([r.steps for r in reflex_rows])
    be = np.mean([r.energy for r in base_rows]); xe = np.mean([r.energy for r in reflex_rows])
    bs = np.mean([r.sims_per_step for r in base_rows]); xs = np.mean([r.sims_per_step for r in reflex_rows])
    print('\nV7.2 vs V6.1')
    print(f"Sobrevivencia: {(x/b-1)*100:+.2f}%")
    print(f"Energia: {(xe/be-1)*100:+.2f}%")
    print(f"Sims/passo: {(xs/bs-1)*100:+.2f}%")
    print(f"Eventos de risco: {total_risk}")
    print(f"Reflexos encontrados: {total_hits}")
    print(f"Reflexos sem correspondencia: {total_misses}")


if __name__ == '__main__':
    main()
