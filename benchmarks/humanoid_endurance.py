from __future__ import annotations

import argparse
import numpy as np

from benchmarks.humanoid_balance import (
    BaseAgent,
    FullFieldAgent,
    MemoryV5Agent,
    TrajectoryV6Agent,
    run_episode,
)
from robocop.memory import DescriptiveMemory
from robocop.trajectory_memory import TrajectoryMemory


def train_memory(agent_factory, seeds, max_steps, label):
    print(f"FASE A - treino {label}")
    for seed in seeds:
        r = run_episode(agent_factory(), seed, max_steps)
        print(f"seed={seed} passos={r.steps} hit={r.hit_rate:.2f} sims={r.sims_per_step:.2f}")


def summarize(name, rows, max_steps):
    steps = np.array([r.steps for r in rows], dtype=float)
    rewards = np.array([r.reward for r in rows], dtype=float)
    energies = np.array([r.energy for r in rows], dtype=float)
    cpus = np.array([r.cpu_ms for r in rows], dtype=float)
    sims = np.array([r.sims_per_step for r in rows], dtype=float)
    hits = np.array([r.hit_rate for r in rows], dtype=float)
    completed = steps >= max_steps
    print(f"\n{name}")
    print(f"Passos medios: {steps.mean():.3f}")
    print(f"Mediana: {np.median(steps):.3f}")
    print(f"Melhor episodio: {steps.max():.0f}")
    print(f"Conclusao completa: {100.0 * completed.mean():.2f}%")
    print(f"Recompensa media: {rewards.mean():.3f}")
    print(f"Energia: {energies.mean():.6f}")
    print(f"CPU: {cpus.mean():.4f} ms/passo")
    print(f"Sims/passo: {sims.mean():.3f}")
    print(f"Hit total: {100.0 * hits.mean():.2f}%")
    for threshold in (100, 250, 500, 750, 1000):
        if threshold <= max_steps:
            print(f">={threshold} passos: {100.0 * np.mean(steps >= threshold):.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--train-seeds", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    train_range = range(900, 900 + args.train_seeds)
    memory_v5 = DescriptiveMemory()
    memory_v61 = TrajectoryMemory()

    train_memory(lambda: MemoryV5Agent(memory_v5, True), train_range, 500, "V5")
    memory_v5.freeze()
    print("Memoria V5:", memory_v5.stats())

    train_memory(lambda: TrajectoryV6Agent(memory_v61, True), train_range, 500, "V6.1")
    memory_v61.freeze()
    print("Memoria V6.1:", memory_v61.stats())

    results = []
    for seed in range(1100, 1100 + args.seeds):
        agents = (
            BaseAgent(),
            FullFieldAgent(),
            MemoryV5Agent(memory_v5, False),
            TrajectoryV6Agent(memory_v61, False),
        )
        for agent in agents:
            r = run_episode(agent, seed, args.max_steps)
            results.append(r)
            status = "COMPLETO" if r.steps >= args.max_steps else "CAIU"
            print(
                f"{r.controller:18s} seed={seed} passos={r.steps} status={status} "
                f"E={r.energy:.6f} CPU={r.cpu_ms:.3f} sims={r.sims_per_step:.2f} hit={r.hit_rate:.2f}"
            )

    print("\n" + "=" * 82)
    print(f"ENDURANCE - LIMITE {args.max_steps} PASSOS")
    print("=" * 82)
    for name in sorted(set(r.controller for r in results)):
        summarize(name, [r for r in results if r.controller == name], args.max_steps)


if __name__ == "__main__":
    main()
