from __future__ import annotations

import argparse

import numpy as np

from benchmarks.humanoid_balance import TrajectoryV6Agent
from robocop.mujoco_env import make_humanoid_env, extract_humanoid_state
from robocop.trajectory_memory import TrajectoryMemory
from robocop.transition_memory import SensorSnapshot, TransitionSample, TransitionRecorder, separation_report


def train_v61(memory: TrajectoryMemory, train_seeds: int, max_steps: int) -> None:
    from benchmarks.humanoid_balance import run_episode

    print("FASE A - treino V6.1")
    for seed in range(900, 900 + train_seeds):
        r = run_episode(TrajectoryV6Agent(memory, True), seed, max_steps)
        print(f"seed={seed} passos={r.steps} hit={r.hit_rate:.2f} sims={r.sims_per_step:.2f}")
    memory.freeze()
    print("Memoria V6.1:", memory.stats())


def record_episode(memory: TrajectoryMemory, seed: int, max_steps: int) -> TransitionRecorder:
    env = make_humanoid_env()
    env.reset(seed=seed)
    agent = TrajectoryV6Agent(memory, False)
    agent.reset(env)
    recorder = TransitionRecorder()

    try:
        for step in range(max_steps):
            before_obs = extract_humanoid_state(env, agent.q_target)
            before = SensorSnapshot.from_parts(before_obs.field_state, before_obs.q, before_obs.qd)
            action = agent.act(env)
            _, reward, terminated, truncated, _ = env.step(action)
            after_obs = extract_humanoid_state(env, agent.q_target)
            after = SensorSnapshot.from_parts(after_obs.field_state, after_obs.q, after_obs.qd)
            recorder.add(
                TransitionSample(
                    before=before,
                    action=np.asarray(action, dtype=float).copy(),
                    after=after,
                    reward=float(reward),
                    energy=float(np.mean(np.asarray(action, dtype=float) ** 2)),
                    terminated=bool(terminated or truncated),
                    step=step,
                )
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    return recorder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--train-seeds", type=int, default=30)
    parser.add_argument("--train-max-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--prefall-window", type=int, default=12)
    args = parser.parse_args()

    memory = TrajectoryMemory()
    train_v61(memory, args.train_seeds, args.train_max_steps)

    recorders = []
    print("\nFASE B - gravando transicoes sensoriais")
    for seed in range(1200, 1200 + args.seeds):
        rec = record_episode(memory, seed, args.max_steps)
        recorders.append(rec)
        fell = bool(rec.samples and rec.samples[-1].terminated)
        last = rec.samples[-1] if rec.samples else None
        print(
            f"seed={seed} passos={len(rec.samples)} caiu={fell} "
            f"altura_final={(last.after.height if last else float('nan')):.3f} "
            f"vertical_final={(last.after.vertical if last else float('nan')):.3f} "
            f"omega_final={(last.after.omega if last else float('nan')):.3f}"
        )

    report = separation_report(recorders, prefall_window=args.prefall_window)
    print("\n" + "=" * 82)
    print("V7 - DIAGNOSTICO DE TRANSICOES PRE-QUEDA")
    print("=" * 82)
    print(f"Transicoes totais: {report['n']}")
    print(f"Transicoes pre-queda: {report['prefall']}")
    print(f"Transicoes estaveis: {report['stable']}")
    print(f"Distancia entre centroides padronizados: {report['centroid_distance']:.4f}")
    print("Variaveis mais separadoras:")
    for name, effect in report["top_features"]:
        print(f"  {name:12s} efeito={effect:.4f}")

    if report["centroid_distance"] >= 1.0:
        print("\nSINAL: FORTE - ha separacao preditiva suficiente para testar controle antecipatorio.")
    elif report["centroid_distance"] >= 0.5:
        print("\nSINAL: MODERADO - ha estrutura preditiva, mas precisa refinamento.")
    else:
        print("\nSINAL: FRACO - sensores atuais nao separam bem a pre-queda; ampliar observacao antes da V7 de controle.")


if __name__ == "__main__":
    main()
