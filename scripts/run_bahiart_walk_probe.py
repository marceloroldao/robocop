#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / ".external" / "BahiaRT-MujOCo-base"
if not EXTERNAL.exists():
    raise SystemExit("BahiaRT external checkout not found. Run bash scripts/fetch_bahiart_mujoco_external.sh first.")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXTERNAL))

from memory.transition_memory import ResolutiveTransitionMemory  # noqa: E402
from robocop.integrations.bahiart_passive import BahiaRTPassiveBridge  # noqa: E402
from mujococodebase.agent import Agent  # type: ignore  # noqa: E402


def state_dict(state):
    return {
        "height": state.height,
        "roll": state.roll,
        "pitch": state.pitch,
        "angular_speed": state.angular_speed,
        "vertical_speed": state.vertical_speed,
        "support_margin": state.support_margin,
    }


def velocity_for_cycle(cycle: int, block: int) -> np.ndarray:
    phase = (cycle // block) % 4
    commands = (
        np.array([0.35, 0.00], dtype=float),
        np.array([0.15, 0.18], dtype=float),
        np.array([-0.20, 0.00], dtype=float),
        np.array([0.15, -0.18], dtype=float),
    )
    return commands[phase]


def main() -> None:
    p = argparse.ArgumentParser(description="Episodic BahiaRT Walk probe with passive RoboCOP transition recording.")
    p.add_argument("--team", default="RoboCOPWalkProbe")
    p.add_argument("--number", type=int, default=2)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=60000)
    p.add_argument("--field", default="fifa")
    p.add_argument("--target-height", type=float, default=1.0)
    p.add_argument("--min-gain", type=float, default=0.005)
    p.add_argument("--recall-confidence", type=float, default=0.65)
    p.add_argument("--block", type=int, default=150, help="walking cycles per commanded velocity phase")
    p.add_argument("--max-walk-cycles", type=int, default=12000)
    p.add_argument("--max-recovery-cycles", type=int, default=2500)
    p.add_argument("--trace", type=Path, default=ROOT / "results" / "bahiart_walk_probe_trace.jsonl")
    args = p.parse_args()

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.write_text("", encoding="utf-8")

    memory = ResolutiveTransitionMemory(min_gain=args.min_gain, target_height=args.target_height)
    bridge = BahiaRTPassiveBridge(memory, recall_confidence=args.recall_confidence)
    agent = Agent(team_name=args.team, number=args.number, host=args.host, port=args.port, field=args.field)

    sim_cycle = 0
    walk_cycle = 0
    episode = 1
    episode_walk_cycle = 0
    recovering = False
    recovery_cycles = 0
    falls = 0
    episode_lengths: list[int] = []
    stop_reason = "MAX_WALK_CYCLES"

    def walk_probe_decision():
        nonlocal sim_cycle, walk_cycle, episode, episode_walk_cycle
        nonlocal recovering, recovery_cycles, falls, stop_reason

        sim_cycle += 1
        fallen = bool(agent.world.is_fallen())

        if recovering:
            recovery_cycles += 1
            finished = agent.skills_manager.execute("GetUp")
            agent.robot.commit_motor_targets_pd()
            if finished and not bool(agent.world.is_fallen()):
                bridge.reset_temporal_context()
                recovering = False
                recovery_cycles = 0
                episode += 1
                episode_walk_cycle = 0
                print(f"[RoboCOP-WALK] recovered -> episode={episode} sim_cycle={sim_cycle}")
            elif recovery_cycles >= args.max_recovery_cycles:
                stop_reason = "RECOVERY_TIMEOUT"
                print(
                    f"[RoboCOP-WALK] RECOVERY_TIMEOUT episode={episode} "
                    f"recovery_cycles={recovery_cycles} walk_total={walk_cycle}"
                )
                raise KeyboardInterrupt
            return

        if fallen:
            bridge.before_decision(agent)
            bridge.reset_temporal_context()
            falls += 1
            episode_lengths.append(episode_walk_cycle)
            recovering = True
            recovery_cycles = 0
            print(
                f"[RoboCOP-WALK] FALL episode={episode} walk_len={episode_walk_cycle} "
                f"walk_total={walk_cycle} records={memory.size}"
            )
            agent.skills_manager.execute("GetUp")
            agent.robot.commit_motor_targets_pd()
            return

        walk_cycle += 1
        episode_walk_cycle += 1
        admitted = bridge.before_decision(agent)
        recall = bridge.current_recall

        velocity = velocity_for_cycle(walk_cycle - 1, args.block)
        agent.skills_manager.execute(
            "Walk",
            target_2d=velocity,
            is_target_absolute=False,
            orientation=0.0,
            is_orientation_absolute=False,
        )
        agent.robot.commit_motor_targets_pd()
        action = bridge.after_decision(agent)

        current = bridge._current_state
        stats = bridge.stats()
        row = {
            "sim_cycle": sim_cycle,
            "cycle": walk_cycle,
            "episode": episode,
            "episode_cycle": episode_walk_cycle,
            "server_time": getattr(agent.world, "server_time", None),
            "fallen": False,
            "command_velocity": velocity.tolist(),
            "admitted_previous_transition": admitted,
            "state": state_dict(current) if current is not None else None,
            "baseline_action": np.asarray(action, dtype=float).tolist(),
            "recall": None if recall is None else {
                "action": recall.action.tolist(),
                "confidence": recall.confidence,
                "gain": recall.gain,
                "distance": recall.distance,
                "z1_match": recall.z1_match,
                "z2_match": recall.z2_match,
                "layer": getattr(recall, "layer", "legacy"),
                "confirmations": getattr(recall, "confirmations", 1),
            },
            "memory": memory.stats(),
            "probe": {
                "cycles": stats.cycles,
                "completed_transitions": stats.completed_transitions,
                "admitted_transitions": stats.admitted_transitions,
                "recalls": stats.recalls,
            },
        }
        with args.trace.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, separators=(",", ":")) + "\n")

        if walk_cycle % 100 == 0:
            print(
                f"[RoboCOP-WALK] walk={walk_cycle} episode={episode} ep_cycle={episode_walk_cycle} "
                f"records={memory.size} recalls={stats.recalls} falls={falls} "
                f"cmd=({velocity[0]:+.2f},{velocity[1]:+.2f})"
            )
        if walk_cycle >= args.max_walk_cycles:
            episode_lengths.append(episode_walk_cycle)
            stop_reason = "MAX_WALK_CYCLES"
            raise KeyboardInterrupt

    agent.decision_maker.update_current_behavior = walk_probe_decision

    print("[RoboCOP-WALK] episodic BahiaRT Walk probe enabled")
    print("[RoboCOP-WALK] BahiaRT Walk network remains the only walking joint-action generator")
    print("[RoboCOP-WALK] GetUp cycles are excluded from transition learning")
    print(f"[RoboCOP-WALK] trace: {args.trace}")
    try:
        agent.run()
    except KeyboardInterrupt:
        mean_len = float(np.mean(episode_lengths)) if episode_lengths else float(episode_walk_cycle)
        best_len = max(episode_lengths) if episode_lengths else episode_walk_cycle
        print(
            f"[RoboCOP-WALK] completed reason={stop_reason} walk_cycles={walk_cycle} "
            f"episodes={episode} falls={falls} mean_episode={mean_len:.1f} "
            f"best_episode={best_len} records={memory.size}"
        )


if __name__ == "__main__":
    main()
