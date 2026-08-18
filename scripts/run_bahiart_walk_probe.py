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
    # Four deliberately simple relative commands. The BahiaRT Walk network
    # remains the sole generator of joint targets.
    phase = (cycle // block) % 4
    commands = (
        np.array([0.35, 0.00], dtype=float),   # forward
        np.array([0.15, 0.18], dtype=float),   # forward-left
        np.array([-0.20, 0.00], dtype=float),  # backward
        np.array([0.15, -0.18], dtype=float),  # forward-right
    )
    return commands[phase]


def main() -> None:
    p = argparse.ArgumentParser(description="Exercise BahiaRT Walk while RoboCOP passively records transitions.")
    p.add_argument("--team", default="RoboCOPWalkProbe")
    p.add_argument("--number", type=int, default=2)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=60000)
    p.add_argument("--field", default="fifa")
    p.add_argument("--target-height", type=float, default=1.0)
    p.add_argument("--min-gain", type=float, default=0.005)
    p.add_argument("--recall-confidence", type=float, default=0.65)
    p.add_argument("--block", type=int, default=750, help="cycles per commanded velocity phase")
    p.add_argument("--max-cycles", type=int, default=12000)
    p.add_argument("--trace", type=Path, default=ROOT / "results" / "bahiart_walk_probe_trace.jsonl")
    args = p.parse_args()

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.write_text("", encoding="utf-8")

    memory = ResolutiveTransitionMemory(
        min_gain=args.min_gain,
        target_height=args.target_height,
    )
    bridge = BahiaRTPassiveBridge(memory, recall_confidence=args.recall_confidence)
    agent = Agent(team_name=args.team, number=args.number, host=args.host, port=args.port, field=args.field)

    cycle = 0

    def walk_probe_decision():
        nonlocal cycle
        cycle += 1
        admitted = bridge.before_decision(agent)
        recall = bridge.current_recall

        velocity = velocity_for_cycle(cycle, args.block)
        # The external BahiaRT Walk skill computes the full 23-joint command.
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
            "cycle": cycle,
            "server_time": getattr(agent.world, "server_time", None),
            "fallen": bool(agent.world.is_fallen()),
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

        if cycle % 100 == 0:
            print(
                f"[RoboCOP-WALK] cycle={cycle} records={memory.size} "
                f"recalls={stats.recalls} fallen={row['fallen']} "
                f"cmd=({velocity[0]:+.2f},{velocity[1]:+.2f})"
            )
        if cycle >= args.max_cycles:
            raise KeyboardInterrupt

    agent.decision_maker.update_current_behavior = walk_probe_decision

    print("[RoboCOP-WALK] BahiaRT Walk probe enabled")
    print("[RoboCOP-WALK] BahiaRT Walk network remains the only joint-action generator")
    print(f"[RoboCOP-WALK] trace: {args.trace}")
    try:
        agent.run()
    except KeyboardInterrupt:
        print(f"[RoboCOP-WALK] completed {cycle} cycles")


if __name__ == "__main__":
    main()
