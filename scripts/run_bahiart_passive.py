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
    raise SystemExit(
        "BahiaRT external checkout not found. Run "
        "bash scripts/fetch_bahiart_mujoco_external.sh first."
    )

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BahiaRT unchanged while RoboCOP passively learns transitions."
    )
    parser.add_argument("--team", default="RoboCOPProbe")
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=60000)
    parser.add_argument("--field", default="fifa")
    parser.add_argument("--target-height", type=float, default=1.0)
    parser.add_argument("--min-gain", type=float, default=0.005)
    parser.add_argument("--recall-confidence", type=float, default=0.65)
    parser.add_argument(
        "--trace",
        type=Path,
        default=ROOT / "results" / "bahiart_passive_trace.jsonl",
    )
    args = parser.parse_args()

    args.trace.parent.mkdir(parents=True, exist_ok=True)

    memory = ResolutiveTransitionMemory(
        min_gain=args.min_gain,
        target_height=args.target_height,
    )
    bridge = BahiaRTPassiveBridge(
        memory,
        recall_confidence=args.recall_confidence,
    )
    agent = Agent(
        team_name=args.team,
        number=args.number,
        host=args.host,
        port=args.port,
        field=args.field,
    )

    original_decision = agent.decision_maker.update_current_behavior
    cycle = 0

    def instrumented_decision():
        nonlocal cycle
        cycle += 1
        admitted = bridge.before_decision(agent)
        recall = bridge.current_recall

        # Baseline decision runs unchanged and remains the only authority that
        # writes motor targets.
        original_decision()
        action = bridge.after_decision(agent)

        current = bridge._current_state
        stats = bridge.stats()
        row = {
            "cycle": cycle,
            "server_time": getattr(agent.world, "server_time", None),
            "fallen": bool(agent.world.is_fallen()),
            "admitted_previous_transition": admitted,
            "state": state_dict(current) if current is not None else None,
            "baseline_action": np.asarray(action, dtype=float).tolist(),
            "recall": None
            if recall is None
            else {
                "action": recall.action.tolist(),
                "confidence": recall.confidence,
                "gain": recall.gain,
                "distance": recall.distance,
                "z1_match": recall.z1_match,
                "z2_match": recall.z2_match,
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
                f"[RoboCOP] cycle={cycle} records={memory.size} "
                f"recalls={stats.recalls} fallen={row['fallen']}"
            )

    agent.decision_maker.update_current_behavior = instrumented_decision

    print("[RoboCOP] BahiaRT passive instrumentation enabled")
    print(f"[RoboCOP] external checkout: {EXTERNAL}")
    print(f"[RoboCOP] trace: {args.trace}")
    print("[RoboCOP] baseline motor commands are NOT modified")
    agent.run()


if __name__ == "__main__":
    main()
