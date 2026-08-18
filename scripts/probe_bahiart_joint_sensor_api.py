#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / ".external" / "BahiaRT-MujOCo-base"
if not EXTERNAL.exists():
    raise SystemExit("BahiaRT external checkout not found")
sys.path.insert(0, str(EXTERNAL))

from mujococodebase.agent import Agent  # type: ignore  # noqa: E402


def describe(name: str, value) -> None:
    print(f"[{name}] type={type(value)!r}")
    if isinstance(value, dict):
        keys = list(value.keys())
        vals = list(value.values())
        print(f"[{name}] len={len(value)} keys={keys[:30]!r}")
        print(f"[{name}] sample_values={vals[:10]!r}")
    else:
        try:
            arr = np.asarray(value)
            print(f"[{name}] shape={arr.shape} dtype={arr.dtype} sample={arr.reshape(-1)[:10].tolist()}")
        except Exception as exc:
            print(f"[{name}] repr={value!r} conversion_error={exc!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=60000)
    p.add_argument("--number", type=int, default=2)
    args = p.parse_args()

    agent = Agent(team_name="RoboCOPSensorAudit", number=args.number, host=args.host, port=args.port, field="fifa")
    cycles = 0

    def decision():
        nonlocal cycles
        cycles += 1
        if cycles == 1:
            robot = agent.robot
            print("=== RoboCOP V11.3 JOINT SENSOR API AUDIT ===")
            print(f"ROBOT_MOTORS type={type(robot.ROBOT_MOTORS)!r} value={robot.ROBOT_MOTORS!r}")
            for attr in ("motor_positions", "motor_speeds", "target_position"):
                if hasattr(robot, attr):
                    describe(attr, getattr(robot, attr))
                else:
                    print(f"[{attr}] MISSING")
        agent.skills_manager.execute(
            "Walk", target_2d=np.array([0.20, 0.0]), is_target_absolute=False,
            orientation=0.0, is_orientation_absolute=False,
        )
        agent.robot.commit_motor_targets_pd()
        if cycles >= 8:
            raise KeyboardInterrupt

    agent.decision_maker.update_current_behavior = decision
    try:
        agent.run()
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        if getattr(exc, "errno", None) != 9:
            raise


if __name__ == "__main__":
    main()
