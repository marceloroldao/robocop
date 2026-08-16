#!/usr/bin/env python3
from __future__ import annotations

import argparse
from robocop.fcp_transition_analysis import analyze_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze FC Portugal transition JSONL recorded by RoboCOP")
    p.add_argument("path")
    p.add_argument("--min-gain", type=float, default=0.01)
    args = p.parse_args()

    s = analyze_jsonl(args.path, min_gain=args.min_gain)
    print("FC PORTUGAL TRANSITION ANALYSIS")
    print(f"Transitions: {s['transitions']}")
    print(f"Stabilizing: {s['stabilizing']}")
    print(f"Stabilizing fraction: {100*s['stabilizing_fraction']:.2f}%")
    print(f"Terminal: {s['terminal']}")
    print(f"Mean gain all: {s['mean_gain_all']:.6f}")
    print(f"Mean gain stabilizing: {s['mean_gain_stabilizing']:.6f}")
    print(f"Mean energy stabilizing: {s['mean_energy_stabilizing']:.6f}")


if __name__ == "__main__":
    main()
