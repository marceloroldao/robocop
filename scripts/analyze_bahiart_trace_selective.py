#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from memory.transition_memory import BalanceState, ResolutiveTransitionMemory


def to_state(obj: dict) -> BalanceState:
    return BalanceState(
        height=float(obj["height"]),
        roll=float(obj["roll"]),
        pitch=float(obj["pitch"]),
        angular_speed=float(obj["angular_speed"]),
        vertical_speed=float(obj["vertical_speed"]),
        support_margin=float(obj.get("support_margin", 0.0)),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Replay a BahiaRT passive trace through selective resolutive memory.")
    p.add_argument("trace", nargs="?", type=Path, default=Path("results/bahiart_passive_trace.jsonl"))
    p.add_argument("--target-height", type=float, default=1.0)
    p.add_argument("--min-gain", type=float, default=0.005)
    p.add_argument("--min-confirmations", type=int, default=3)
    p.add_argument("--min-confidence", type=float, default=0.65)
    args = p.parse_args()

    if not args.trace.exists():
        raise SystemExit(f"trace not found: {args.trace}")

    memory = ResolutiveTransitionMemory(
        min_gain=args.min_gain,
        target_height=args.target_height,
        min_confirmations=args.min_confirmations,
    )

    previous_state = None
    previous_action = None
    total = 0
    recalls = 0
    layers = Counter()
    distances = []
    confidences = []
    admitted = 0

    with args.trace.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("state"):
                continue
            current = to_state(row["state"])
            fallen = bool(row.get("fallen", False))

            if previous_state is not None and previous_action is not None:
                if memory.observe(previous_state, previous_action, current, terminal=fallen):
                    admitted += 1

            recall = memory.recall(
                current,
                recent_state=previous_state,
                min_confidence=args.min_confidence,
            )
            total += 1
            if recall is None:
                layers["MISS"] += 1
            else:
                recalls += 1
                layers[recall.layer] += 1
                distances.append(recall.distance)
                confidences.append(recall.confidence)

            action = row.get("baseline_action")
            previous_state = current
            previous_action = None if action is None else np.asarray(action, dtype=np.float64)

    stats = memory.stats()
    print("=" * 76)
    print("ROBocop — SELECTIVE MEMORY TRACE REPLAY")
    print("=" * 76)
    print(f"Trace cycles:              {total}")
    print(f"Useful observations:       {admitted}")
    print(f"Consolidated prototypes:   {stats['records']}")
    print(f"Confirmed prototypes:      {stats['confirmed_records']}")
    print(f"Observations merged:       {stats['observations_merged']}")
    print(f"Mean confirmations/proto:  {stats['mean_confirmations']:.3f}")
    print(f"Z1 regions:                {stats['z1_regions']}")
    print(f"Z2 patterns:               {stats['z2_patterns']}")
    print()
    for layer in ("Z1", "Z2", "Z3", "MISS"):
        n = layers[layer]
        pct = 100.0 * n / max(total, 1)
        print(f"{layer:4s}: {n:7d}  ({pct:6.2f}%)")
    print()
    print(f"Selective recall rate:     {100.0 * recalls / max(total, 1):.2f}%")
    if distances:
        print(f"Recall distance mean/p95:  {np.mean(distances):.4f} / {np.percentile(distances, 95):.4f}")
        print(f"Recall confidence mean:    {np.mean(confidences):.4f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
