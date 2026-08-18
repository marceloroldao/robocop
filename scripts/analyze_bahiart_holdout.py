#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    scale = max(1.0, float(np.sqrt(np.mean(b * b))))
    return float(np.sqrt(np.mean((a - b) ** 2)) / scale)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da == 0.0 and db == 0.0:
        return 1.0
    if da == 0.0 or db == 0.0:
        return 0.0
    return float(np.dot(a, b) / (da * db))


def summarize(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    return float(arr.mean()), float(np.median(arr)), float(np.percentile(arr, 95))


def main() -> None:
    p = argparse.ArgumentParser(description="Frozen-memory temporal holdout evaluation for BahiaRT trace.")
    p.add_argument("trace", nargs="?", type=Path, default=Path("results/bahiart_passive_trace.jsonl"))
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--target-height", type=float, default=1.0)
    p.add_argument("--min-gain", type=float, default=0.005)
    p.add_argument("--min-confirmations", type=int, default=3)
    p.add_argument("--min-confidence", type=float, default=0.65)
    p.add_argument("--good-nrmse", type=float, default=0.20)
    p.add_argument("--good-cosine", type=float, default=0.95)
    args = p.parse_args()

    if not 0.1 <= args.train_fraction <= 0.9:
        raise SystemExit("--train-fraction must be between 0.1 and 0.9")
    if not args.trace.exists():
        raise SystemExit(f"trace not found: {args.trace}")

    rows = []
    with args.trace.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("state") and row.get("baseline_action") is not None:
                rows.append(row)

    if len(rows) < 10:
        raise SystemExit("trace has too few usable rows")

    split = int(len(rows) * args.train_fraction)
    train_rows = rows[:split]
    test_rows = rows[split:]

    memory = ResolutiveTransitionMemory(
        min_gain=args.min_gain,
        target_height=args.target_height,
        min_confirmations=args.min_confirmations,
    )

    prev_state = None
    prev_action = None
    admitted_train = 0
    for row in train_rows:
        current = to_state(row["state"])
        fallen = bool(row.get("fallen", False))
        if prev_state is not None and prev_action is not None:
            if memory.observe(prev_state, prev_action, current, terminal=fallen):
                admitted_train += 1
        prev_state = current
        prev_action = np.asarray(row["baseline_action"], dtype=np.float64)

    frozen_stats = memory.stats().copy()

    # Critical methodological guarantee: from here onward, observe() is never called.
    layers = Counter()
    recalls = 0
    distances = []
    confidences = []
    metrics = defaultdict(lambda: {"nrmse": [], "cosine": [], "good": 0, "count": 0, "confirmations": [], "gain": []})

    # Maintain temporal trend across the train/test boundary, but do not learn from it.
    recent_state = prev_state
    for row in test_rows:
        current = to_state(row["state"])
        baseline = np.asarray(row["baseline_action"], dtype=np.float64)
        recall = memory.recall(current, recent_state=recent_state, min_confidence=args.min_confidence)
        if recall is None:
            layers["MISS"] += 1
        else:
            recalls += 1
            layers[recall.layer] += 1
            distances.append(recall.distance)
            confidences.append(recall.confidence)
            e = nrmse(recall.action, baseline)
            c = cosine(recall.action, baseline)
            m = metrics[recall.layer]
            m["count"] += 1
            m["nrmse"].append(e)
            m["cosine"].append(c)
            m["confirmations"].append(recall.confirmations)
            m["gain"].append(recall.gain)
            if e <= args.good_nrmse and c >= args.good_cosine:
                m["good"] += 1
        recent_state = current

    after_stats = memory.stats().copy()
    frozen_ok = frozen_stats == after_stats

    print("=" * 84)
    print("RoboCOP — V8.1 FROZEN-MEMORY TEMPORAL HOLDOUT")
    print("=" * 84)
    print(f"Usable trace rows:             {len(rows)}")
    print(f"Training rows:                 {len(train_rows)} ({100*len(train_rows)/len(rows):.2f}%)")
    print(f"Holdout rows:                  {len(test_rows)} ({100*len(test_rows)/len(rows):.2f}%)")
    print(f"Training useful observations:  {admitted_train}")
    print(f"Frozen prototypes:             {frozen_stats['records']}")
    print(f"Frozen confirmed prototypes:   {frozen_stats['confirmed_records']}")
    print(f"Frozen observations merged:    {frozen_stats['observations_merged']}")
    print(f"Memory unchanged in holdout:   {'PASS' if frozen_ok else 'FAIL'}")
    print()
    total_test = len(test_rows)
    for layer in ("Z1", "Z2", "Z3", "MISS"):
        n = layers[layer]
        print(f"{layer:4s}: {n:7d} ({100.0*n/max(total_test,1):6.2f}%)")
    print()
    print(f"Holdout selective recall rate: {100.0*recalls/max(total_test,1):.2f}%")
    if distances:
        print(f"Recall distance mean/p95:      {np.mean(distances):.4f} / {np.percentile(distances,95):.4f}")
        print(f"Recall confidence mean:        {np.mean(confidences):.4f}")
    print()
    print(f"Good-action thresholds:        NRMSE <= {args.good_nrmse:.3f}, cosine >= {args.good_cosine:.3f}")
    for layer in ("Z1", "Z2", "Z3"):
        m = metrics[layer]
        print(f"\n{layer}")
        print(f"  comparisons:                 {m['count']}")
        if m["count"]:
            ns = summarize(m["nrmse"])
            cs = summarize(m["cosine"])
            print(f"  good action agreement:       {m['good']} ({100.0*m['good']/m['count']:.2f}%)")
            print(f"  NRMSE mean/median/p95:        {ns[0]:.4f} / {ns[1]:.4f} / {ns[2]:.4f}")
            print(f"  cosine mean/median/p95:       {cs[0]:.4f} / {cs[1]:.4f} / {cs[2]:.4f}")
            print(f"  confirmations mean:          {np.mean(m['confirmations']):.2f}")
            print(f"  historical gain mean:        {np.mean(m['gain']):.6f}")
    print("=" * 84)


if __name__ == "__main__":
    main()
