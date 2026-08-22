#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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


def action_metrics(recalled: np.ndarray, baseline: np.ndarray) -> dict[str, float] | None:
    recalled = np.asarray(recalled, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if recalled.shape != baseline.shape or recalled.size == 0:
        return None
    if not np.all(np.isfinite(recalled)) or not np.all(np.isfinite(baseline)):
        return None

    delta = recalled - baseline
    rmse = float(np.sqrt(np.mean(delta * delta)))
    mae = float(np.mean(np.abs(delta)))
    base_rms = float(np.sqrt(np.mean(baseline * baseline)))
    rec_rms = float(np.sqrt(np.mean(recalled * recalled)))
    scale = max(base_rms, rec_rms, 1e-9)
    nrmse = float(rmse / scale)

    denom = float(np.linalg.norm(recalled) * np.linalg.norm(baseline))
    cosine = float(np.dot(recalled, baseline) / denom) if denom > 1e-12 else 1.0
    cosine = float(np.clip(cosine, -1.0, 1.0))

    return {
        "rmse": rmse,
        "mae": mae,
        "nrmse": nrmse,
        "cosine": cosine,
        "baseline_rms": base_rms,
        "recalled_rms": rec_rms,
    }


def summarize(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(np.median(arr)), float(np.percentile(arr, 95))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Replay BahiaRT trace and compare selective resolutive recall actions with baseline actions."
    )
    p.add_argument("trace", nargs="?", type=Path, default=Path("results/bahiart_passive_trace.jsonl"))
    p.add_argument("--target-height", type=float, default=1.0)
    p.add_argument("--min-gain", type=float, default=0.005)
    p.add_argument("--min-confirmations", type=int, default=3)
    p.add_argument("--min-confidence", type=float, default=0.65)
    p.add_argument("--good-nrmse", type=float, default=0.20)
    p.add_argument("--good-cosine", type=float, default=0.95)
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
    total_cycles = 0
    useful = 0
    recall_count = 0
    comparable = 0

    by_layer: dict[str, dict[str, list[float] | int]] = defaultdict(
        lambda: {
            "count": 0,
            "good": 0,
            "nrmse": [],
            "cosine": [],
            "confidence": [],
            "distance": [],
            "confirmations": [],
            "gain": [],
        }
    )

    with args.trace.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            state_obj = row.get("state")
            if not state_obj:
                continue

            current = to_state(state_obj)
            fallen = bool(row.get("fallen", False))
            baseline_obj = row.get("baseline_action")
            baseline = None if baseline_obj is None else np.asarray(baseline_obj, dtype=np.float64).reshape(-1)

            # Match the online passive bridge ordering: first complete the prior
            # transition with the newly received state, then query memory for the
            # current state before the baseline action is used as future evidence.
            if previous_state is not None and previous_action is not None:
                if memory.observe(previous_state, previous_action, current, terminal=fallen):
                    useful += 1

            recall = memory.recall(
                current,
                recent_state=previous_state,
                min_confidence=args.min_confidence,
            )
            total_cycles += 1

            if recall is not None:
                recall_count += 1
                if baseline is not None:
                    metrics = action_metrics(recall.action, baseline)
                    if metrics is not None:
                        comparable += 1
                        bucket = by_layer[recall.layer]
                        bucket["count"] += 1
                        bucket["nrmse"].append(metrics["nrmse"])
                        bucket["cosine"].append(metrics["cosine"])
                        bucket["confidence"].append(recall.confidence)
                        bucket["distance"].append(recall.distance)
                        bucket["confirmations"].append(float(recall.confirmations))
                        bucket["gain"].append(recall.gain)
                        if metrics["nrmse"] <= args.good_nrmse and metrics["cosine"] >= args.good_cosine:
                            bucket["good"] += 1

            previous_state = current
            previous_action = baseline

    stats = memory.stats()
    print("=" * 84)
    print("RoboCOP — SELECTIVE ACTION AGREEMENT REPLAY")
    print("=" * 84)
    print(f"Trace cycles:                 {total_cycles}")
    print(f"Useful observations:          {useful}")
    print(f"Consolidated prototypes:      {stats['records']}")
    print(f"Confirmed prototypes:         {stats['confirmed_records']}")
    print(f"Selective recalls:            {recall_count} ({100.0 * recall_count / max(total_cycles, 1):.2f}%)")
    print(f"Comparable recall/actions:    {comparable}")
    print(f"Good-action thresholds:       NRMSE <= {args.good_nrmse:.3f}, cosine >= {args.good_cosine:.3f}")
    print()

    all_nrmse: list[float] = []
    all_cosine: list[float] = []
    all_good = 0
    all_count = 0

    for layer in ("Z1", "Z2", "Z3"):
        bucket = by_layer[layer]
        count = int(bucket["count"])
        good = int(bucket["good"])
        nrmse = list(bucket["nrmse"])
        cosine = list(bucket["cosine"])
        confidence = list(bucket["confidence"])
        distance = list(bucket["distance"])
        confirmations = list(bucket["confirmations"])
        gains = list(bucket["gain"])

        all_nrmse.extend(nrmse)
        all_cosine.extend(cosine)
        all_good += good
        all_count += count

        n_mean, n_med, n_p95 = summarize(nrmse)
        c_mean, c_med, c_p95 = summarize(cosine)
        conf_mean, _, _ = summarize(confidence)
        dist_mean, _, dist_p95 = summarize(distance)
        confirm_mean, _, _ = summarize(confirmations)
        gain_mean, _, _ = summarize(gains)

        print(f"{layer}")
        print(f"  comparisons:                {count}")
        print(f"  good action agreement:      {good} ({100.0 * good / max(count, 1):.2f}%)")
        print(f"  NRMSE mean/median/p95:       {n_mean:.4f} / {n_med:.4f} / {n_p95:.4f}")
        print(f"  cosine mean/median/p95:      {c_mean:.4f} / {c_med:.4f} / {c_p95:.4f}")
        print(f"  confidence mean:            {conf_mean:.4f}")
        print(f"  distance mean/p95:          {dist_mean:.4f} / {dist_p95:.4f}")
        print(f"  confirmations mean:         {confirm_mean:.2f}")
        print(f"  historical gain mean:       {gain_mean:.6f}")
        print()

    n_mean, n_med, n_p95 = summarize(all_nrmse)
    c_mean, c_med, _ = summarize(all_cosine)
    print("OVERALL")
    print(f"  good action agreement:      {all_good} ({100.0 * all_good / max(all_count, 1):.2f}%)")
    print(f"  NRMSE mean/median/p95:       {n_mean:.4f} / {n_med:.4f} / {n_p95:.4f}")
    print(f"  cosine mean/median:          {c_mean:.4f} / {c_med:.4f}")
    print()
    print("Interpretation for first active experiment:")
    print("  - Z1/Z2 remain observation-only.")
    print("  - Z3 becomes eligible only if action agreement is strong enough.")
    print("  - MISS always leaves BahiaRT baseline fully authoritative.")
    print("=" * 84)


if __name__ == "__main__":
    main()
