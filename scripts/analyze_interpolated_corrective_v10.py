#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory.interpolated_corrective_memory import InterpolatedCorrectiveTrajectoryMemory  # noqa: E402
from memory.transition_memory import BalanceState  # noqa: E402


def to_state(row: dict) -> BalanceState:
    s = row["state"]
    return BalanceState(
        height=float(s["height"]),
        roll=float(s["roll"]),
        pitch=float(s["pitch"]),
        angular_speed=float(s["angular_speed"]),
        vertical_speed=float(s["vertical_speed"]),
        support_margin=float(s.get("support_margin", 0.0)),
    )


def load_episodes(path: Path) -> dict[int, list[dict]]:
    episodes: dict[int, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not row.get("state") or not row.get("baseline_action"):
                continue
            run = int(row.get("run_id", row.get("episode", 0)))
            episodes[run].append(row)
    return dict(sorted(episodes.items()))


def iter_windows(rows: list[dict], context: int, horizon: int):
    n = len(rows)
    for i in range(context - 1, n - horizon):
        history = [to_state(x) for x in rows[i - context + 1 : i + 1]]
        actions = np.asarray([x["baseline_action"] for x in rows[i : i + horizon]], dtype=float)
        future = [to_state(x) for x in rows[i + 1 : i + horizon + 1]]
        yield history, actions, future


def metrics(predicted: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    denom = max(1.0, float(np.sqrt(np.mean(actual * actual))))
    nrmse = float(np.sqrt(np.mean((predicted - actual) ** 2)) / denom)
    a = predicted.reshape(-1)
    b = actual.reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    cosine = float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0
    return nrmse, cosine


def main() -> None:
    p = argparse.ArgumentParser(description="RoboCOP V10 interpolated corrective trajectory benchmark")
    p.add_argument("--trace", type=Path, default=ROOT / "results/bahiart_multi_episode/combined_trace.jsonl")
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--context", type=int, default=5)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--direct-min-confidence", type=float, default=0.60)
    p.add_argument("--interp-min-confidence", type=float, default=0.45)
    p.add_argument("--interp-k", type=int, default=5)
    p.add_argument("--interp-min-neighbors", type=int, default=3)
    p.add_argument("--interp-max-distance", type=float, default=1.25)
    p.add_argument("--interp-min-coherence", type=float, default=0.70)
    args = p.parse_args()

    episodes = load_episodes(args.trace)
    ids = list(episodes)
    if len(ids) < 2:
        raise SystemExit("Need at least two independent episodes in combined trace")

    split = max(1, min(len(ids) - 1, int(round(len(ids) * args.train_fraction))))
    train_ids = ids[:split]
    test_ids = ids[split:]

    memory = InterpolatedCorrectiveTrajectoryMemory()
    train_windows = 0
    for run in train_ids:
        for history, actions, future in iter_windows(episodes[run], args.context, args.horizon):
            train_windows += 1
            memory.observe_window(history, actions, future)

    frozen = memory.stats().copy()

    test_windows = 0
    direct = 0
    interp = 0
    misses = 0
    direct_nrmse: list[float] = []
    direct_cos: list[float] = []
    interp_nrmse: list[float] = []
    interp_cos: list[float] = []
    interp_conf: list[float] = []
    interp_coherence: list[float] = []
    interp_neighbors: list[int] = []
    by_run: dict[int, dict[str, int]] = defaultdict(lambda: {"direct": 0, "interp": 0, "miss": 0})

    for run in test_ids:
        for history, actual_actions, _future in iter_windows(episodes[run], args.context, args.horizon):
            test_windows += 1
            recall = memory.recall(history, min_confidence=args.direct_min_confidence)
            if recall is not None:
                direct += 1
                by_run[run]["direct"] += 1
                n, c = metrics(recall.action_sequence, actual_actions)
                direct_nrmse.append(n)
                direct_cos.append(c)
                continue

            ir = memory.interpolate_recall(
                history,
                k=args.interp_k,
                min_neighbors=args.interp_min_neighbors,
                max_neighbor_distance=args.interp_max_distance,
                min_coherence=args.interp_min_coherence,
                min_confidence=args.interp_min_confidence,
            )
            if ir is None:
                misses += 1
                by_run[run]["miss"] += 1
                continue

            interp += 1
            by_run[run]["interp"] += 1
            n, c = metrics(ir.action_sequence, actual_actions)
            interp_nrmse.append(n)
            interp_cos.append(c)
            interp_conf.append(ir.confidence)
            interp_coherence.append(ir.coherence)
            interp_neighbors.append(ir.neighbors)

    unchanged = frozen == memory.stats()

    def mean(xs): return float(np.mean(xs)) if xs else 0.0
    def p95(xs): return float(np.percentile(xs, 95)) if xs else 0.0
    def good(ns, cs): return sum(n <= 0.25 and c >= 0.90 for n, c in zip(ns, cs))

    combined_n = direct + interp
    direct_good = good(direct_nrmse, direct_cos)
    interp_good = good(interp_nrmse, interp_cos)

    print("=" * 88)
    print("RoboCOP — V10 INTERPOLATED CORRECTIVE MEMORY / EPISODE HOLDOUT")
    print("=" * 88)
    print(f"Episodes total:                 {len(ids)}")
    print(f"Training episodes:              {train_ids}")
    print(f"Holdout episodes:               {test_ids}")
    print(f"Context / action horizon:       {args.context} / {args.horizon}")
    print(f"Training windows:               {train_windows}")
    print(f"Corrective prototypes:          {frozen['records']}")
    print(f"Confirmed corrective reflexes:  {frozen['confirmed_records']}")
    print(f"Memory unchanged in holdout:    {'PASS' if unchanged else 'FAIL'}")
    print()
    print(f"Holdout windows:                {test_windows}")
    print(f"Direct recalls:                 {direct} ({100.0*direct/test_windows if test_windows else 0.0:.2f}%)")
    print(f"Interpolated recalls:           {interp} ({100.0*interp/test_windows if test_windows else 0.0:.2f}%)")
    print(f"Combined coverage:              {combined_n} ({100.0*combined_n/test_windows if test_windows else 0.0:.2f}%)")
    print(f"MISS:                           {misses} ({100.0*misses/test_windows if test_windows else 0.0:.2f}%)")
    print()
    print("DIRECT V9")
    print(f"Action NRMSE mean/p95:          {mean(direct_nrmse):.4f} / {p95(direct_nrmse):.4f}")
    print(f"Action cosine mean:             {mean(direct_cos):.4f}")
    print(f"Good sequence agreement:        {direct_good}/{len(direct_nrmse)} ({100.0*direct_good/len(direct_nrmse) if direct_nrmse else 0.0:.2f}%)")
    print()
    print("INTERPOLATED V10")
    print(f"Interpolation confidence mean:  {mean(interp_conf):.4f}")
    print(f"Neighborhood coherence mean:    {mean(interp_coherence):.4f}")
    print(f"Neighbors mean:                 {mean(interp_neighbors):.2f}")
    print(f"Action NRMSE mean/p95:          {mean(interp_nrmse):.4f} / {p95(interp_nrmse):.4f}")
    print(f"Action cosine mean:             {mean(interp_cos):.4f}")
    print(f"Good sequence agreement:        {interp_good}/{len(interp_nrmse)} ({100.0*interp_good/len(interp_nrmse) if interp_nrmse else 0.0:.2f}%)")
    print()
    print("HOLDOUT COVERAGE BY EPISODE")
    for run in test_ids:
        windows = max(0, len(episodes[run]) - args.context - args.horizon + 1)
        d = by_run[run]["direct"]
        i = by_run[run]["interp"]
        m = by_run[run]["miss"]
        print(f"run={run:03d} windows={windows:4d} direct={d:3d} interp={i:3d} miss={m:3d} coverage={(100.0*(d+i)/windows if windows else 0.0):6.2f}%")
    print("=" * 88)


if __name__ == "__main__":
    main()
