#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory.corrective_trajectory_memory import CorrectiveTrajectoryMemory  # noqa: E402
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


def main() -> None:
    p = argparse.ArgumentParser(description="RoboCOP V9 corrective-trajectory replay with episode holdout")
    p.add_argument("--trace", type=Path, default=ROOT / "results/bahiart_multi_episode/combined_trace.jsonl")
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--context", type=int, default=5)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--min-confidence", type=float, default=0.60)
    args = p.parse_args()

    episodes = load_episodes(args.trace)
    ids = list(episodes)
    if len(ids) < 2:
        raise SystemExit("Need at least two independent episodes in combined trace")

    split = max(1, min(len(ids) - 1, int(round(len(ids) * args.train_fraction))))
    train_ids = ids[:split]
    test_ids = ids[split:]

    memory = CorrectiveTrajectoryMemory()
    train_windows = 0
    for run in train_ids:
        for history, actions, future in iter_windows(episodes[run], args.context, args.horizon):
            train_windows += 1
            memory.observe_window(history, actions, future)

    frozen = memory.stats().copy()
    test_windows = 0
    recalls = 0
    recall_by_run: dict[int, int] = defaultdict(int)
    confidence = []
    distance = []
    action_nrmse = []
    action_cosine = []

    for run in test_ids:
        for history, actual_actions, _future in iter_windows(episodes[run], args.context, args.horizon):
            test_windows += 1
            recall = memory.recall(history, min_confidence=args.min_confidence)
            if recall is None:
                continue
            recalls += 1
            recall_by_run[run] += 1
            confidence.append(recall.confidence)
            distance.append(recall.distance)
            predicted = recall.action_sequence
            if predicted.shape != actual_actions.shape:
                continue
            denom = max(1.0, float(np.sqrt(np.mean(actual_actions * actual_actions))))
            nrmse = float(np.sqrt(np.mean((predicted - actual_actions) ** 2)) / denom)
            action_nrmse.append(nrmse)
            a = predicted.reshape(-1)
            b = actual_actions.reshape(-1)
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            action_cosine.append(float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0)

    after = memory.stats()
    unchanged = frozen == after

    def mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    def p95(xs):
        return float(np.percentile(xs, 95)) if xs else 0.0

    good = 0
    if action_nrmse:
        good = sum(n <= 0.25 and c >= 0.90 for n, c in zip(action_nrmse, action_cosine))

    print("=" * 84)
    print("RoboCOP — V9 CORRECTIVE TRAJECTORY MEMORY / EPISODE HOLDOUT")
    print("=" * 84)
    print(f"Episodes total:                {len(ids)}")
    print(f"Training episodes:             {train_ids}")
    print(f"Holdout episodes:              {test_ids}")
    print(f"Context / action horizon:      {args.context} / {args.horizon}")
    print(f"Training windows:              {train_windows}")
    print(f"Candidate degrading windows:   {frozen['candidates']}")
    print(f"Corrective windows admitted:   {frozen['admitted']}")
    print(f"Corrective prototypes:         {frozen['records']}")
    print(f"Confirmed corrective reflexes: {frozen['confirmed_records']}")
    print(f"Merged observations:           {frozen['merged']}")
    print(f"Mean confirmations:            {frozen['mean_confirmations']:.3f}")
    print(f"Max confirmations:             {frozen['max_confirmations']}")
    print(f"Memory unchanged in holdout:   {'PASS' if unchanged else 'FAIL'}")
    print()
    print(f"Holdout windows:               {test_windows}")
    print(f"Corrective recalls:            {recalls}")
    print(f"Recall rate:                   {(100.0*recalls/test_windows if test_windows else 0.0):.2f}%")
    print(f"Recall confidence mean:        {mean(confidence):.4f}")
    print(f"Recall distance mean/p95:      {mean(distance):.4f} / {p95(distance):.4f}")
    print(f"Action NRMSE mean/p95:         {mean(action_nrmse):.4f} / {p95(action_nrmse):.4f}")
    print(f"Action cosine mean:            {mean(action_cosine):.4f}")
    print(f"Good sequence agreement:       {good}/{len(action_nrmse)} ({100.0*good/len(action_nrmse) if action_nrmse else 0.0:.2f}%)")
    print()
    print("HOLDOUT RECALLS BY EPISODE")
    for run in test_ids:
        windows = max(0, len(episodes[run]) - args.context - args.horizon + 1)
        r = recall_by_run.get(run, 0)
        print(f"run={run:03d} windows={windows:4d} recalls={r:4d} rate={(100.0*r/windows if windows else 0.0):6.2f}%")
    print("=" * 84)


if __name__ == "__main__":
    main()
