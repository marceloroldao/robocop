#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory.dual_space_full_body_memory import DualSpaceFullBodyMemory  # noqa: E402
from memory.transition_memory import BalanceState, stability_score  # noqa: E402


def bal(row: dict) -> BalanceState:
    s = row["state"]
    return BalanceState(
        float(s["height"]),
        float(s["roll"]),
        float(s["pitch"]),
        float(s["angular_speed"]),
        float(s["vertical_speed"]),
        float(s.get("support_margin", 0.0)),
    )


def body(row: dict) -> np.ndarray:
    return np.asarray(row["full_body_state"]["vector"], dtype=float)


def load(path: Path) -> dict[int, list[dict]]:
    episodes: dict[int, list[dict]] = defaultdict(list)
    schema = None
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not row.get("state") or not row.get("full_body_state"):
                continue
            names = tuple(row["full_body_state"].get("names", []))
            if schema is None:
                schema = names
            elif names != schema:
                raise SystemExit("V11.2 full-body schema changed inside trace")
            run = int(row.get("run_id", row.get("episode", 0)))
            episodes[run].append(row)
    return dict(sorted(episodes.items()))


def windows(rows: list[dict], context: int, horizon: int):
    for i in range(context - 1, len(rows) - horizon):
        hist = rows[i - context + 1 : i + 1]
        future = rows[i + 1 : i + horizon + 1]
        yield hist, future


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--context", type=int, default=5)
    p.add_argument("--horizon", type=int, default=12)
    args = p.parse_args()

    t0 = time.time()
    eps = load(args.trace)
    ids = list(eps)
    if len(ids) < 2:
        raise SystemExit("Need at least two V11 episodes")

    split = max(1, min(len(ids) - 1, int(round(len(ids) * args.train_fraction))))
    train_ids, test_ids = ids[:split], ids[split:]

    vectors = [body(row) for run in train_ids for row in eps[run]]
    dims = int(vectors[0].size)
    memory = DualSpaceFullBodyMemory(context=args.context)
    memory.fit_scales(vectors)

    train_windows = 0
    candidates = 0
    corrective = 0

    print(
        f"[V11.2] loaded episodes={len(ids)} train={train_ids} "
        f"holdout={test_ids} channels={dims}",
        flush=True,
    )

    for run in train_ids:
        for hist, future in windows(eps[run], args.context, args.horizon):
            train_windows += 1
            bh = [bal(x) for x in hist]
            old = stability_score(bh[0], 1.0)
            now = stability_score(bh[-1], 1.0)
            if old - now < 0.01:
                continue

            candidates += 1
            fb = [bal(x) for x in future]
            scores = np.asarray([stability_score(x, 1.0) for x in fb], dtype=float)
            j = int(np.argmax(scores))
            gain = float(scores[j] - now)
            if gain < 0.03:
                continue

            corrective += 1
            memory.observe([body(x) for x in hist], body(future[j]), gain)
            if corrective % 250 == 0:
                print(
                    f"[V11.2] train corrective={corrective} "
                    f"prototypes={memory.size} elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )

    frozen = memory.stats().copy()

    holdout = 0
    recalls = 0
    direct = 0
    interpolated = 0
    confidence: list[float] = []
    coherence: list[float] = []
    sensory_rms: list[float] = []
    body_rms: list[float] = []
    sensory_max: list[float] = []
    body_max: list[float] = []
    direction: list[float] = []
    target_distance: list[float] = []
    by_run = defaultdict(lambda: [0, 0, 0])

    for run in test_ids:
        print(f"[V11.2] holdout run={run} start", flush=True)
        for hist, future in windows(eps[run], args.context, args.horizon):
            holdout += 1
            by_run[run][0] += 1

            recall, diag = memory.recall_with_diagnostics([body(x) for x in hist])
            if recall is None:
                continue

            recalls += 1
            direct += int(recall.direct)
            interpolated += int(not recall.direct)
            by_run[run][1] += 1
            by_run[run][2] += int(not recall.direct)

            confidence.append(float(recall.confidence))
            coherence.append(float(recall.coherence))
            sensory_rms.append(float(diag.sensory_rms))
            body_rms.append(float(diag.body_rms))
            sensory_max.append(float(diag.sensory_max))
            body_max.append(float(diag.body_max))

            current = body(hist[-1])
            future_balance = [bal(x) for x in future]
            scores = np.asarray(
                [stability_score(x, 1.0) for x in future_balance],
                dtype=float,
            )
            j = int(np.argmax(scores))
            true_target = body(future[j])

            predicted_delta = (recall.target_state - current) / memory.scales
            true_delta = (true_target - current) / memory.scales
            nx = float(np.linalg.norm(predicted_delta))
            ny = float(np.linalg.norm(true_delta))
            if nx > 0.0 and ny > 0.0:
                direction.append(
                    float(np.dot(predicted_delta, true_delta) / (nx * ny))
                )
            else:
                direction.append(0.0)

            normalized_target_error = (
                (recall.target_state - true_target) / memory.scales
            )
            target_distance.append(
                float(np.sqrt(np.mean(normalized_target_error ** 2)))
            )

        print(
            f"[V11.2] holdout run={run} done "
            f"recalls={by_run[run][1]}/{by_run[run][0]} "
            f"elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    p95 = lambda xs: float(np.percentile(xs, 95)) if xs else 0.0
    aligned = sum(x >= 0.70 for x in direction)
    index_stats = memory.index_stats()

    print("=" * 96)
    print("RoboCOP — V11.2 COUPLED SENSORY/BODY SPACES / EPISODE HOLDOUT")
    print("=" * 96)
    print(f"Episodes total:                   {len(ids)}")
    print(f"Training episodes:                {train_ids}")
    print(f"Holdout episodes:                 {test_ids}")
    print("Sensory channels:                 16")
    print(f"Corporal channels:                {dims - 16}")
    print(f"Coupled trajectory dimensions:    {dims * args.context}")
    print(f"Training windows:                 {train_windows}")
    print(f"Candidate degrading windows:      {candidates}")
    print(f"Corrective windows admitted:      {corrective}")
    print(f"Prototypes:                       {frozen['records']}")
    print(f"Confirmed prototypes:             {frozen['confirmed_records']}")
    print(
        f"Memory unchanged in holdout:      "
        f"{'PASS' if frozen == memory.stats() else 'FAIL'}"
    )
    print(f"Index mean candidates/query:      {index_stats['mean_candidates']:.2f}")
    print(f"Analysis elapsed seconds:         {time.time() - t0:.2f}")
    print()
    print(f"Holdout windows:                  {holdout}")
    print(
        f"Recalls:                          {recalls} "
        f"({100 * recalls / holdout if holdout else 0:.2f}%)"
    )
    print(f"Direct / interpolated:            {direct} / {interpolated}")
    print(
        f"MISS:                             {holdout - recalls} "
        f"({100 * (holdout - recalls) / holdout if holdout else 0:.2f}%)"
    )
    print(f"Recall confidence mean:           {mean(confidence):.4f}")
    print(f"Neighborhood coherence mean:      {mean(coherence):.4f}")
    print(f"Sensory RMS mean/p95:             {mean(sensory_rms):.4f} / {p95(sensory_rms):.4f}")
    print(f"Corporal RMS mean/p95:            {mean(body_rms):.4f} / {p95(body_rms):.4f}")
    print(f"Sensory max error mean/p95:       {mean(sensory_max):.4f} / {p95(sensory_max):.4f}")
    print(f"Corporal max error mean/p95:      {mean(body_max):.4f} / {p95(body_max):.4f}")
    print(f"Recovery direction cosine mean:   {mean(direction):.4f}")
    print(
        f"Direction aligned >=0.70:         {aligned}/{len(direction)} "
        f"({100 * aligned / len(direction) if direction else 0:.2f}%)"
    )
    print(
        f"Target distance mean/p95:         "
        f"{mean(target_distance):.4f} / {p95(target_distance):.4f}"
    )
    print()
    print("HOLDOUT COVERAGE BY EPISODE")
    for run in test_ids:
        w, r, i = by_run[run]
        rate = 100.0 * r / w if w else 0.0
        print(
            f"run={run:03d} windows={w:5d} recalls={r:5d} "
            f"interp={i:5d} rate={rate:6.2f}%"
        )
    print("=" * 96)


if __name__ == "__main__":
    main()
