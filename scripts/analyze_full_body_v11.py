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

from memory.full_body_trajectory_memory import FullBodyTrajectoryMemory  # noqa: E402
from memory.transition_memory import BalanceState, stability_score  # noqa: E402


def balance(row: dict) -> BalanceState:
    s = row["state"]
    return BalanceState(float(s["height"]), float(s["roll"]), float(s["pitch"]),
                        float(s["angular_speed"]), float(s["vertical_speed"]),
                        float(s.get("support_margin", 0.0)))


def body(row: dict) -> np.ndarray:
    return np.asarray(row["full_body_state"]["vector"], dtype=float)


def load(path: Path) -> dict[int, list[dict]]:
    episodes = defaultdict(list)
    schema = None
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("state") or not r.get("full_body_state"):
                continue
            names = tuple(r["full_body_state"].get("names", []))
            if schema is None:
                schema = names
            elif names != schema:
                raise SystemExit("V11 full-body schema changed inside trace")
            episodes[int(r.get("run_id", r.get("episode", 0)))].append(r)
    return dict(sorted(episodes.items()))


def windows(rows: list[dict], context: int, horizon: int):
    for i in range(context - 1, len(rows) - horizon):
        hist = rows[i-context+1:i+1]
        future = rows[i+1:i+horizon+1]
        yield hist, future


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, default=ROOT / "results/v11_multi_episode/combined_trace.jsonl")
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--context", type=int, default=5)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--min-degradation", type=float, default=0.01)
    p.add_argument("--min-recovery", type=float, default=0.03)
    args = p.parse_args()

    eps = load(args.trace)
    ids = list(eps)
    if len(ids) < 2:
        raise SystemExit("Need at least two V11 episodes")
    split = max(1, min(len(ids)-1, int(round(len(ids)*args.train_fraction))))
    train_ids, test_ids = ids[:split], ids[split:]

    all_train_vectors = [body(r) for run in train_ids for r in eps[run]]
    dims = int(all_train_vectors[0].size)
    memory = FullBodyTrajectoryMemory(context=args.context)
    memory.fit_scales(all_train_vectors)

    train_windows = candidates = corrective = 0
    for run in train_ids:
        for hist, future in windows(eps[run], args.context, args.horizon):
            train_windows += 1
            b_hist = [balance(x) for x in hist]
            old = stability_score(b_hist[0], 1.0)
            now = stability_score(b_hist[-1], 1.0)
            if old - now < args.min_degradation:
                continue
            candidates += 1
            bs = [balance(x) for x in future]
            scores = np.asarray([stability_score(x, 1.0) for x in bs])
            j = int(np.argmax(scores))
            gain = float(scores[j] - now)
            if gain < args.min_recovery:
                continue
            corrective += 1
            memory.observe([body(x) for x in hist], body(future[j]), gain)

    frozen = memory.stats().copy()
    holdout = recalls = direct = interp = 0
    conf = []; coh = []; rms = []; mx = []; den = []; fac = []
    direction = []; target_dist = []; predicted_gain = []; observed_gain = []
    by = defaultdict(lambda: [0, 0, 0])

    for run in test_ids:
        for hist, future in windows(eps[run], args.context, args.horizon):
            holdout += 1; by[run][0] += 1
            rr = memory.recall([body(x) for x in hist])
            if rr is None:
                continue
            recalls += 1; direct += int(rr.direct); interp += int(not rr.direct)
            by[run][1] += 1; by[run][2] += int(not rr.direct)
            conf.append(rr.confidence); coh.append(rr.coherence); rms.append(rr.rms_distance)
            mx.append(rr.max_channel_error); den.append(rr.local_density); fac.append(rr.resolution_factor)

            current = body(hist[-1])
            future_bal = [balance(x) for x in future]
            scores = np.asarray([stability_score(x, 1.0) for x in future_bal])
            j = int(np.argmax(scores)); true_target = body(future[j])
            x = (rr.target_state-current)/memory.scales
            y = (true_target-current)/memory.scales
            nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
            direction.append(float(np.dot(x,y)/(nx*ny)) if nx > 0 and ny > 0 else 0.0)
            target_dist.append(float(np.sqrt(np.mean(((rr.target_state-true_target)/memory.scales)**2))))

            now = stability_score(balance(hist[-1]), 1.0)
            # Map target's first channels back to legacy balance channels only for gain calibration.
            # full-body vector: global z=2, euler roll/pitch=7/8, gyro=10:13. vertical speed is not raw.
            pred_height = float(rr.target_state[2])
            pred_roll = float(np.deg2rad(rr.target_state[7]))
            pred_pitch = float(np.deg2rad(rr.target_state[8]))
            pred_omega = float(np.linalg.norm(np.deg2rad(rr.target_state[10:13])))
            pred_bal = BalanceState(pred_height, pred_roll, pred_pitch, pred_omega, 0.0, 0.0)
            predicted_gain.append(float(stability_score(pred_bal,1.0)-now))
            observed_gain.append(float(scores[j]-now))

    after = memory.stats()
    mean = lambda x: float(np.mean(x)) if x else 0.0
    p95 = lambda x: float(np.percentile(x,95)) if x else 0.0
    aligned = sum(x >= 0.70 for x in direction)

    print("="*94)
    print("RoboCOP — V11 FULL BODY SENSOR ADDRESS / EPISODE HOLDOUT")
    print("="*94)
    print(f"Episodes total:                   {len(ids)}")
    print(f"Training episodes:                {train_ids}")
    print(f"Holdout episodes:                 {test_ids}")
    print(f"Raw sensor channels:              {dims}")
    print(f"Trajectory address dimensions:    {dims*args.context} ({args.context} x {dims})")
    print(f"Training windows:                 {train_windows}")
    print(f"Candidate degrading windows:      {candidates}")
    print(f"Corrective windows admitted:      {corrective}")
    print(f"Full-body prototypes:             {frozen['records']}")
    print(f"Confirmed prototypes:             {frozen['confirmed_records']}")
    print(f"Merged observations:              {frozen['merged']}")
    print(f"Max confirmations:                {frozen['max_confirmations']}")
    print(f"Memory unchanged in holdout:      {'PASS' if frozen == after else 'FAIL'}")
    print()
    print(f"Holdout windows:                  {holdout}")
    print(f"Full-body recalls:                {recalls} ({100*recalls/holdout if holdout else 0:.2f}%)")
    print(f"Direct / interpolated:            {direct} / {interp}")
    print(f"MISS:                             {holdout-recalls} ({100*(holdout-recalls)/holdout if holdout else 0:.2f}%)")
    print(f"Recall confidence mean:           {mean(conf):.4f}")
    print(f"Neighborhood coherence mean:      {mean(coh):.4f}")
    print(f"Trajectory RMS mean/p95:          {mean(rms):.4f} / {p95(rms):.4f}")
    print(f"Max channel error mean/p95:       {mean(mx):.4f} / {p95(mx):.4f}")
    print(f"Local density mean/p95:           {mean(den):.2f} / {p95(den):.2f}")
    print(f"Resolution factor mean/p95:       {mean(fac):.4f} / {p95(fac):.4f}")
    print(f"Recovery direction cosine mean:   {mean(direction):.4f}")
    print(f"Direction aligned >=0.70:         {aligned}/{len(direction)} ({100*aligned/len(direction) if direction else 0:.2f}%)")
    print(f"Target distance mean/p95:         {mean(target_dist):.4f} / {p95(target_dist):.4f}")
    print(f"Predicted gain mean:              {mean(predicted_gain):.4f}")
    print(f"Observed best gain mean:          {mean(observed_gain):.4f}")
    print()
    print("HOLDOUT COVERAGE BY EPISODE")
    for run in test_ids:
        w,r,i = by[run]
        print(f"run={run:03d} windows={w:4d} recalls={r:4d} interp={i:4d} rate={(100*r/w if w else 0):6.2f}%")
    print("="*94)


if __name__ == "__main__":
    main()
